"""Differentiable rigid drone + free-pivot DDER cable with optional residuals.

The cable root is eliminated kinematically, not pinned to a prescribed path.
Its mass contributes to the generalized mass matrix exactly once. Cable damping
and length-constraint impulses act on translation AND body rotation. Position
projection is a finite-step approximation; convergence must be checked.
"""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import torch

from .dder_math import analytic_isotropic_bending_force, _curvature_rate_jacobian_impl
from .rotations import rotation_exp, skew


@dataclass(frozen=True)
class CoupledState:
    position_com: torch.Tensor
    rotation: torch.Tensor
    velocity_com: torch.Tensor
    omega_body: torch.Tensor
    free_positions: torch.Tensor
    free_velocities: torch.Tensor


class CoupledDynamics:
    def __init__(self, config, *, device="cpu", batched=False, residual=None):
        self.config = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else config
        self.device = device
        self.batched = batched
        drone, cable, numeric = (self.config[k] for k in ("drone", "cable", "simulation"))
        self.mass = self.tensor(drone["mass_kg"])
        self.inertia = self.tensor(drone["inertia_body_about_com_kg_m2"])
        self.com_tracking = self.tensor(drone["center_of_mass_tracking_m"])
        self.attachment_tracking = self.tensor(self.config["attachment"]["position_tracking_m"])
        self.arm = self.attachment_tracking - self.com_tracking
        if (self.inertia.shape != (3, 3) or self.com_tracking.shape != (3,)
                or self.attachment_tracking.shape != (3,)):
            raise ValueError("Supply COM/attachment XYZ and a 3x3 inertia matrix")
        if self.config["attachment"]["type"] != "free_pivot" or self.config["attachment"]["transmit_direct_moment"]:
            raise ValueError("Only a free-pivot attachment is implemented")
        if (not bool(torch.isfinite(self.inertia).all()) or not torch.allclose(self.inertia,self.inertia.T)
                or float(torch.linalg.eigvalsh(self.inertia).min()) <= 0):
            raise ValueError("Inertia must be finite, symmetric and positive definite")
        principal = torch.linalg.eigvalsh(self.inertia)
        if float(2 * principal[-1] - principal.sum()) > 1e-12:
            raise ValueError("Inertia principal moments violate the triangle inequality")
        counts = np.asarray(numeric["segments_per_marker_interval"])
        lengths = np.asarray(cable["marker_interval_lengths_m"], dtype=float)
        markers = np.asarray(cable["moving_marker_masses_kg"], dtype=float)
        if (lengths.ndim != 1 or not lengths.size or counts.shape != lengths.shape
                or markers.shape != lengths.shape or not np.isfinite(lengths).all()
                or not np.isfinite(markers).all() or np.any(lengths <= 0) or np.any(markers < 0)
                or np.any(counts != counts.astype(int)) or np.any(counts < 1)):
            raise ValueError("Cable intervals, subdivisions and marker masses must agree")
        self.marker_indices = np.r_[0, np.cumsum(counts)].astype(int)
        rest = np.repeat(lengths / counts, counts.astype(int))
        bare_mass = float(cable["total_mass_kg"]) - markers.sum()
        if not np.isfinite(bare_mass) or bare_mass <= 0 or not 0 < float(self.mass) < float('inf'):
            raise ValueError("Positive drone and bare-cable masses are required")
        masses = np.r_[rest[0], rest[:-1] + rest[1:], rest[-1]] * (0.5 * bare_mass / rest.sum())
        masses[self.marker_indices[1:]] += markers
        self.rest = self.tensor(rest)
        self.node_mass = self.tensor(masses)
        self.dual = (self.rest[:-1] + self.rest[1:]) / 2
        self.ei = self.tensor(cable["EI_n_m2"])
        self.cb = self.tensor(cable["Cb_n_m2_s"])
        self.drag = self.tensor(cable["external_drag_s_inv"])
        self.gravity = self.tensor(numeric["gravity_world_m_s2"])
        self.dt = float(numeric["timestep_s"])
        self.substeps = int(numeric["cable_substeps"])
        self.iterations = int(numeric["constraint_iterations"])
        self.regularization = float(numeric["curvature_frame_regularization"])
        if (not np.isfinite([self.dt,self.regularization]).all() or self.dt <= 0
                or self.substeps != numeric["cable_substeps"] or self.iterations != numeric["constraint_iterations"]
                or self.substeps < 1 or self.iterations < 1 or self.regularization <= 0
                or self.gravity.shape != (3,) or not bool(torch.isfinite(self.gravity).all())
                or not bool(torch.isfinite(self.com_tracking).all())
                or not bool(torch.isfinite(self.attachment_tracking).all())):
            raise ValueError("Invalid integration or geometry parameters")
        for value in (self.ei, self.cb, self.drag):
            if value.numel()!=1 or not bool(torch.isfinite(value)) or float(value)<0:
                raise ValueError("Cable stiffness/damping/drag must be finite and nonnegative")
        self.n = len(masses)
        self.size = 6 + 3 * (self.n - 1)
        self.eye3 = torch.eye(3, dtype=torch.float64, device=device)
        self.free_mass_matrix = torch.diag(self.node_mass[1:].repeat_interleave(3))
        # Maps node displacements to edge displacements.
        incidence = torch.eye(self.n, device=device, dtype=torch.float64)[1:] - torch.eye(self.n, device=device, dtype=torch.float64)[:-1]
        self.edge_map = torch.kron(incidence, self.eye3)
        # These depend only on fixed geometry/masses, not the six fit parameters.
        # State-dependent maps, damping and constraint systems remain recomputed.
        self.arm_skew = skew(self.arm)
        self.root_translation_mass = self.mass + self.node_mass[0]
        # Eliminating translation leaves this constant body-frame inertia.
        # R.T @ R = I for the rigid-body rotation; masses/geometry are fixed
        # during a fit. Rebuild this cache when constructing a new model.
        root_reduced_mass = self.mass*self.node_mass[0]/self.root_translation_mass
        root_inertia = self.inertia + root_reduced_mass*(self.arm_skew.T @ self.arm_skew)
        self.root_inertia_inverse = torch.linalg.solve(root_inertia, self.eye3)
        self.root_padding = self.eye3.new_zeros(3, self.size-6)
        self.free_map = torch.cat((self.eye3.new_zeros(self.size-6, 6),
                                  torch.eye(self.size-6, device=device, dtype=torch.float64)), dim=1)
        self.base_mass = torch.block_diag(self.mass*self.eye3, self.inertia, self.free_mass_matrix)
        self.node_mass_xyz = self.node_mass.repeat_interleave(3)[:, None]
        self.cable_gravity = self.node_mass[:, None]*self.gravity
        self.drone_gravity = self.mass*self.gravity
        self.free_force_zeros = self.eye3.new_zeros(self.size-6)
        self.edge_blocks = self.edge_map.reshape(self.n-1, 3, -1)
        self.arc_fraction = self.rest.cumsum(0)/self.rest.sum()
        self.tip_mask = (torch.arange(self.n-1,device=device)==self.n-2).to(torch.float64)
        self.residual = residual
        reference = self.config.get('residual', {})
        if self.residual is None and reference.get('enabled', False):
            from .residuals import load_residual
            self.residual = load_residual(reference, device)

    def tensor(self, value):
        if value is None:
            raise ValueError("Physical parameters must be supplied")
        return torch.as_tensor(value, dtype=torch.float64, device=self.device)

    @staticmethod
    def _solve(matrix, rhs):
        if not matrix.is_cuda:
            return torch.linalg.solve(matrix, rhs)
        # Avoid host synchronization in CUDA graphs and batched fitting.
        # Failed factorizations become nonfinite states, rejected by callers.
        result, info = torch.linalg.solve_ex(matrix, rhs, check_errors=False)
        return torch.where(info == 0, result, torch.full_like(result, float('nan')))

    def _mass_solve(self, matrix, rhs):
        """Block elimination with constant body-frame Schur complement."""
        coupling = matrix[:3,3:6]
        translation = rhs[:3]/self.root_translation_mass
        angular = self.root_inertia_inverse @ (rhs[3:6]-coupling.T @ translation)
        body = torch.cat((translation-coupling @ angular/self.root_translation_mass, angular), dim=0)
        free = rhs[6:] / self.node_mass_xyz[3:]
        return torch.cat((body, free), dim=0)

    def _mass_solve_reference(self, matrix, rhs):
        """Exact block solve: coupled six-DOF body, diagonal free-node masses."""
        body = self._solve(matrix[:6,:6], rhs[:6])
        free = rhs[6:] / self.node_mass_xyz[3:]
        return torch.cat((body, free), dim=0)

    def _chain_solve(self, matrix, rhs):
        """Solve the SPD tridiagonal mass-projection constraint system.

        Adjacent cable edges share one node; the body affects only the first
        edge. This does NOT apply to the dense implicit-damping constraints.
        Functional elimination preserves autograd/vmap support. Invalid pivots
        yield nonfinite output, rejected by the existing runtime/fit checks.
        """
        # Scalar tensor elimination needs fusion to be efficient. Preserve
        # LAPACK on CPU and the existing uncompiled GPU reference path.
        if not matrix.is_cuda or not torch.compiler.is_compiling():
            return self._solve(matrix, rhs)
        diagonal = matrix.diagonal()
        lower = matrix.diagonal(offset=-1)
        upper = matrix.diagonal(offset=1)
        pivots = [diagonal[0]]
        values = [rhs[0]]
        for i in range(1, self.n-1):
            ratio = lower[i-1]/pivots[-1]
            pivots.append(diagonal[i]-ratio*upper[i-1])
            values.append(rhs[i]-ratio*values[-1])
        solution = [values[-1]/pivots[-1]]
        for i in range(self.n-3, -1, -1):
            solution.append((values[i]-upper[i]*solution[-1])/pivots[i])
        result = torch.stack(solution[::-1])
        valid = (torch.stack(pivots) > 0).all()
        return torch.where(valid, result, torch.full_like(result, float('nan')))

    def initial_state(self, tracking_position=(0.,0.,2.), *, rotation=None,
                      tracking_velocity=(0.,0.,0.), omega_body=(0.,0.,0.),
                      cable_positions=None, cable_velocities=None):
        """Build an owned state without projecting supplied geometry or velocities.

        Drone P/V refer to the tracking origin in world coordinates; rotation is
        body-to-world and angular velocity is body-frame rad/s. Cable arrays are
        world-frame (n,3), INCLUDING attachment node 0. Defaults give a vertical
        cable translating at the root velocity. Constraint tolerances are 1e-6 m
        for positions/lengths and 1e-6 m/s for root velocity/segment length rates.
        """
        def supplied(value, shape, name):
            tensor = self.tensor(value).clone()
            if tensor.shape != shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be finite with shape {shape}")
            return tensor

        R = self.eye3.clone() if rotation is None else supplied(rotation, (3,3), "Initial rotation")
        if (R.shape != (3,3) or not bool(torch.isfinite(R).all())
                or not torch.allclose(R.T@R,self.eye3,atol=1e-8,rtol=0)
                or abs(float(torch.det(R))-1)>1e-8):
            raise ValueError("Initial orientation must be a proper rotation")
        tracking = supplied(tracking_position, (3,), "Initial tracking position")
        tracking_v = supplied(tracking_velocity, (3,), "Initial tracking velocity")
        omega = supplied(omega_body, (3,), "Initial body angular velocity")
        position = tracking + R @ self.com_tracking
        root = position + R @ self.arm
        com_v = tracking_v + R @ torch.linalg.cross(omega, self.com_tracking)
        root_v = com_v + R @ torch.linalg.cross(omega, self.arm)
        if cable_positions is None:
            q = root[None].expand(self.n,3).clone()
            q[1:,2] -= self.rest.cumsum(0)
        else:
            q = supplied(cable_positions, (self.n,3), "Initial cable positions")
        v = (root_v[None].expand(self.n,3).clone() if cable_velocities is None else
             supplied(cable_velocities, (self.n,3), "Initial cable velocities"))
        if float(torch.linalg.vector_norm(q[0]-root)) > 1e-6:
            raise ValueError("Initial cable node 0 must match the drone attachment (within 1e-6 m)")
        if float(torch.linalg.vector_norm(v[0]-root_v)) > 1e-6:
            raise ValueError("Initial cable node 0 velocity must match the attachment (within 1e-6 m/s)")
        # The root is kinematically determined by the drone. Check constraints
        # against that exact root, not an independently supplied approximation.
        edges = q[1:]-torch.cat((root[None], q[1:-1]))
        lengths = torch.linalg.vector_norm(edges, dim=1)
        if float((lengths-self.rest).abs().max()) > 1e-6 or bool((lengths <= 0).any()):
            raise ValueError("Initial cable segment lengths must match the model (within 1e-6 m)")
        relative_v = v[1:]-torch.cat((root_v[None], v[1:-1]))
        length_rates = ((edges/lengths[:,None])*relative_v).sum(dim=1)
        if not bool(torch.isfinite(length_rates).all()) or float(length_rates.abs().max()) > 1e-6:
            raise ValueError("Initial cable velocities must preserve segment lengths (within 1e-6 m/s)")
        return CoupledState(position, R, com_v, omega, q[1:], v[1:])

    def tracking_pose(self, state):
        arm = -self.com_tracking
        return (state.position_com + state.rotation @ arm,
                state.velocity_com + state.rotation @ torch.linalg.cross(state.omega_body, arm))

    def cable_state(self, state):
        root = state.position_com + state.rotation @ self.arm
        root_v = state.velocity_com + state.rotation @ torch.linalg.cross(state.omega_body,self.arm)
        return (torch.cat((root[None],state.free_positions)),
                torch.cat((root_v[None],state.free_velocities)))

    def _maps(self, state):
        boundary = torch.cat((self.eye3,-state.rotation @ self.arm_skew),dim=1)
        root_map = torch.cat((boundary,self.root_padding),dim=1)
        D = torch.cat((root_map,self.free_map))
        # Only the six body coordinates share the attachment's inertia.
        # Preserve the multiplication order; do not multiply the zero padding.
        body = self.base_mass[:6,:6] + (self.node_mass[0]*boundary.T) @ boundary
        M = torch.cat((torch.cat((body,self.base_mass[:6,6:]),dim=1),self.base_mass[6:]),dim=0)
        return D,M

    def _projection_maps(self, state):
        """Only the attachment and body blocks consumed by mass projection."""
        boundary = torch.cat((self.eye3,-state.rotation @ self.arm_skew),dim=1)
        # Keep _maps' body arithmetic; omit its unused full-matrix padding.
        body = self.base_mass[:6,:6] + (self.node_mass[0]*boundary.T) @ boundary
        return boundary,body

    @staticmethod
    def _map_product(matrix, D):
        """Apply the attachment map and copy the free-node identity block."""
        return torch.cat((matrix[:,:3] @ D[:3,:6],matrix[:,3:]),dim=1)

    def _drag_matrix(self, D):
        """Exact nodal-mass metric, including attachment translation/rotation."""
        boundary = D[:3,:6]
        body = boundary.T @ (self.node_mass[0]*boundary)
        return torch.cat((torch.cat((body,self.base_mass[:6,6:]),dim=1),self.base_mass[6:]),dim=0)

    def _constraints(self, state, D):
        q,_ = self.cable_state(state)
        edges = q[1:]-q[:-1]
        lengths = edges.norm(dim=1).clamp_min(1e-12)
        direction = edges/lengths[:,None]
        J = (direction[:,:,None] * self.edge_blocks).sum(1)
        return lengths-self.rest,self._map_product(J,D)

    @staticmethod
    def _velocity(state):
        return torch.cat((state.velocity_com,state.omega_body,state.free_velocities.flatten()))

    @staticmethod
    def _with_velocity(state, velocity):
        return replace(state,velocity_com=velocity[:3],omega_body=velocity[3:6],
                       free_velocities=velocity[6:].reshape(-1,3))

    @staticmethod
    def _displace(state, delta):
        return replace(state,position_com=state.position_com+delta[:3],
                       rotation=state.rotation @ rotation_exp(delta[3:6]),
                       free_positions=state.free_positions+delta[6:].reshape(-1,3))

    def substep(self, state, thrust_n, torque_body_nm, h):
        """Semi-implicit damping followed by coupled position/velocity constraints.

        All root/cable masses participate in the solve; there is no externally
        imposed root trajectory or separate, delayed reaction-force injection.
        """
        q,_ = self.cable_state(state)
        D,M = self._maps(state)
        bending = analytic_isotropic_bending_force(q[None],self.ei.reshape(1),self.dual)[0]
        cable_force = bending + self.cable_gravity
        if self.residual is not None:
            extra = self.node_mass[1:,None]*self.residual.cable_acceleration(self,state)
            cable_force = cable_force + torch.cat((torch.zeros_like(extra[:1]),extra))
        body_force = self.drone_gravity + self.tensor(thrust_n)*state.rotation[:,2]
        gyro = torch.linalg.cross(state.omega_body,self.inertia @ state.omega_body)
        force = torch.cat((body_force,self.tensor(torque_body_nm)-gyro,
                           self.free_force_zeros)) + D.T @ cable_force.flatten()
        centripetal = state.rotation @ torch.linalg.cross(state.omega_body,
                                      torch.linalg.cross(state.omega_body,self.arm))
        force = force-self.node_mass[0]*D[:3].T @ centripetal
        J,_,weight = _curvature_rate_jacobian_impl(q[None],self.rest,None,None,
                          frame_regularization=self.regularization,vectorized=True,
                          functional_assembly=self.batched)
        rate = self._map_product(J[0],D)
        damping = self.cb*(rate.T @ (weight[0,:,None]*rate))
        damping = damping + self.drag*self._drag_matrix(D)
        system = M+h*damping
        # Enforce constraints inside the implicit damping solve. Damping an
        # unconstrained falling cable before projecting would generate a
        # spurious net drag force even in stationary hover.
        _,G = self._constraints(state,D)
        # One factorization with both RHS blocks, instead of solving the same
        # implicit matrix twice. The constraint system remains unchanged.
        solution = self._solve(system, torch.cat(((M @ self._velocity(state)+h*force)[:,None],G.T),dim=1))
        u, inverse_j = solution[:,0], solution[:,1:]
        u = u-inverse_j @ self._solve(G @ inverse_j,G @ u)
        next_state = self._displace(state,h*u)
        for _ in range(self.iterations):
            Dn,Mn = self._projection_maps(next_state)
            error,G = self._constraints(next_state,Dn)
            inverse_j = self._mass_solve(Mn,G.T)
            delta = inverse_j @ self._chain_solve(G @ inverse_j,-error)
            next_state = self._displace(next_state,delta)
            u = u+delta/h
        Dn,Mn = self._projection_maps(next_state)
        _,G = self._constraints(next_state,Dn)
        inverse_j = self._mass_solve(Mn,G.T)
        u = u-inverse_j @ self._chain_solve(G @ inverse_j,G @ u)
        return self._with_velocity(next_state,u)

    def step(self, state, command, controller):
        """One output interval; caller supplies the already-delayed command."""
        command = self.tensor(command)
        if command.shape != (11,) or not bool(torch.isfinite(command).all()):
            raise ValueError("Expected finite P/V/A/yaw/yaw-rate command")
        if float(command[10]) != 0:
            raise ValueError("This first controller has no yaw-rate feedforward")
        return self.step_unchecked(state, command, controller)

    def step_unchecked(self, state, command, controller):
        """Same integrator for prevalidated inputs (CUDA capture / vmap)."""
        h = self.dt/self.substeps
        for _ in range(self.substeps):
            thrust,torque = controller(state,command)
            state = self.substep(state,thrust,torque,h)
        return state
