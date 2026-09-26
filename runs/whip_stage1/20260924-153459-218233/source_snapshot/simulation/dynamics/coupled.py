"""Differentiable rigid drone + free-pivot DDER cable with optional residuals.

The cable root is eliminated kinematically, not pinned to a prescribed path.
Its mass contributes to the generalized mass matrix exactly once. Cable forces
and length-constraint impulses act on translation AND body rotation. Position
projection is a finite-step approximation; convergence must be checked.
The naturally straight isotropic cable retains centerline bending elasticity
and external drag, without independent material spin or internal bending damping.
"""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import torch

from .dder_math import analytic_isotropic_bending_force
from .coupled_control import CoupledController
from .residuals import CoupledResidual
from .rotations import rotation_exp, skew
from simulation.timing import integration_steps


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
        self.drag = self.tensor(cable["external_drag_s_inv"])
        self.gravity = self.tensor(numeric["gravity_world_m_s2"])
        self.dt = float(numeric["timestep_s"])
        if 'physics_timestep_s' in numeric:
            if 'cable_substeps' in numeric:
                raise ValueError('Use physics_timestep_s or legacy cable_substeps, not both')
            self.physics_dt = float(numeric['physics_timestep_s'])
        else:
            legacy_steps = numeric['cable_substeps']
            if isinstance(legacy_steps, bool) or int(legacy_steps) != legacy_steps or legacy_steps < 1:
                raise ValueError('Legacy cable_substeps must be a positive integer')
            self.physics_dt = self.dt/int(legacy_steps)
        self.substeps, self.integration_dt = integration_steps(self.dt, self.physics_dt)
        self.iterations = int(numeric["constraint_iterations"])
        # Historical artifacts may retain Cb_n_m2_s and
        # curvature_frame_regularization; neither enters this reduced model.
        if (not np.isfinite(self.dt) or self.dt <= 0
                or self.iterations != numeric["constraint_iterations"]
                or self.substeps < 1 or self.iterations < 1
                or self.gravity.shape != (3,) or not bool(torch.isfinite(self.gravity).all())
                or not bool(torch.isfinite(self.com_tracking).all())
                or not bool(torch.isfinite(self.attachment_tracking).all())):
            raise ValueError("Invalid integration or geometry parameters")
        for value in (self.ei, self.drag):
            if value.numel()!=1 or not bool(torch.isfinite(value)) or float(value)<0:
                raise ValueError("Cable stiffness/drag must be finite and nonnegative")
        self.n = len(masses)
        self.size = 6 + 3 * (self.n - 1)
        self.eye3 = torch.eye(3, dtype=torch.float64, device=device)
        self.free_mass_matrix = torch.diag(self.node_mass[1:].repeat_interleave(3))
        # Maps node displacements to edge displacements.
        incidence = torch.eye(self.n, device=device, dtype=torch.float64)[1:] - torch.eye(self.n, device=device, dtype=torch.float64)[:-1]
        self.edge_map = torch.kron(incidence, self.eye3)
        # These depend only on fixed geometry/masses, not the fit parameters.
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
        """Dense-assembly compatibility for numerical diagnostics."""
        return self._tridiagonal_solve(matrix.diagonal(), matrix.diagonal(offset=1), rhs)

    def _tridiagonal_solve(self, diagonal, off_diagonal, rhs):
        """Solve the SPD chain from its two nonzero diagonals.

        Adjacent cable edges share one node; the body affects only the first
        edge. Used for acceleration constraints and terminal projections.
        Functional elimination preserves autograd/vmap support. Invalid pivots
        yield nonfinite output, rejected by the existing runtime/fit checks.
        """
        # Scalar tensor elimination needs fusion to be efficient. Preserve
        # LAPACK on CPU and the existing uncompiled GPU reference path.
        if not diagonal.is_cuda or not torch.compiler.is_compiling():
            matrix = (torch.diag(diagonal) + torch.diag(off_diagonal, diagonal=1)
                      + torch.diag(off_diagonal, diagonal=-1))
            return self._solve(matrix, rhs)
        pivots = [diagonal[0]]
        values = [rhs[0]]
        for i in range(1, self.n-1):
            ratio = off_diagonal[i-1]/pivots[-1]
            pivots.append(diagonal[i]-ratio*off_diagonal[i-1])
            values.append(rhs[i]-ratio*values[-1])
        solution = [values[-1]/pivots[-1]]
        for i in range(self.n-3, -1, -1):
            solution.append((values[i]-off_diagonal[i]*solution[-1])/pivots[i])
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
        There is no retained material-angle or axial-spin state.
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

    def _body_solve(self, body, rhs):
        """Same coupled six-DOF mass inverse, without free-node zero blocks."""
        coupling = body[:3, 3:6]
        translation = rhs[:3]/self.root_translation_mass
        angular = self.root_inertia_inverse @ (rhs[3:6]-coupling.T @ translation)
        return torch.cat((translation-coupling @ angular/self.root_translation_mass, angular))

    def _constraint_system(self, positions, boundary, body):
        """Local assembly of G M^-1 G.T; only adjacent edges share mass.

        The first row includes the attachment's full rigid-body response.
        Tangent norms are retained rather than assuming exact unit roundoff.
        No geometry or mass is detached: state/parameter gradients are preserved.
        """
        edges = positions[1:]-positions[:-1]
        lengths = edges.norm(dim=1).clamp_min(1e-12)
        tangent = edges/lengths[:, None]
        root_row = -(tangent[0] @ boundary)
        root_response = self._body_solve(body, root_row)
        right = (tangent*(tangent/self.node_mass[1:, None])).sum(-1)
        left = (tangent[1:]*(tangent[1:]/self.node_mass[1:-1, None])).sum(-1)
        diagonal = right + torch.cat(((root_row @ root_response).reshape(1), left))
        off = -(tangent[:-1]*(tangent[1:]/self.node_mass[1:-1, None])).sum(-1)
        return lengths, tangent, root_response, diagonal, off

    @staticmethod
    def _constraint_product(tangent, boundary, generalized):
        """Apply G to a generalized velocity/acceleration without building G."""
        free = generalized[6:].reshape(-1, 3)
        left = torch.cat(((boundary @ generalized[:6])[None], free[:-1]))
        return (tangent*(free-left)).sum(-1)

    def _constraint_impulse(self, tangent, root_response, multipliers):
        """Apply M^-1 G.T to multipliers, including drone reaction torque."""
        edge_force = tangent*multipliers[:, None]
        free = torch.cat((edge_force[:-1]-edge_force[1:], edge_force[-1:]))
        return torch.cat((root_response*multipliers[0],
                          (free/self.node_mass[1:, None]).flatten()))

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

    def _acceleration(self, state, thrust_n, torque_body_nm):
        """Constrained acceleration, including moving-constraint curvature.

        For g(q)=edge_length-rest, G*a + dG/dt*u = 0. The eliminated
        attachment contributes a centripetal acceleration as well as inertia.
        All cable/root reactions act on both drone translation and rotation.
        """
        q,v = self.cable_state(state)
        boundary,body = self._projection_maps(state)
        bending = analytic_isotropic_bending_force(q[None],self.ei.reshape(1),self.dual)[0]
        cable_force = bending + self.cable_gravity
        if self.residual is not None:
            extra = self.node_mass[1:,None]*self.residual.cable_acceleration(self,state)
            cable_force = cable_force + torch.cat((torch.zeros_like(extra[:1]),extra))
        cable_force = cable_force-self.drag*self.node_mass[:,None]*v
        body_force = self.drone_gravity + self.tensor(thrust_n)*state.rotation[:,2]
        gyro = torch.linalg.cross(state.omega_body,self.inertia @ state.omega_body)
        force = (torch.cat((body_force,self.tensor(torque_body_nm)-gyro))
                 + boundary.T @ cable_force[0])
        centripetal = state.rotation @ torch.linalg.cross(state.omega_body,
                                      torch.linalg.cross(state.omega_body,self.arm))
        force = force-self.node_mass[0]*(boundary.T @ centripetal)
        a = torch.cat((self._body_solve(body,force),
                       (cable_force[1:]/self.node_mass[1:,None]).flatten()))
        lengths,tangents,root_response,diagonal,off = self._constraint_system(q,boundary,body)
        relative = v[1:]-v[:-1]
        transverse = relative-tangents*(relative*tangents).sum(1,keepdim=True)
        bias = transverse.square().sum(1)/lengths
        root_bias = torch.cat(((tangents[0] @ centripetal).reshape(1),lengths.new_zeros(self.n-2)))
        rhs = self._constraint_product(tangents,boundary,a)+bias-root_bias
        multipliers = self._tridiagonal_solve(diagonal,off,rhs)
        return a-self._constraint_impulse(tangents,root_response,multipliers)

    def _project(self, state, velocity):
        """Project configuration and tangent velocity independently.

        Do not turn the high-order position correction into a delta/h velocity
        impulse: doing so would reduce the order of the time integrator.
        """
        for _ in range(self.iterations):
            boundary,body = self._projection_maps(state)
            q,_ = self.cable_state(state)
            lengths,tangent,root_response,diagonal,off = self._constraint_system(q,boundary,body)
            multipliers = self._tridiagonal_solve(diagonal,off,self.rest-lengths)
            delta = self._constraint_impulse(tangent,root_response,multipliers)
            state = self._displace(state,delta)
        boundary,body = self._projection_maps(state)
        q,_ = self.cable_state(state)
        _,tangent,root_response,diagonal,off = self._constraint_system(q,boundary,body)
        multipliers = self._tridiagonal_solve(diagonal,off,
            self._constraint_product(tangent,boundary,velocity))
        velocity = velocity-self._constraint_impulse(tangent,root_response,multipliers)
        return self._with_velocity(state,velocity)

    @staticmethod
    def _rotation_rate(theta, omega):
        # Right-Jacobian inverse for R=R0 exp(theta), through the order needed
        # by RK4. Omitting these commutators loses order for changing axes.
        cross = torch.linalg.cross(theta,omega)
        return omega+.5*cross+torch.linalg.cross(theta,cross)/12

    def _rk4(self, state, h, actuation):
        """Fourth-order Lie-group RK step plus terminal constraint projection.

        Explicit, not symplectic or unconditionally stable. Internal timestep
        refinement is still required for sharp bends/high-frequency motion.
        """
        u1 = self._velocity(state)
        a1 = self._acceleration(state,*actuation(state))
        r1 = u1[3:6]
        d2 = .5*h*u1
        s2 = self._with_velocity(self._displace(state,d2),u1+.5*h*a1)
        u2 = self._velocity(s2)
        a2 = self._acceleration(s2,*actuation(s2))
        r2 = self._rotation_rate(d2[3:6],u2[3:6])
        d3 = .5*h*torch.cat((u2[:3],r2,u2[6:]))
        s3 = self._with_velocity(self._displace(state,d3),u1+.5*h*a2)
        u3 = self._velocity(s3)
        a3 = self._acceleration(s3,*actuation(s3))
        r3 = self._rotation_rate(d3[3:6],u3[3:6])
        d4 = h*torch.cat((u3[:3],r3,u3[6:]))
        s4 = self._with_velocity(self._displace(state,d4),u1+h*a3)
        u4 = self._velocity(s4)
        a4 = self._acceleration(s4,*actuation(s4))
        r4 = self._rotation_rate(d4[3:6],u4[3:6])
        average = (u1+2*u2+2*u3+u4)/6
        displacement = h*torch.cat((average[:3],(r1+2*r2+2*r3+r4)/6,average[6:]))
        return self._project(self._displace(state,displacement),u1+h*(a1+2*a2+2*a3+a4)/6)

    def substep(self, state, thrust_n, torque_body_nm, h):
        """One step with constant collective thrust and body torque."""
        return self._rk4(state,h,lambda _: (thrust_n,torque_body_nm))

    def controlled_substep(self, state, command, controller, h):
        """Integrate controller feedback at RK stages; hold the input command."""
        if (type(controller) is CoupledController and controller.model is self
                and type(self.residual) is CoupledResidual
                and self.residual.drone_equilibrium_centered):
            # Same weights and hover features across the four RK stages.
            # Recompute each substep and retain autograd; no persistent cache.
            equilibrium = self.residual.drone(self.residual._hover_features)
            return self._rk4(state,h,lambda stage: controller(
                stage,command,drone_equilibrium=equilibrium))
        return self._rk4(state,h,lambda stage: controller(stage,command))

    def step(self, state, command, controller):
        """One output interval; caller supplies the already-delayed command."""
        command = self.tensor(command)
        if command.shape != (11,) or not bool(torch.isfinite(command).all()):
            raise ValueError("Expected finite P/V/A/yaw/yaw-rate command")
        if float(command[10]) != 0:
            raise ValueError("This first controller has no yaw-rate feedforward")
        return self.step_unchecked(state, command, controller)

    def step_unchecked(self, state, command, controller, duration_s=None):
        """Same integrator for prevalidated inputs (CUDA capture / vmap)."""
        count, h = ((self.substeps, self.integration_dt) if duration_s is None else
                    integration_steps(duration_s, self.physics_dt))
        for _ in range(count):
            state = self.controlled_substep(state,command,controller,h)
        return state
