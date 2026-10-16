"""Differentiable counterpart of the geometric controller, without learned drone correction."""
import torch
from controllers.full_state_controller.geometric_controller import load_gains


class GeometricController:
    def __init__(self, model):
        self.model = model
        config = load_gains(model.config['controller'])
        for name in ('position_gain_s_inv2','velocity_gain_s_inv',
                     'attitude_frequency_rad_s','attitude_damping_ratio'):
            setattr(self, name, model.tensor(config[name]))
        self.controller_period_s = config['controller_period_s']

    @staticmethod
    def desired_rotation(force, yaw):
        z = force/force.norm()
        heading = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)))
        raw_y = torch.linalg.cross(z, heading)
        y = raw_y/raw_y.norm()
        # Degeneracy produces nonfinite output, rejected by rollout validity.
        return torch.stack((torch.linalg.cross(y,z),y,z),dim=1)

    def __call__(self, state, command):
        m = self.model
        p,v = m.tracking_pose(state)
        acceleration = (command[6:9]+self.position_gain_s_inv2*(command[:3]-p)
                        +self.velocity_gain_s_inv*(command[3:6]-v))
        trim = (m.mass+m.node_mass.sum())*m.gravity
        force = m.mass*acceleration-trim
        desired = self.desired_rotation(force,command[9])
        reference = self.desired_rotation(m.mass*command[6:9]-trim,command[9])
        R,omega = state.rotation,state.omega_body
        rates = command[10:13] if command.shape[-1] == 13 else torch.zeros_like(omega)
        rate = R.T@reference@rates
        return self._wrench(state,force,desired,rate)

    def velocity(self, state, command):
        """World XYZ velocity and fixed yaw; no position reference or rate feedforward."""
        m = self.model
        _,v = m.tracking_pose(state)
        acceleration = self.velocity_gain_s_inv*(command[:3]-v)
        force = m.mass*acceleration-(m.mass+m.node_mass.sum())*m.gravity
        desired = self.desired_rotation(force,command[3])
        return self._wrench(state,force,desired,torch.zeros_like(state.omega_body))

    def acceleration(self, state, command):
        """World XYZ acceleration request [m/s²], excluding gravity, and fixed yaw.

        Direct nominal force conversion with the existing static cable-weight
        trim. No position/velocity servo, integrated reference or load observer.
        This requests acceleration; the coupled plant need not achieve it.
        """
        m = self.model
        force = m.mass*command[:3]-(m.mass+m.node_mass.sum())*m.gravity
        desired = self.desired_rotation(force,command[3])
        return self._wrench(state,force,desired,torch.zeros_like(state.omega_body))

    def _wrench(self, state, force, desired, rate):
        m = self.model
        R,omega = state.rotation,state.omega_body
        skew = R.T@desired-desired.T@R
        error = .5*torch.stack((skew[2,1],skew[0,2],skew[1,0]))
        frequency = self.attitude_frequency_rad_s
        alpha = (frequency.square()*error-2*self.attitude_damping_ratio*frequency*(omega-rate)
                 -torch.linalg.cross(omega,rate))
        torque = m.inertia@alpha+torch.linalg.cross(omega,m.inertia@omega)
        return torch.dot(force,R[:,2]).clamp_min(0),torque
