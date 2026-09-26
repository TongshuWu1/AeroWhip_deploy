"""Small effective PVA controller for fitting, not a firmware emulation."""

import torch


class CoupledController:
    def __init__(self, model):
        self.model = model
        config = model.config["controller"]
        for name in ("position_gain_s_inv2","velocity_gain_s_inv","acceleration_feedforward",
                     "attitude_frequency_rad_s","attitude_damping_ratio"):
            value = model.tensor(config[name])
            if value.shape!=(3,) or not bool(torch.isfinite(value).all()) or bool((value<0).any()):
                raise ValueError(f"Invalid controller parameter: {name}")
            setattr(self,name,value)
        self.hover_scale = model.tensor(config["hover_thrust_scale"])
        if not bool(torch.isfinite(self.hover_scale)) or float(self.hover_scale)<=0:
            raise ValueError("Hover thrust scale must be positive")

    def __call__(self, state, command, *, drone_equilibrium=None):
        model = self.model
        p,v = model.tracking_pose(state)
        acceleration = (self.position_gain_s_inv2*(command[:3]-p)
                        +self.velocity_gain_s_inv*(command[3:6]-v)
                        +self.acceleration_feedforward*command[6:9])
        if model.residual is not None:
            if drone_equilibrium is None:
                correction = model.residual.drone_acceleration(model,state,command)
            else:
                correction = model.residual.drone_acceleration(
                    model,state,command,equilibrium=drone_equilibrium)
            acceleration = acceleration + correction
        # Nominal hover trim supplies total weight; drone inertia/mass in the
        # plant still exclude cable mass. This is controller effort, not a
        # second cable force added to the dynamics.
        force = model.mass*acceleration-self.hover_scale*(model.mass+model.node_mass.sum())*model.gravity
        z = force/force.norm().clamp_min(1e-12)
        yaw = command[9]
        heading = torch.stack((yaw.cos(),yaw.sin(),torch.zeros_like(yaw)))
        y = torch.linalg.cross(z,heading)
        y = y/y.norm().clamp_min(1e-12)
        desired = torch.stack((torch.linalg.cross(y,z),y,z),dim=1)
        error = state.rotation.T @ desired-desired.T @ state.rotation
        rotation_error = .5*torch.stack((error[2,1],error[0,2],error[1,0]))
        alpha = (self.attitude_frequency_rad_s.square()*rotation_error
                 -2*self.attitude_damping_ratio*self.attitude_frequency_rad_s*state.omega_body)
        return (torch.dot(force,state.rotation[:,2]).clamp_min(0),model.inertia @ alpha)
