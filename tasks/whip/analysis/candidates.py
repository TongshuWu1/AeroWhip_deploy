"""Study-only paired spline priors; no learning, physics, clipping or repair."""

import math

import torch

from trajectories.exploration import smooth_action_matrix


RANGES = {
    'preparation_time_s': [.6, 1.1],
    'strike_duration_s': [.6, 1.3],
    'preparation_backward_m': [.10, .30],
    'strike_forward_m': [.15, .35],
    'preparation_z_m': [-.10, .10],
    'endpoint_z_m': [-.10, .10],
    'swing_amplitude_m': [.15, .40],
    'swing_span_s': [1.4, 2.8],
}


def _sigma(learner):
    """Mirror the configured policy log-std storage/clamping, not its RNG."""
    value = torch.tensor(math.log(learner['initial_std']), dtype=getattr(torch, learner['dtype']))
    return float(value.clamp(learner['min_log_std'], learner['max_log_std']).double().exp())


def _smoothstep(value):
    phase = value.clamp(0, 1)  # Definition of a finite transition, not action repair.
    return phase**3*(10+phase*(-15+6*phase))


def candidate_actions(configuration, exploration, learner, target_normal, seed, batch_size):
    """Return three CPU float64 (B,81) action batches and serializable provenance.

    All arms share full-rank Gaussian perturbations before tanh. Their physical
    covariance/excursion need not match after changing the nominal mean. Greville
    samples define control polygons, not interpolating waypoint trajectories.
    """
    if configuration.control_count != 32 or configuration.duration_s != 3.0:
        raise ValueError('This prespecified study requires 32 controls and a 3 s maneuver')
    if type(batch_size) is not int or batch_size < 1 or type(seed) is not int or seed < 0:
        raise ValueError('Provide a positive batch size and nonnegative integer seed')
    lower = torch.tensor(configuration.offset_lower_m, dtype=torch.float64)
    upper = torch.tensor(configuration.offset_upper_m, dtype=torch.float64)
    if not torch.equal(lower, -upper) or not bool((upper > 0).all()):
        raise ValueError('The unchanged zero-centred prior requires symmetric offset bounds')
    direction = torch.as_tensor(target_normal, dtype=torch.float64, device='cpu').clone()
    if (direction.shape != (3,) or not bool(torch.isfinite(direction).all())
            or abs(float(direction[2])) > 1e-12
            or abs(float(direction.norm())-1) > 1e-12):
        raise ValueError('Provide a finite horizontal unit target normal')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    matrix = smooth_action_matrix(configuration.control_count, configuration.duration_s, **exploration)
    sigma = _sigma(learner)
    noise = torch.randn((batch_size, configuration.action_size), generator=generator, dtype=torch.float64)
    perturbation = (sigma*noise) @ matrix.T
    uniform = torch.rand((batch_size, len(RANGES)), generator=generator, dtype=torch.float64)
    parameters = {name: bounds[0]+uniform[:, j]*(bounds[1]-bounds[0])
                  for j, (name, bounds) in enumerate(RANGES.items())}

    count, duration = configuration.control_count, configuration.duration_s
    spans = count-5
    knots = torch.cat((torch.zeros(6, dtype=torch.float64),
                       torch.arange(1, spans, dtype=torch.float64)/spans,
                       torch.ones(6, dtype=torch.float64)))
    greville = torch.stack([knots[j+1:j+6].mean() for j in range(count)])*duration
    free_times = torch.cat((greville[3:-3], greville[-1:]))
    times = free_times[None, :]
    preparation = -parameters['preparation_backward_m'][:, None]*direction
    preparation[:, 2] = parameters['preparation_z_m']
    endpoint = parameters['strike_forward_m'][:, None]*direction
    endpoint[:, 2] = parameters['endpoint_z_m']
    start = parameters['preparation_time_s'][:, None]
    strike_duration = parameters['strike_duration_s'][:, None]
    preparation_shape = (preparation[:, None, :]*_smoothstep(times/start)[:, :, None]
        +(endpoint-preparation)[:, None, :]*_smoothstep((times-start)/strike_duration)[:, :, None])
    swing_phase = times/parameters['swing_span_s'][:, None]
    swing_shape = torch.where((swing_phase > 0) & (swing_phase < 1),
                             torch.sin(math.pi*swing_phase)**4, 0.)
    swing = (parameters['swing_amplitude_m'][:, None, None]
             *swing_shape[:, :, None]*direction)
    templates = {'current_smooth': torch.zeros_like(preparation_shape),
                 'preparation_strike': preparation_shape, 'swing_candidates': swing}
    actions = {}
    for name, offsets in templates.items():
        normalized = 2*(offsets-lower)/(upper-lower)-1
        if not bool(torch.isfinite(normalized).all() & (normalized.abs() < 1).all()):
            raise ValueError(f'{name}: nominal controls must be strictly inside the unchanged box')
        actions[name] = torch.tanh(torch.atanh(normalized.flatten(1))+perturbation)
    metadata = dict(seed=seed, batch_size=batch_size, action_size=configuration.action_size,
        device='cpu', dtype='float64', sigma=sigma, exploration=dict(exploration),
        target_normal=direction.tolist(), ranges={name: list(bounds) for name, bounds in RANGES.items()},
        parameters={name: value.tolist() for name, value in parameters.items()},
        free_control_times_s=free_times.tolist(),
        pairing='One shared CPU float64 Gaussian noise array for every arm; independent uniform parameters.',
        sampling='Same initial Gaussian distribution, not a bitwise match to the CUDA learner RNG stream.',
        templates='Greville-sampled control polygons; production decoder imposes initial PVA and stopped endpoint.',
        limitations='Prior-discovery study, not learning. Nominal excursion and transformed covariance differ. '
                    'Swing candidates are an intended control family, not ground-truth non-whips. '
                    'No clipping, rejection sampling, action repair or adaptive search.')
    return actions, metadata


def verify_candidates(configuration, exploration, learner, target_normal, seed=0, batch_size=16):
    """Command-generation checks only; does not run physics or certify feasibility."""
    actions, metadata = candidate_actions(configuration, exploration, learner, target_normal, seed, batch_size)
    repeated, repeated_metadata = candidate_actions(configuration, exploration, learner, target_normal, seed, batch_size)
    assert metadata == repeated_metadata
    for name, value in actions.items():
        assert value.shape == (batch_size, 81) and value.dtype == torch.float64 and value.device.type == 'cpu'
        assert bool(torch.isfinite(value).all() & (value.abs() <= 1).all())
        assert torch.equal(value, repeated[name])
    generator = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn((batch_size, 81), generator=generator, dtype=torch.float64)
    matrix = smooth_action_matrix(32, 3., **exploration)
    expected = torch.tanh((_sigma(learner)*noise) @ matrix.T)
    assert torch.equal(actions['current_smooth'], expected)
    return ['CPU float64 Bx81 finite bounded actions', 'Exact same-seed reproducibility',
            'Exact manual tanh(M*z) baseline equality; no CUDA-stream equivalence claim']
