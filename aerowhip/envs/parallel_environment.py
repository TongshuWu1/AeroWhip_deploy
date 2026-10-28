"""The batch environment with its lanes split over worker processes.

Each worker process owns a contiguous slice of the cohort's lanes and runs a complete
`BatchEnvironment` for them (native physics with one thread, the OptiTrack sensor and estimator,
the task), so every core simulates whole lanes and nothing waits on one Python process. The
learner process keeps the policy on its device and steps all workers in lockstep at every policy
decision: it sends each worker its slice of the actions and receives the new frame rows, the
critic state and the per-lane bookkeeping. Lanes are bit-identical to the same lanes in one
`BatchEnvironment` (every random draw is a function of the seeds, the cohort and the global lane
number; `BatchEnvironment(lane_offset=..., total_lanes=...)`).

The attributes the learner and the run records read (frames, critic_states, ledger, lengths,
transition_rewards, history, numerical_valid, terminal_times, ...) are assembled here; the
methods that need a lane's full state (replay_arrays, episode_records, workspace_metrics) are
answered by the workers.
"""
import multiprocessing
import os
import traceback

import numpy as np
import torch


def _serve(connection, arguments):
    """Worker process: one BatchEnvironment over a lane slice, driven by messages."""
    torch.set_num_threads(1)
    try:
        from aerowhip.envs.batch_environment import BatchEnvironment
        config, task_config, model, scene, keywords = arguments
        env = BatchEnvironment(config, task_config, model, scene, device='cpu', **keywords)
        connection.send(('ready', dict(frame_size=env.frame_size, critic_state_size=env.critic_state_size)))
    except BaseException:
        connection.send(('error', traceback.format_exc()))
        return
    first = [0, 0]          # index and decision at the start of the current decision
    while True:
        command, value = connection.recv()
        try:
            if command == 'close':
                return
            if command in ('reset', 'state'):     # 'state': the reset done when the environment was built
                if command == 'reset':
                    env.reset()
                connection.send(('ok', dict(frame=env.frames[:, 0].numpy(), critic=env.critic_states[:, 0].numpy(),
                                            active=env.active_np.copy())))
            elif command == 'begin':
                env.begin_decision(torch.as_tensor(value), cohort_active=True)
                first[:] = [env.index, env.decision]
                env.substep()
                connection.send(('ok', env.active_np.copy()))
            elif command == 'substep':
                env.substep()
                connection.send(('ok', env.active_np.copy()))
            elif command == 'end':
                index, decision = first
                env.end_decision()
                connection.send(('ok', dict(frames=env.frames[:, index+1:env.index+1].numpy(),
                                            critic=env.critic_states[:, decision+1].numpy(),
                                            ledger=env.ledger[:, decision].numpy(), lengths=env.lengths.numpy(),
                                            active=env.active_np.copy(), failed=env.failed_np.copy(),
                                            violation=env.physical_violation_np.copy(),
                                            breached=env.workspace_breached_np.copy(), index=env.index)))
            elif command == 'finish':
                returns, valid, components = env.finish()
                connection.send(('ok', dict(returns=returns.numpy(), valid=valid.numpy(),
                                            components={k: v.numpy() for k, v in components.items()},
                                            transition_rewards=env.transition_rewards.numpy(),
                                            history={k: v.numpy() for k, v in env.history.items()},
                                            numerical_valid=env.numerical_valid.numpy(),
                                            terminal_times=np.asarray(env.terminal_times, float),
                                            failure_records=env.failure_records,
                                            records=env.episode_records(),
                                            workspace={k: v.numpy() for k, v in env.workspace_metrics().items()})))
            elif command == 'replay':
                connection.send(('ok', env.replay_arrays(value)))
            elif command == 'stage':
                env.set_curriculum_stage(value)
                connection.send(('ok', None))
            elif command == 'randomised':
                env.randomised = bool(value) and env.randomisation.active
                connection.send(('ok', None))
            elif command == 'provenance':
                connection.send(('ok', env.provenance()))
            else:
                raise ValueError(f'Unknown command {command!r}')
        except BaseException:
            connection.send(('error', traceback.format_exc()))


class ParallelEnvironment:
    def __init__(self, config, task_config, model, scene, *, device, lanes, workers, actor_batch=1, **keywords):
        """lanes split into `workers` contiguous slices, one worker process each (at most one lane per
        worker). keywords: BatchEnvironment options (sensing_seed, randomised, randomisation_seed,
        reference_states, firmware, battery_voltage)."""
        from aerowhip.envs.batch_environment import BatchEnvironment
        self.config, self.device, self.batch, self.actor_batch = config, device, lanes, actor_batch
        # A one-lane local copy answers the static interface (shapes, task, schedule).
        local = dict(keywords, randomised=False, reference_states=False)
        self.meta = BatchEnvironment(config, task_config, model, scene, device='cpu', lanes=1, **local)
        m = self.meta
        for name in ('nodes', 'stride', 'samples', 'time_samples', 'decisions', 'time', 'action_size', 'frame_size', 'confidence_offset',
                     'actor_columns', 'critic_state_size', 'action_interface', 'randomisation', 'origin_np', 'model'):
            setattr(self, name, getattr(m, name))
        self.task = m.task
        self.stage = m.stage
        self._randomised = bool(keywords.get('randomised', False)) and self.randomisation.active
        slices = [part for part in np.array_split(np.arange(lanes), min(workers, lanes)) if len(part)]
        self.slices = [(int(part[0]), len(part)) for part in slices]
        context = multiprocessing.get_context('spawn')
        self.connections, self.processes = [], []
        previous = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
        os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'     # inherited by the workers
        try:
            for offset, count in self.slices:
                parent, child = context.Pipe()
                process = context.Process(target=_serve, daemon=True, args=(child, (
                    config, task_config, model, scene,
                    dict(keywords, lanes=count, lane_offset=offset, total_lanes=lanes, actor_batch=1))))
                process.start()
                self.connections.append(parent); self.processes.append(process)
        finally:
            for k, v in previous.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        for answer in self._gather():
            if answer['frame_size'] != self.frame_size or answer['critic_state_size'] != self.critic_state_size:
                raise RuntimeError('Worker environments differ from the learner layout')
        self.execution_backend = f'parallel_{len(self.processes)}_worker_processes'
        self._start(self._all('state'))      # each worker environment reset once when it was built

    # ------------------------------------------------------------------ messaging
    def _send(self, command, values=None):
        for k, connection in enumerate(self.connections):
            connection.send((command, None if values is None else values[k]))

    def _gather(self):
        answers = []
        for connection in self.connections:
            status, value = connection.recv()
            if status == 'error':
                self.close()
                raise RuntimeError(f'Environment worker failed:\n{value}')
            answers.append(value)
        return answers

    def _all(self, command, value=None):
        self._send(command, None if value is None else [value]*len(self.connections))
        return self._gather()

    def close(self):
        for connection, process in zip(self.connections, self.processes):
            try:
                if process.is_alive():
                    connection.send(('close', None))
            except (OSError, BrokenPipeError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
        self.connections, self.processes = [], []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ interface
    @property
    def randomised(self):
        return self._randomised

    @randomised.setter
    def randomised(self, value):
        self._randomised = bool(value) and self.randomisation.active
        self._all('randomised', self._randomised)

    def set_curriculum_stage(self, stage):
        self.stage = stage
        self.meta.set_curriculum_stage(stage)
        self._all('stage', stage)

    @torch.no_grad()
    def reset(self):
        self._start(self._all('reset'))
        return self.frames, self.indices()

    def _start(self, answers):
        self.index, self.decision = 0, 0
        self.frames = torch.zeros((self.batch, self.samples, self.frame_size), device=self.device)
        self.critic_states = torch.zeros((self.batch, self.decisions+1, self.critic_state_size), device=self.device)
        self.actions = torch.zeros((self.batch, self.decisions, self.action_size), device=self.device)
        self.ledger = torch.zeros((self.batch, self.decisions), dtype=torch.float64, device=self.device)
        self.lengths = torch.zeros(self.batch, device=self.device, dtype=torch.long)
        self.transition_rewards = None
        self.frames[:, 0] = torch.as_tensor(np.concatenate([a['frame'] for a in answers]), device=self.device)
        self.critic_states[:, 0] = torch.as_tensor(np.concatenate([a['critic'] for a in answers]), device=self.device)
        self.active_np = np.concatenate([a['active'] for a in answers])
        self.failed_np = np.zeros(self.batch, bool)
        self.physical_violation_np = np.zeros(self.batch, bool)
        self.workspace_breached_np = np.zeros(self.batch, bool)
        self.history = self.numerical_valid = None
        self.terminal_times = np.zeros(self.batch)
        self.failure_records = []
        self._finished = None

    @property
    def active(self):
        return torch.as_tensor(self.active_np, device=self.device)

    @property
    def failed(self):
        return torch.as_tensor(self.failed_np, device=self.device)

    @property
    def physical_violation(self):
        return torch.as_tensor(self.physical_violation_np, device=self.device)

    @property
    def workspace_breached(self):
        return torch.as_tensor(self.workspace_breached_np, device=self.device)

    def indices(self):
        return torch.full((self.batch,), self.index, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def step(self, action):
        if self.decision >= self.decisions or not self.active_np.any():
            raise RuntimeError('Cohort finished')
        if action.shape != (self.batch, 3) or not bool(torch.isfinite(action).all()) or bool((action.abs() > 1.000001).any()):
            raise ValueError('Expected normalized finite acceleration actions')
        action = action.detach().clamp(-1, 1)
        self.actions[:, self.decision] = torch.where(self.active[:, None], action, 0.)
        rows = action.cpu().numpy()
        # Sub-step by sub-step: the cohort stops when every lane has ended, as in one environment;
        # a slice whose own lanes have all ended keeps stepping until then.
        self._send('begin', [rows[offset:offset+count] for offset, count in self.slices])
        for k in range(self.stride):
            active = np.concatenate(self._gather())
            if k == self.stride-1 or not active.any():
                break
            self._send('substep', [None]*len(self.connections))
        self._send('end', [None]*len(self.connections))
        answers = self._gather()
        index = answers[0]['index']
        frames = torch.as_tensor(np.concatenate([a['frames'] for a in answers]), device=self.device)
        self.frames[:, self.index+1:index+1] = frames
        self.critic_states[:, self.decision+1] = torch.as_tensor(np.concatenate([a['critic'] for a in answers]), device=self.device)
        self.ledger[:, self.decision] = torch.as_tensor(np.concatenate([a['ledger'] for a in answers]), device=self.device)
        self.lengths = torch.as_tensor(np.concatenate([a['lengths'] for a in answers]), device=self.device)
        self.active_np = np.concatenate([a['active'] for a in answers])
        self.failed_np = np.concatenate([a['failed'] for a in answers])
        self.physical_violation_np = np.concatenate([a['violation'] for a in answers])
        self.workspace_breached_np = np.concatenate([a['breached'] for a in answers])
        self.index = index
        self.decision += 1
        return self.frames, self.indices(), not self.active_np.any()

    @torch.no_grad()
    def finish(self):
        answers = self._all('finish')
        tensor = lambda values: torch.as_tensor(np.concatenate(values), device=self.device)
        returns = tensor([a['returns'] for a in answers]).float()
        valid = tensor([a['valid'] for a in answers])
        components = {k: tensor([a['components'][k] for a in answers]) for k in answers[0]['components']}
        self.transition_rewards = tensor([a['transition_rewards'] for a in answers]).float()
        self.history = {k: torch.as_tensor(np.concatenate([a['history'][k] for a in answers]))
                        for k in answers[0]['history']}
        self.numerical_valid = torch.as_tensor(np.concatenate([a['numerical_valid'] for a in answers]))
        self.terminal_times = np.concatenate([a['terminal_times'] for a in answers])
        self.failure_records = [dict(record, lane=record['lane']+offset)
                                for (offset, _), a in zip(self.slices, answers) for record in a['failure_records']]
        self._finished = answers
        return returns, valid, components

    def episode_records(self):
        return [record for a in self._finished for record in a['records']]

    def workspace_metrics(self):
        keys = self._finished[0]['workspace']
        return {k: torch.as_tensor(np.concatenate([a['workspace'][k] for a in self._finished]), device=self.device)
                for k in keys}

    def replay_arrays(self, lane=0):
        for k, (offset, count) in enumerate(self.slices):
            if offset <= lane < offset+count:
                self.connections[k].send(('replay', lane-offset))
                status, value = self.connections[k].recv()
                if status == 'error':
                    raise RuntimeError(f'Environment worker failed:\n{value}')
                return value
        raise IndexError(lane)

    def policy_action(self, callback):
        raise ValueError('Deterministic fixed-state evaluation uses one lane; use the in-process environment')

    def provenance(self):
        records = self._all('provenance')
        first = dict(records[0])
        first.update(environment_lanes=self.batch, environment_workers=len(self.processes),
                     battery_voltage_v=[v for r in records for v in r['battery_voltage_v']],
                     hover_trim_mps2=[v for r in records for v in r['hover_trim_mps2']],
                     model_variants=[v for r in records for v in r['model_variants']],
                     reference_starts=sum(r['reference_starts'] for r in records),
                     reference_unlabelled=sum(r['reference_unlabelled'] for r in records),
                     unique_initial_states=sum(r['unique_initial_states'] for r in records))
        return first
