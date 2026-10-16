"""Transport/frame rejection and native command-clock parity, without ROS mocks."""
from dataclasses import replace
import ast
import math
from pathlib import Path
import socket
from types import SimpleNamespace
import time
import unittest

import numpy as np

from deployment.custom_simulation import DomainInUseError, DomainLease, PortInUseError, RosSimulationSource, SimulationTransport, StreamingCommands
from deployment.ros_bridge import PacketReader, encode, full_state_reference, body_velocity, parameter_batch
from deployment.configuration import select_robot
from deployment.flight_commands import FlightCommands, polynomial_reference, validate_plan
from simulation.configuration import DEFAULT_MODEL, load_configuration
from simulation.runtime import Simulation


def xyz(values):
    return SimpleNamespace(**dict(zip(('x','y','z'), values)))


class ReferenceTests(unittest.TestCase):
    def test_lost_api_ack_stops_session_instead_of_serving_uncertain_parameters(self):
        # Exercise the actual ROS method without creating a server or radio.
        source = Path(__file__).resolve().parents[1]/'deployment/ros_bridge.py'
        tree = ast.parse(source.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'request')
        ticks = iter((0., 2.))
        sent = []
        scope = dict(args=SimpleNamespace(robot='cf_2'), encode=encode,
                     time=SimpleNamespace(monotonic=lambda: next(ticks)))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
        bridge = SimpleNamespace(done=False, last={'running': True}, request_id=0,
                                 api_acks={}, peer=SimpleNamespace(sendall=sent.append))
        with self.assertRaisesRegex(ValueError, 'acknowledge'):
            scope['request'](bridge, 'parameters', {'high_level': 0})
        self.assertTrue(bridge.done)
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            scope['request'](bridge, 'plan', {})
        self.assertEqual(len(sent), 1)

    def message(self):
        return SimpleNamespace(header=SimpleNamespace(frame_id='/world'),
            pose=SimpleNamespace(position=xyz([1.,2.,3.]),
                                 orientation=SimpleNamespace(x=0.,y=0.,z=math.sin(.2),w=math.cos(.2))),
            twist=SimpleNamespace(linear=xyz([4.,5.,6.]),angular=xyz([.1,.2,.3])), acc=xyz([7.,8.,9.]))

    def test_full_state_units_yaw_and_explicit_body_rates(self):
        np.testing.assert_allclose(full_state_reference(self.message()),
                                   [1,2,3,4,5,6,7,8,9,.4,.1,.2,.3], rtol=0, atol=1e-15)
        message = self.message(); message.header.frame_id = 'map'
        with self.assertRaisesRegex(ValueError, 'frame'):
            full_state_reference(message)
        message = self.message(); message.pose.orientation.w = 2.
        with self.assertRaisesRegex(ValueError, 'normalized'):
            full_state_reference(message)
        message = self.message(); message.pose.orientation = SimpleNamespace(x=math.sin(.1),y=0.,z=0.,w=math.cos(.1))
        with self.assertRaisesRegex(ValueError, 'yaw only'):
            full_state_reference(message)
        message = self.message(); message.acc.x = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            full_state_reference(message)

    def test_odometry_child_body_frame_velocity(self):
        np.testing.assert_allclose(body_velocity([1,0,0], [0,0,math.sqrt(.5),math.sqrt(.5)]), [0,-1,0], atol=5e-16)

    def test_framing_handles_fragmentation_and_rejects_wrong_schema(self):
        reader = PacketReader(); data = encode(dict(kind='ping'))+encode(dict(kind='hello'))
        self.assertEqual(reader.feed(data[:8]), [])
        self.assertEqual([p['kind'] for p in reader.feed(data[8:])], ['ping','hello'])
        with self.assertRaisesRegex(ValueError, 'schema'):
            PacketReader().feed(b'{"schema":"other"}\n')


class FlightApiTests(unittest.TestCase):
    def test_parameter_batch_rejects_invalid_tail_and_conflicting_aliases(self):
        params = [SimpleNamespace(name='all.params.commander.enHighLevel', value=0),
                  SimpleNamespace(name='use_sim_time', value=False)]
        with self.assertRaisesRegex(ValueError, 'use_sim_time'):
            parameter_batch(params, 'cf_2')
        params[1] = SimpleNamespace(name='cf_2.params.commander.enHighLevel', value=1)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            parameter_batch(params, 'cf_2')
        params[1].value = 0
        params.append(SimpleNamespace(name='cf_2.params.hlCommander.groupmask', value=17))
        self.assertEqual(parameter_batch(params, 'cf_2'), {'high_level': 0, 'group_mask': 17})

    def test_host_parameter_commit_is_atomic_immediate_and_survives_disarm(self):
        commands = FlightCommands('cf_2')
        before = commands.high_level_enabled, commands.group_mask
        with self.assertRaises(ValueError):
            commands.apply_parameters(self.api('parameters', {'high_level': 0, 'group_mask': 256}), self.snapshot())
        self.assertEqual((commands.high_level_enabled, commands.group_mask), before)
        commands.submit_api(self.api('arm', {'enabled': False, 'hold': [0.]*13}), self.snapshot())
        commands.apply_parameters(self.api('parameters', {'high_level': 0, 'group_mask': 17}), self.snapshot())
        self.assertEqual((commands.high_level_enabled, commands.group_mask), (False, 17))
        commands.events(0., .01)
        self.assertEqual((commands.high_level_enabled, commands.group_mask), (False, 17))

    def test_disabling_commander_cancels_pending_plan(self):
        commands = FlightCommands('cf_2')
        commands.submit_api(self.api('plan', self.plan()), self.snapshot())
        commands.apply_parameters(self.api('parameters', {'high_level': 0}), self.snapshot())
        self.assertEqual(commands.events(0., .01), [])
        with self.assertRaisesRegex(ValueError, 'enabled commander'):
            commands.submit_api(self.api('plan', self.plan()), self.snapshot())

    def test_domain_lease_is_exclusive_and_released_without_deleting_files(self):
        lease = DomainLease(252)
        try:
            with self.assertRaisesRegex(DomainInUseError,'--port 0'):
                DomainLease(252)
        finally:
            lease.close()
        restarted = DomainLease(252)
        restarted.close()
    def plan(self):
        # Smooth seventh-order endpoint interpolation, checked independently
        # through finite-difference P/V/A rather than copied evaluation logic.
        return validate_plan(dict(pieces=[dict(duration_s=1., coefficients=[
            [0.,0.,0.,0.,.35,-.84,.70,-.20], [0.]*8,
            [1.1,0.,0.,0.,1.05,-2.52,2.1,-.6], [0.]*8])]))

    def snapshot(self):
        return SimpleNamespace(running=True, error=None, state=SimpleNamespace(time_s=0.))

    def api(self, action, payload):
        return dict(kind='api', robot='cf_2', action=action, payload=payload)

    def test_yaml_enabled_identity_frame_precedence_and_explicit_override(self):
        base = dict(fileversion=3, all={'reference_frame':'odom'}, robot_types={'bolt':{'reference_frame':'map'}},
            robots={'cf_2':dict(enabled=True,type='bolt',uri='radio://0/80/2M/E7E7E7EA02',
                reference_frame='world',initial_position=[0,0,0]),
                'cf_3':dict(enabled=False,type='bolt')})
        default_record = select_robot(base)
        self.assertEqual(default_record['initial_position'], [0.,0.,0.])
        self.assertEqual(default_record['yaml_initial_position'], [0.,0.,0.])
        self.assertEqual(default_record['airborne_landing_height_m'], 0.)
        record = select_robot(base, {'robots':{'cf_2':{'initial_position':[0,0,1.1]}}})
        self.assertEqual(record['name'], 'cf_2')
        self.assertEqual(record['initial_position'], [0.,0.,1.1])
        self.assertEqual(record['yaml_initial_position'], [0.,0.,0.])
        self.assertEqual(select_robot(base, {})['initial_position'], [0.,0.,0.])
        arbitrary = dict(base, robots={'test_robot':dict(base['robots']['cf_2'], initial_position=[2.3,-1.7,-.4])})
        record = select_robot(arbitrary)
        self.assertEqual(record['name'], 'test_robot')
        self.assertEqual(record['initial_position'], [2.3,-1.7,-.4])
        base['robots']['cf_2'].pop('reference_frame')
        with self.assertRaisesRegex(ValueError, 'world'):
            select_robot(base, {})

    def test_polynomial_derivatives_and_reverse_timescale(self):
        plan = self.plan()
        h = 1e-5
        for t in (.2,.5,.8):
            before, current, after = (polynomial_reference(plan, s) for s in (t-h,t,t+h))
            np.testing.assert_allclose((after[:3]-before[:3])/(2*h),current[3:6],atol=3e-9,rtol=0)
            np.testing.assert_allclose((after[3:6]-before[3:6])/(2*h),current[6:9],atol=2e-8,rtol=0)
        reversed_plan = dict(plan, reverse=True, timescale=2.)
        direct = polynomial_reference(plan,.7)
        reverse = polynomial_reference(reversed_plan,.6)
        np.testing.assert_allclose(reverse[:3],direct[:3],atol=1e-14)
        np.testing.assert_allclose(reverse[3:6],-direct[3:6]/2,atol=1e-14)
        np.testing.assert_allclose(reverse[6:9],direct[6:9]/4,atol=1e-14)
        np.testing.assert_array_equal(polynomial_reference(plan,2.)[3:9],np.zeros(6))

    def test_streaming_release_preserves_duration_and_emergency_clears_commands(self):
        commands = FlightCommands('cf_2')
        commands.reset()
        packet = dict(robot='cf_2', frame_id='world', sequence=1,stamp_s=0.,command=[0,0,1.1]+[0]*10)
        commands.submit(packet,self.snapshot())
        commands.events(0.,.01)
        commands.submit_api(self.api('plan',self.plan()),self.snapshot())
        self.assertEqual(commands.events(.005,.01), [])
        self.assertTrue(commands.status()['high_level_waiting_for_notify'])
        commands.submit_api(self.api('notify',dict(millis=200)),self.snapshot())
        commands.submit_api(self.api('plan',self.plan()),self.snapshot())
        self.assertEqual(commands.events(.01,.02), [])
        self.assertEqual(commands.events(.02,.03), [])
        self.assertEqual(commands.events(.20,.21), [])
        events = commands.events(.21,.22)
        self.assertEqual(len(events),1)
        np.testing.assert_allclose(events[0][1][:3],[0,0,1.1])
        commands.halt()
        self.assertTrue(commands.status()['emergency_latched'])
        self.assertEqual(commands.events(.22,.23), [])
        with self.assertRaisesRegex(ValueError,'armed'):
            commands.submit(dict(packet,sequence=2),self.snapshot())

    def test_disarm_cancels_plan_and_rejects_stream_without_stopping_clock(self):
        commands = FlightCommands('cf_2')
        commands.reset()
        commands.submit_api(self.api('plan',self.plan()),self.snapshot())
        commands.events(0.,.01)
        hold = [0,0,1.3]+[0]*10
        commands.submit_api(self.api('arm',dict(enabled=False,hold=hold)),self.snapshot())
        np.testing.assert_array_equal(commands.events(.1,.11)[0][1],hold)
        self.assertFalse(commands.status()['armed'])
        self.assertEqual(commands.events(.2,.21), [])

    def test_late_topic_delivery_cannot_cancel_notify_service(self):
        commands = FlightCommands('cf_2')
        commands.reset()
        packet = dict(robot='cf_2',frame_id='world',sequence=1,stamp_s=0.,command=[0,0,1.1]+[0]*10)
        commands.submit(packet,self.snapshot()); commands.events(0.,.01)
        commands.submit_api(self.api('notify',dict(millis=200)),self.snapshot())
        commands.events(.01,.02)
        # A previously sent ROS topic message can arrive after the service.
        commands.submit(dict(packet,sequence=2,stamp_s=.01),self.snapshot())
        commands.events(.02,.03)
        commands.submit_api(self.api('plan',self.plan()),self.snapshot())
        commands.events(.03,.04)
        self.assertEqual(commands.events(.20,.21), [])
        self.assertEqual(len(commands.events(.21,.22)),1)
        self.assertEqual(commands.status()['flight_mode'],'high_level')

    def test_land_service_delivered_before_notify_waits_for_explicit_release(self):
        commands = FlightCommands('cf_2')
        commands.reset()
        packet = dict(robot='cf_2',frame_id='world',sequence=1,stamp_s=0.,command=[0,0,1.1]+[0]*10)
        commands.submit(packet,self.snapshot()); commands.events(0.,.01)
        commands.submit_api(self.api('plan',self.plan()),self.snapshot())
        self.assertEqual(commands.events(.01,.02), [])
        self.assertTrue(commands.status()['high_level_waiting_for_notify'])
        self.assertEqual(commands.events(.2,.21), [])
        commands.submit_api(self.api('notify',dict(millis=100)),self.snapshot())
        self.assertEqual(commands.events(.21,.22), [])
        self.assertEqual(commands.events(.30,.31), [])
        self.assertEqual(len(commands.events(.31,.32)),1)


class AccelerationConnectionTests(unittest.TestCase):
    def source(self):
        commands = FlightCommands('cf_2')
        sim = Simulation(load_configuration('configs/scenes/default.toml'), model_config=DEFAULT_MODEL,
                         command_source=commands, command_reset=commands.reset)
        return RosSimulationSource(sim, commands, 'acceleration-test')

    def disarm(self, source):
        source.commands.submit_api(dict(robot='cf_2',action='arm',payload=dict(enabled=False,
            hold=source.packet['position_world_m']+[0.]*10)), source.latest)
        for _ in range(5):
            source.step()

    def test_default_wrapper_preserves_native_pva_and_initial_reference(self):
        source = self.source()
        direct = Simulation(load_configuration('configs/scenes/default.toml'),model_config=DEFAULT_MODEL)
        try:
            for i in range(12):
                command = [.001*i,-.0005*i,2+.0002*i,0,0,0,0,0,0,.002*i,.001,0,.002]
                source.step(command); direct.step(command)
                np.testing.assert_array_equal(source.checkpoint()['state'],direct.dynamics.checkpoint()['state'])
        finally:
            source.close()

    def test_acceleration_native_motion_ignores_position_velocity_placeholders(self):
        left, right = self.source(), self.source()
        try:
            for source in (left,right):
                self.disarm(source)
                source._call('acceleration_mode',True)
                source.commands.submit_api(dict(robot='cf_2',action='arm',payload=dict(enabled=True)),source.latest)
                source.step()
            for i in range(35):
                for source, placeholder in ((left,[0.]*6),(right,[10.,-20.,30.,4.,-5.,6.])):
                    source.commands.submit(dict(robot='cf_2',frame_id='world',sequence=i,
                        stamp_s=source.state.time_s,command=placeholder+[.6,-.2,.1,.2,0.,0.,0.]),
                        replace(source.latest,running=True))
                    source.step()
                np.testing.assert_array_equal(left.state.drone.position_world_m,right.state.drone.position_world_m)
                np.testing.assert_array_equal(left._simulation.dynamics.native.state(),right._simulation.dynamics.native.state())
            self.assertGreater(abs(left.state.drone.position_world_m[0]),.001)
            self.assertEqual(left.checkpoint()['schema'],'aerowhip_ros_acceleration_runtime_v1')
            with self.assertRaisesRegex(ValueError,'schema'):
                left._simulation.dynamics.restore(left.checkpoint())
        finally:
            left.close(); right.close()

    def test_mode_guard_delay_rejection_and_reset(self):
        source = self.source()
        try:
            with self.assertRaisesRegex(ValueError,'Disarm'):
                source._call('acceleration_mode',True)
            hold = source.packet['position_world_m']+[0.]*10
            hold[0] += .1
            source.commands.submit_api(dict(robot='cf_2',action='arm',payload=dict(enabled=False,hold=hold)),source.latest)
            source.step()
            with self.assertRaisesRegex(ValueError,'settle'):
                source._call('acceleration_mode',True)
            for _ in range(5): source.step()
            source._call('acceleration_mode',True)
            self.assertTrue(source.packet['acceleration_only'])
            with self.assertRaisesRegex(ValueError,'PVA mode'):
                source.commands.submit_api(dict(robot='cf_2',action='plan',payload={}),source.latest)
            source.commands.submit_api(dict(robot='cf_2',action='arm',payload=dict(enabled=True)),source.latest)
            source.step()
            with self.assertRaisesRegex(ValueError,'zero body-rate'):
                source.commands.submit(dict(robot='cf_2',frame_id='world',sequence=1,
                    stamp_s=source.state.time_s,command=[0.]*12+[1.]),replace(source.latest,running=True))
            source.reset()
            self.assertFalse(source.packet['acceleration_only'])
            self.assertEqual(source.checkpoint()['schema'],'aerowhip_runtime_bdf2_v1')
        finally:
            source.close()

    def test_mode_parameter_requires_standalone_boolean(self):
        value = SimpleNamespace(name='aerowhip.acceleration_only',value=True)
        self.assertEqual(parameter_batch([value],'cf_2'),{'acceleration_only':True})
        with self.assertRaisesRegex(ValueError,'alone'):
            parameter_batch([value,SimpleNamespace(name='all.params.commander.enHighLevel',value=1)],'cf_2')
        value.value = 1
        with self.assertRaisesRegex(ValueError,'boolean'):
            parameter_batch([value],'cf_2')


class NativeConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = load_configuration('configs/scenes/default.toml')

    def source(self):
        commands = StreamingCommands('cf_2')
        sim = Simulation(self.scene, model_config=DEFAULT_MODEL, command_source=commands, command_reset=commands.reset)
        return RosSimulationSource(sim, commands, 'test-session')

    def packet(self, sequence=1, stamp=0., command=None):
        return dict(kind='command', robot='cf_2', frame_id='world', sequence=sequence,
                    stamp_s=stamp, command=command or [0,0,2,0,0,0,0,0,0,0,0,0,0])

    def test_native_checkpoints_delayed_commands_and_controller_clocks_identical(self):
        source = self.source()
        direct = Simulation(self.scene, model_config=DEFAULT_MODEL)
        try:
            for i in range(10):
                command = [.001*i,-.0005*i,2+.0002*i,0,0,0,0,0,0,.002*i,.001,0,.002]
                source.commands.submit(self.packet(i+1, source.state.time_s, command), replace(source.latest, running=True))
                source.step(); direct.step(command)
                np.testing.assert_array_equal(source.checkpoint()['state'], direct.dynamics.checkpoint()['state'])
                self.assertEqual(source.packet['controller_update_times'], list(direct.dynamics.last_controller_update_times))
                self.assertEqual(source.latest.frame.robots[0].name, 'cf_2')
            self.assertAlmostEqual(source.packet['command_delay_s'], .03)
            self.assertEqual(source.packet['commands_received'], 10)
            self.assertEqual(source.packet['commands_scheduled'], 10)
            self.assertEqual(source.packet['applied_events'][0][0], .09)
        finally:
            source.close()
        self.assertFalse(source._thread.is_alive())

    def test_invalid_clock_order_and_overflow_cannot_mutate_native_state(self):
        source = self.source()
        try:
            initial = source.checkpoint()['state']
            running = replace(source.latest, running=True)
            with self.assertRaisesRegex(ValueError, 'paused'):
                source.commands.submit(self.packet(), source.latest)
            with self.assertRaisesRegex(ValueError, 'clock'):
                source.commands.submit(self.packet(stamp=1700000000.), running)
            source.commands.submit(self.packet(), running)
            with self.assertRaisesRegex(ValueError, 'regressed'):
                source.commands.submit(self.packet(), running)
            for i in range(2,129):
                source.commands.submit(self.packet(i), running)
            with self.assertRaisesRegex(ValueError, 'full'):
                source.commands.submit(self.packet(129), running)
            np.testing.assert_array_equal(source.checkpoint()['state'], initial)
            source.reset()
            self.assertEqual(len(source.commands.pending), 0)
        finally:
            source.close()

    def test_socket_handshake_identity_rejection_and_disconnect_pauses(self):
        source = self.source()
        transport = SimulationTransport(source, port=0, domain=142)
        peer = socket.create_connection(('127.0.0.1', transport.port), timeout=1.)
        try:
            peer.sendall(encode(dict(kind='hello', robot='cf_2', domain=142, session='test-session')))
            self.assertTrue(transport.connected.wait(1.))
            reader = PacketReader(); packet = None
            deadline = time.monotonic()+1.
            while packet is None and time.monotonic() < deadline:
                values = reader.feed(peer.recv(65536))
                packet = next((v for v in values if v['kind']=='state'), None)
            self.assertEqual(packet['robot'], 'cf_2')
            self.assertEqual(packet['frame_id'], 'world')
            peer.close()
            transport.thread.join(2.)
            self.assertIn('disconnected', source.latest.error)
            self.assertFalse(source.running)
        finally:
            peer.close(); transport.close(); source.close()

        source = self.source(); transport = SimulationTransport(source, port=0, domain=142)
        peer = socket.create_connection(('127.0.0.1', transport.port), timeout=1.)
        try:
            peer.sendall(encode(dict(kind='hello', robot='cf_2', domain=42, session='test-session')))
            transport.thread.join(2.)
            self.assertFalse(transport.connected.is_set())
            self.assertIn('mismatch', source.latest.error)
        finally:
            peer.close(); transport.close(); source.close()

    def test_same_port_restart_and_live_listener_exclusivity(self):
        source = self.source()
        transport = SimulationTransport(source, port=0, domain=142)
        port = transport.port
        peer = socket.create_connection(('127.0.0.1', port), timeout=1.)
        try:
            peer.sendall(encode(dict(kind='hello', robot='cf_2', domain=142, session='test-session')))
            self.assertTrue(transport.connected.wait(1.))
            # The server closes the accepted connection first, exercising TCP
            # TIME_WAIT rather than merely closing an unused listening socket.
            transport.close()
            while peer.recv(65536):
                pass
        finally:
            peer.close(); transport.close(); source.close()

        source = self.source()
        transport = SimulationTransport(source, port=port, domain=142)
        peer = None
        try:
            with self.assertRaisesRegex(PortInUseError, 'Close any existing custom simulation'):
                SimulationTransport(source, port=port, domain=142)
            # A failed duplicate launch must leave the original listener usable.
            peer = socket.create_connection(('127.0.0.1', port), timeout=1.)
            peer.sendall(encode(dict(kind='hello', robot='cf_2', domain=142, session='test-session')))
            self.assertTrue(transport.connected.wait(1.))
            self.assertTrue(transport.thread.is_alive())
            self.assertIsNone(transport.error)
        finally:
            if peer is not None:
                peer.close()
            transport.close(); source.close()

if __name__ == '__main__':
    unittest.main()
