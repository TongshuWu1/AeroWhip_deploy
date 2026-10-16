"""Shutdown identity checks with local process fixtures; no ROS or hardware."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from deployment.stack_lifecycle import PROCESS_SCRIPT


class StackLifecycleTests(unittest.TestCase):
    def request(self, action, selector, domain=221):
        result = subprocess.run([sys.executable, '-c', PROCESS_SCRIPT, action,
            str(domain), json.dumps(selector)], check=True, capture_output=True, text=True)
        return json.loads(result.stdout)

    def launch(self, directory, domain):
        ready = Path(directory)/f'{domain}.json'
        script = '''import json,os,signal,subprocess,sys,time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
def close(*_):
 child.terminate();child.wait();sys.exit(0)
signal.signal(signal.SIGINT,close)
Path(sys.argv[1]).write_text(json.dumps(dict(pid=os.getpid(),child=child.pid)))
while True: time.sleep(.05)
'''
        process = subprocess.Popen([sys.executable, '-c', script, str(ready), 'ros2',
            'launch', 'crazyflie', 'launch.py', 'backend:=cflib', 'mocap:=True'],
            env=dict(os.environ, ROS_DOMAIN_ID=str(domain)), start_new_session=True)
        def clean():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        self.addCleanup(clean)
        deadline = time.monotonic()+3.
        while not ready.exists() and time.monotonic()<deadline:
            time.sleep(.01)
        self.assertTrue(ready.exists())
        return process

    def test_shutdown_stops_matching_launch_children_and_preserves_other_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            selected = self.launch(directory, 221)
            foreign = self.launch(directory, 222)
            independent = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                env=dict(os.environ, ROS_DOMAIN_ID='221'))
            def close_independent():
                independent.terminate(); independent.wait()
            self.addCleanup(close_independent)
            target = self.request('capture', dict(kind='real'))
            self.assertEqual(target['root']['pid'], selected.pid)
            self.assertEqual(len(target['members']), 2)
            self.request('signal', dict(target=target, signal=int(signal.SIGINT)))
            self.assertEqual(selected.wait(3.), 0)
            self.assertIsNone(self.request('signal', dict(target=target, signal=0)))
            self.assertIsNone(foreign.poll())
            self.assertIsNone(independent.poll())

    def test_remembered_child_can_be_stopped_after_launch_parent_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            process = self.launch(directory, 221)
            target = self.request('capture', dict(kind='real'))
            child = next(p for p in target['members'] if p['pid'] != process.pid)
            def close_orphan():
                try: os.kill(child['pid'], signal.SIGKILL)
                except ProcessLookupError: pass
            self.addCleanup(close_orphan)
            process.kill(); process.wait()
            surviving = self.request('signal', dict(target=target, signal=0))
            self.assertEqual([p['pid'] for p in surviving['members']], [child['pid']])
            self.request('signal', dict(target=surviving, signal=int(signal.SIGTERM)))
            deadline = time.monotonic()+3.
            while self.request('signal', dict(target=surviving, signal=0)) is not None:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)

    def test_stale_start_time_is_rejected_without_signalling_live_process(self):
        with tempfile.TemporaryDirectory() as directory:
            process = self.launch(directory, 221)
            target = self.request('capture', dict(kind='real'))
            for identity in [target['root'], *target['members']]:
                identity['start'] = 'invalid-start-time'
            self.assertIsNone(self.request('signal', dict(target=target, signal=int(signal.SIGKILL))))
            self.assertIsNone(process.poll())

    def test_ambiguous_launches_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.launch(directory, 221)
            Path(directory, '221.json').unlink()
            second = self.launch(directory, 221)
            with self.assertRaises(subprocess.CalledProcessError):
                self.request('capture', dict(kind='real'))
            self.assertIsNone(first.poll())
            self.assertIsNone(second.poll())


if __name__ == '__main__':
    unittest.main()
