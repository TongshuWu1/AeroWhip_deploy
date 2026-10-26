"""Identify and stop one local ROS launch, without container-wide signals."""
import json
import signal
import subprocess
import time


# Runs in the ROS container. PID start time, argv and domain must still match
# before every signal; remembered children remain identifiable after root exit.
PROCESS_SCRIPT = r'''
import json,os,signal,sys
from pathlib import Path
action,domain,selector=sys.argv[1],int(sys.argv[2]),json.loads(sys.argv[3])
def identity(pid):
    path=Path('/proc')/str(pid)
    try:
        stat=path.joinpath('stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z': return None
        argv=[a.decode() for a in path.joinpath('cmdline').read_bytes().split(b'\0') if a]
        env=dict(v.split(b'=',1) for v in path.joinpath('environ').read_bytes().split(b'\0') if b'=' in v)
        return dict(pid=pid,start=stat[19],argv=argv,domain=int(env.get(b'ROS_DOMAIN_ID',b'0')),parent=int(stat[1]))
    except (OSError,ValueError,UnicodeError): return None
def processes():
    return {int(p.name):i for p in Path('/proc').iterdir() if p.name.isdigit()
            if (i:=identity(int(p.name))) is not None}
def same(expected,current):
    return current is not None and all(expected[k]==current[k] for k in ('pid','start','argv','domain'))
def tree(root,table):
    selected={root['pid']:root}
    while True:
        added={pid:p for pid,p in table.items() if p['parent'] in selected and pid not in selected and p['domain']==domain}
        if not added: return list(selected.values())
        selected.update(added)
table=processes()
if action=='capture':
    def matches(p):
        a=p['argv']
        if p['domain']!=domain: return False
        if selector['kind']=='real':
            return any(Path(v).name=='ros2' and a[i+1:i+4]==['launch','crazyflie','launch.py']
                for i,v in enumerate(a)) and 'backend:=cflib' in a and 'mocap:=True' in a
        return ('--session' in a and a.index('--session')+1<len(a)
                and a[a.index('--session')+1]==selector['session'] and '--robot-config' in a)
    roots=[p for p in table.values() if matches(p)]
    if len(roots)>1: raise RuntimeError('Multiple matching ROS launches; refusing ambiguous shutdown ownership')
    result=dict(root=roots[0],members=tree(roots[0],table)) if roots else None
else:
    target=selector['target']; signum=selector['signal']; root=target['root']
    members={p['pid']:p for p in target['members']}
    if same(root,table.get(root['pid'])):
        members.update({p['pid']:p for p in tree(root,table)})
    active=[p for p in members.values() if same(p,table.get(p['pid']))]
    # SIGINT to ros2 launch lets ROS complete its own orderly node shutdown.
    recipients=[root] if signum==signal.SIGINT and same(root,table.get(root['pid'])) else active
    if signum:
        for p in recipients:
            if same(p,identity(p['pid'])):
                try: os.kill(p['pid'],signum)
                except ProcessLookupError: pass
    result=dict(root=root,members=active) if active else None
print(json.dumps(result))
'''


def _request(container, action, domain, selector):
    result = subprocess.run(['docker', 'exec', '-i', container, 'python3', '-c',
        PROCESS_SCRIPT, action, str(domain), json.dumps(selector)],
        check=True, capture_output=True, text=True, timeout=10.)
    return json.loads(result.stdout)


def capture_real_stack(container, domain):
    """Snapshot the matching local cflib/OptiTrack ROS launch and its children."""
    return _request(container, 'capture', domain, dict(kind='real'))


def capture_simulation_bridge(container, domain, session):
    return _request(container, 'capture', domain, dict(kind='simulation', session=session))


def close_captured_stack(container, domain, target):
    """Gracefully stop captured identities, escalating only their surviving tree."""
    if target is None:
        return
    for signum, timeout in ((signal.SIGINT, 12.), (signal.SIGTERM, 5.), (signal.SIGKILL, 3.)):
        target = _request(container, 'signal', domain, dict(target=target, signal=int(signum)))
        deadline = time.monotonic()+timeout
        while target is not None and time.monotonic() < deadline:
            time.sleep(.2)
            target = _request(container, 'signal', domain, dict(target=target, signal=0))
        if target is None:
            return
    raise RuntimeError('Captured ROS processes did not exit after shutdown')
