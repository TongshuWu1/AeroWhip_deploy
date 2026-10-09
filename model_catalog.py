"""Read-only discovery of manifest-packaged checkpoint and replay evidence."""
import hashlib
import json

from project_paths import ROOT, project_path, contained_child


def packaged_stages(root=ROOT):
    entries = []
    for manifest_path in sorted((project_path(root)/'models').glob('*/manifest.json')):
        manifest_path = project_path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('schema') != 'aerowhip_simulation_milestone_models_v1':
            continue
        for stage in manifest['stages']:
            package = manifest_path.parent
            entries.append(dict(package=package, manifest=manifest, stage=stage,
                checkpoint=contained_child(package, stage['checkpoint']),
                directory=contained_child(package, stage['evidence']['configuration_and_snapshot']).parent))
    return entries


def packaged_stage(selection):
    selection = project_path(selection)
    for entry in packaged_stages():
        if selection == entry['checkpoint'] or selection == entry['directory'] or selection.parent == entry['directory']:
            return entry
    return None


def verified_file(entry, reference):
    """Pin original bytes; relocation changes references in memory, never evidence."""
    path = contained_child(entry['package'], reference)
    expected = entry['manifest']['files'].get(reference)
    if expected is None or not path.is_file() or path.stat().st_size != expected['bytes']:
        raise ValueError('Missing or unregistered packaged artifact: '+reference)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected['sha256']:
        raise ValueError('Packaged artifact hash changed: '+reference)
    return path


def packaged_event(directory):
    entry = packaged_stage(directory)
    if entry is None:
        return None
    evidence = entry['stage']['evidence']
    for key in ('configuration_and_snapshot', 'scene', 'summary', 'final_nominal_replay'):
        verified_file(entry, evidence[key])
    verified_file(entry, entry['stage']['checkpoint'])
    event = json.loads(verified_file(entry, evidence['final_nominal_evaluation']).read_text(encoding='utf-8'))
    event = dict(event, original_replay=event.get('replay'),
                 replay=contained_child(entry['package'], evidence['final_nominal_replay']).name)
    return event


def checkpoint_for_directory(directory):
    directory = project_path(directory)
    entry = packaged_stage(directory)
    return verified_file(entry, entry['stage']['checkpoint']) if entry else directory/'latest.pt'
