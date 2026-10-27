"""The model calibration manifest: one file beside the data (`model_calibration.yaml` in the
calibration data root) that lists the vehicle measurements, bench tests and flights with their
roles ('fit', 'held_out', 'hover', 'excluded'), usable spans and logs. Everything the fits read
about the data comes from here; nothing is hard-coded in the fitting code.
"""
from pathlib import Path

import yaml

SCHEMA = 'aerowhip_model_calibration_v1'
ROLES = ('fit', 'held_out', 'hover', 'excluded')
FILE = 'model_calibration.yaml'


class Manifest:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root/FILE
        self.data = yaml.safe_load(self.path.read_text())
        if self.data.get('schema') != SCHEMA:
            raise ValueError(f'{self.path}: expected schema {SCHEMA}')
        for f in self.data['flights']:
            if f['role'] not in ROLES:
                raise ValueError(f"{f['folder']}: unknown role {f['role']!r}")

    def vehicle(self, setup='cf_2'):
        """Measured values of the vehicle (the setup's base name without '_nocable')."""
        return self.data['vehicle'][setup.replace('_nocable', '')]

    def measurement(self, name, setup='cf_2'):
        entry = self.vehicle(setup)[name]
        if entry.get('value') is None:
            raise ValueError(f'{name} is not measured yet ({entry.get("status", "missing")})')
        return float(entry['value'])

    def flights(self, setup=None, roles=('fit', 'held_out'), session=None):
        """Flight entries in manifest order, filtered by setup, role and session."""
        return [f for f in self.data['flights']
                if (setup is None or f['setup'] == setup) and f['role'] in roles and (session is None or f['session'] == session)]

    def folder(self, entry):
        return self.root/entry['folder']

    def usable(self, entry):
        """(until_t_s or None, excluded phase prefixes) for a flight."""
        u = entry.get('usable', {})
        return u.get('until_t_s'), tuple(u.get('exclude_phases', ()))
