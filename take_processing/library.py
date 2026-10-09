"""Source-linked take catalog. Only the separate catalog is ever written."""
from copy import deepcopy
from pathlib import Path
import json
import os
import tempfile

from take_processing.take_data import digest, discover, read_json
from project_paths import project_path
from take_processing.paths import (recording_root, recording_reference,
                                   resolve_recording_path, rehearsal_archive)


def source_files(path):
    path = resolve_recording_path(path)
    logger = path.with_name("experiment_" + path.name)
    batch = path.parent.parent
    protocol_path = batch / "protocol.json"
    protocol = read_json(protocol_path)
    paths = {
        "tracking": path, "samples": logger,
        "commands": logger.with_suffix(".commands.csv"),
        "identity": path.with_suffix(".tracking.json"),
        "segment": path.with_suffix(".segment.json"),
        "metadata": logger.with_suffix(".metadata.json"),
        "experiment": path.parent / "experiment.json",
        "alignment": batch / "time_alignment.json", "protocol": protocol_path,
    }
    if protocol.get("rehearsal"):
        paths["reference"] = rehearsal_archive(protocol['rehearsal'], path)
    return {name: {"path": str(p), "sha256": digest(p) if p.is_file() else None}
            for name, p in paths.items()}


def session_name(path):
    logger = path.with_name("experiment_" + path.name)
    segment = read_json(logger.with_suffix(".metadata.json")).get("segment", {})
    segment = segment or read_json(path.with_suffix(".segment.json"))
    protocol = read_json(path.parent.parent / "protocol.json")
    group = segment.get("recording_group") or protocol.get("recording_groups", {}).get(path.stem)
    if group:
        return str(group), "recording_group"
    experiment = read_json(path.parent / "experiment.json")
    if experiment.get("experiment_id"):
        return str(experiment["experiment_id"]), "experiment_id"
    folder = path.parent.parent if path.parent.name == "flight_take" else path.parent
    return folder.name, "folder"


def verify_sources(entry, *, root=None):
    for name, source in entry["sources"].items():
        path = resolve_recording_path(source['path'], root=root)
        actual = digest(path) if path.is_file() else None
        if actual != source["sha256"]:
            raise ValueError(f"{name.capitalize()} changed or moved. Import the folder again to review the new files.")


class Library:
    def __init__(self, path, *, root=None):
        self.path = project_path(path)
        self.root = project_path(root) if root is not None else recording_root()
        self._verified_root = None
        self._disk_hash = digest(self.path) if self.path.is_file() else None
        document = read_json(self.path)
        if document and document.get("schema") != "aerowhip_take_library_v1":
            raise ValueError("Unsupported take library")
        self.entries = document.get("takes", {})
        if not isinstance(self.entries, dict):
            raise ValueError("Invalid take library")

    def resolve_path(self, entry):
        return resolve_recording_path(entry['path'], root=self.root)

    def verify(self, entry):
        verify_sources(entry, root=self.root)

    def check_root(self, root):
        """A catalog has one collection root; relocation must preserve its sources."""
        root = project_path(root)
        for entry in self.entries.values():
            try:
                verify_sources(entry, root=root)
            except (OSError, ValueError) as exc:
                raise ValueError('This folder does not contain the same recording collection. '
                                 'Choose a folder within the current collection, or a complete '
                                 'relocated copy with its original layout.') from exc
        self._verified_root = root

    def _portable_sources(self, sources):
        return {name: {**source, 'path': recording_reference(source['path'], root=self.root)}
                for name, source in sources.items()}

    def save(self):
        current = digest(self.path) if self.path.is_file() else None
        if current != self._disk_hash:
            raise ValueError("The library changed in another window. Reopen it before saving.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"schema": "aerowhip_take_library_v1", "takes": self.entries},
                             indent=2, allow_nan=False)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            self._disk_hash = digest(self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def import_folder(self, folder, *, refresh_existing=True):
        folder = project_path(folder)
        if not folder.is_dir():
            raise ValueError(f"Recording folder not found: {folder}")
        if not folder.is_relative_to(self.root):
            raise ValueError('Import a folder within the selected recording collection.')
        if self.entries and self._verified_root != self.root:
            self.check_root(self.root)
        before = deepcopy(self.entries)
        report = {"added": 0, "duplicates": 0, "updated": 0, "changed": 0, "errors": []}
        for path in discover(folder):
            try:
                sources = source_files(path)
                key = sources["tracking"]["sha256"]
                existing = self.entries.get(key)
                if existing is not None:
                    report["duplicates"] += 1
                    same_location = self.resolve_path(existing) == path
                    relocated = not self.resolve_path(existing).is_file()
                    old_hashes = {name: source['sha256'] for name, source in existing['sources'].items()}
                    new_hashes = {name: source['sha256'] for name, source in sources.items()}
                    # Pure relocation preserves selections and historical provenance.
                    if (same_location or relocated) and old_hashes != new_hashes:
                        if not refresh_existing:
                            report["changed"] += 1
                            continue
                        group, grouping = session_name(path)
                        existing.update(path=recording_reference(path, root=self.root), name=path.stem,
                                        sources=self._portable_sources(sources),
                                        session=group, grouping=grouping, selection=None)
                        report["updated"] += 1
                    continue
                group, grouping = session_name(path)
                self.entries[key] = {"path": recording_reference(path, root=self.root), "name": path.stem,
                    "session": group, "grouping": grouping,
                    "sources": self._portable_sources(sources), "selection": None}
                report["added"] += 1
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                report["errors"].append(f"{path.name}: {exc}")
        try:
            if self.entries != before or not self.path.is_file():
                self.save()
        except Exception:
            self.entries = before
            raise
        return report

    def save_selection(self, key, take, first, last):
        if not 0 <= first < last < len(take.time):
            raise ValueError("Select at least two frames, with Start before End")
        entry = self.entries[key]
        if self.resolve_path(entry) != take.path.resolve():
            raise ValueError("Selection does not belong to this take")
        self.verify(entry)
        selection = {"first_index": int(first), "last_index": int(last),
            "first_frame": int(take.frame[first]), "last_frame": int(take.frame[last]),
            "source_start_s": float(take.source_time[first]),
            "source_end_s": float(take.source_time[last]),
            "elapsed_start_s": float(take.time[first]), "elapsed_end_s": float(take.time[last])}
        previous = entry.get("selection")
        entry["selection"] = selection
        try:
            self.save()
        except Exception:
            entry["selection"] = previous
            raise
