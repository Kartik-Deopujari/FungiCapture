"""
The plate map: which strain sits in which well.

Editable inside the application. The original workflow needed a CSV prepared
elsewhere, and a typo in it silently produced a colony named ``unknown`` that
was then hard to trace. Here the map is a first-class object with validation,
so problems surface while you are looking at the plate rather than three steps
later.

Two shapes are supported:

* **Well-keyed** (grid modes) - ``well_id -> strain``, e.g. ``D06 -> AMF270``
* **File-keyed** (dish modes) - ``filename -> strain``
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .project import PlateLayout

WELL_RE = re.compile(r"^([A-Z])(\d{1,2})$", re.IGNORECASE)

# Column names accepted when reading a CSV, in order of preference. Users
# arrive with spreadsheets from many sources, and rejecting a file over a
# capital letter helps nobody.
WELL_COLUMNS = ("well_id", "wellid", "well", "position", "index", "plate_index")
STRAIN_COLUMNS = ("strain", "species", "sample_id", "sample", "accession", "isolate", "name")
FILE_COLUMNS = ("filename", "file", "image", "path")
NOTE_COLUMNS = ("note", "notes", "comment", "comments")


@dataclass
class PlateMapEntry:
    """One well's assignment."""

    key: str
    """Well ID such as ``D06``, or a filename in dish modes."""

    strain: str = ""
    note: str = ""

    @property
    def is_empty(self) -> bool:
        """A well deliberately left blank - no colony expected there."""
        return not self.strain.strip()


@dataclass
class PlateMap:
    """
    A whole plate's assignments.

    Keys are normalised on the way in (``d6`` and ``D06`` become the same
    well), because inconsistent capitalisation and zero-padding is the single
    most common thing wrong with a hand-made plate map.
    """

    entries: dict[str, PlateMapEntry] = field(default_factory=dict)
    key_kind: str = "well"
    """``"well"`` or ``"file"``."""

    source_path: Path | None = None

    # ---------------- key handling ----------------

    @staticmethod
    def normalise_well(key: str) -> str:
        """``d6`` -> ``D06``. Anything unrecognised is returned upper-cased."""
        key = str(key).strip()
        match = WELL_RE.match(key)
        if not match:
            return key.upper()
        letter, number = match.groups()
        return f"{letter.upper()}{int(number):02d}"

    def _key(self, key: str) -> str:
        return self.normalise_well(key) if self.key_kind == "well" else str(key).strip()

    # ---------------- access ----------------

    def get(self, key: str) -> PlateMapEntry | None:
        return self.entries.get(self._key(key))

    def strain_for(self, key: str, default: str = "unknown") -> str:
        entry = self.get(key)
        return entry.strain if entry and entry.strain.strip() else default

    def set(self, key: str, strain: str, note: str = "") -> PlateMapEntry:
        normalised = self._key(key)
        entry = PlateMapEntry(key=normalised, strain=strain.strip(), note=note.strip())
        self.entries[normalised] = entry
        return entry

    def clear(self, key: str) -> None:
        self.entries.pop(self._key(key), None)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[PlateMapEntry]:
        return iter(self.entries.values())

    @property
    def assigned(self) -> list[PlateMapEntry]:
        return [e for e in self.entries.values() if not e.is_empty]

    # ---------------- construction ----------------

    @classmethod
    def blank_for_layout(cls, layout: PlateLayout) -> PlateMap:
        """An empty map with one row per well, ready to type into."""
        plate_map = cls(key_kind="well")
        for _, _, well_id in layout.iter_wells():
            plate_map.entries[well_id] = PlateMapEntry(key=well_id)
        return plate_map

    @classmethod
    def from_filenames(cls, filenames: Iterable[str]) -> PlateMap:
        """An empty map keyed by filename, for the dish modes."""
        plate_map = cls(key_kind="file")
        for name in filenames:
            plate_map.entries[str(name)] = PlateMapEntry(key=str(name))
        return plate_map

    # ---------------- CSV ----------------

    @classmethod
    def load_csv(cls, path: str | Path, key_kind: str = "well") -> PlateMap:
        """
        Read a plate map from CSV, accepting several common column names.

        Raises ``ValueError`` with a message naming the columns it did find,
        rather than returning an empty map - a silent empty map means every
        colony exports as ``unknown``.
        """
        path = Path(path)
        plate_map = cls(key_kind=key_kind, source_path=path)

        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path.name} has no header row.")

            lowered = {name.lower().strip(): name for name in reader.fieldnames}
            wanted = WELL_COLUMNS if key_kind == "well" else FILE_COLUMNS
            key_column = next((lowered[c] for c in wanted if c in lowered), None)
            strain_column = next((lowered[c] for c in STRAIN_COLUMNS if c in lowered), None)
            note_column = next((lowered[c] for c in NOTE_COLUMNS if c in lowered), None)

            if key_column is None or strain_column is None:
                raise ValueError(
                    f"{path.name} needs a key column (one of {', '.join(wanted)}) "
                    f"and a strain column (one of {', '.join(STRAIN_COLUMNS)}).\n"
                    f"Found: {', '.join(reader.fieldnames)}"
                )

            for row in reader:
                key = (row.get(key_column) or "").strip()
                if not key:
                    continue
                plate_map.set(
                    key,
                    (row.get(strain_column) or "").strip(),
                    (row.get(note_column) or "").strip() if note_column else "",
                )
        return plate_map

    def save_csv(self, path: str | Path) -> Path:
        """Write the map back out, so an edit made in the app is portable."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        key_name = "well_id" if self.key_kind == "well" else "filename"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([key_name, "strain", "note"])
            for entry in self.entries.values():
                writer.writerow([entry.key, entry.strain, entry.note])
        self.source_path = path
        return path

    # ---------------- validation ----------------

    def validate(self, layout: PlateLayout | None = None) -> list[str]:
        """
        Problems worth showing the user, in plain language.

        Returns an empty list when the map is fine. These are warnings, not
        errors: an unusual map is often intentional, so the app shows them and
        lets the user decide.
        """
        problems: list[str] = []

        if self.key_kind == "well" and layout is not None:
            valid = {well for _, _, well in layout.iter_wells()}
            unknown = [k for k in self.entries if k not in valid]
            if unknown:
                shown = ", ".join(sorted(unknown)[:8])
                problems.append(
                    f"{len(unknown)} well(s) are not on a {layout.rows}x{layout.cols} "
                    f"plate: {shown}{' ...' if len(unknown) > 8 else ''}"
                )
            missing = valid - set(self.entries)
            if missing and len(missing) < len(valid):
                problems.append(f"{len(missing)} well(s) have no entry at all.")

        blank = [e.key for e in self.entries.values() if e.is_empty]
        if blank and len(blank) < len(self.entries):
            problems.append(
                f"{len(blank)} well(s) have no strain and will export as 'unknown'."
            )

        seen: dict[str, list[str]] = {}
        for entry in self.assigned:
            seen.setdefault(entry.strain, []).append(entry.key)
        repeated = {s: k for s, k in seen.items() if len(k) > 1}
        if repeated:
            example = next(iter(repeated))
            problems.append(
                f"{len(repeated)} strain(s) appear in more than one well "
                f"(e.g. {example} in {', '.join(sorted(repeated[example])[:4])}). "
                "That is normal for replicates - check it is intended."
            )
        return problems

    def summary(self) -> str:
        total = len(self.entries)
        filled = len(self.assigned)
        distinct = len({e.strain for e in self.assigned})
        return f"{filled} of {total} wells assigned, {distinct} distinct strains"
