"""
The Project: one folder, one settings file, everything remembered.

Why this exists
---------------
In the original scripts every path was typed at the top of the file::

    INPUT_DIR  = Path('/media/kartik/8062bb90-.../plate_6_SDA_seg')
    OUTPUT_DIR = Path('/media/kartik/8062bb90-.../plate_6_SDA_vect')

That means the scripts only run on one machine, results cannot be reproduced by
a collaborator, and nothing records which settings produced which numbers.

A Project replaces all of it. It is a folder plus ``fungicapture.toml``. Open
it, work, close it. Two people opening the same project get the same numbers.
"""

from __future__ import annotations

import getpass
import hashlib
import platform
import socket
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import tomli_w

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from .params import AnalysisParams

PROJECT_FILENAME = "fungicapture.toml"
SCHEMA_VERSION = 1

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
    ".nef", ".cr2", ".cr3", ".arw", ".dng",
}
RAW_EXTENSIONS = {".nef", ".cr2", ".cr3", ".arw", ".dng"}


# --------------------------------------------------------------------------
# Imaging mode - the home screen question
# --------------------------------------------------------------------------


class ImagingMode(str, Enum):
    """
    What does one photograph contain?

    The whole downstream pipeline branches on this: whether a grid is needed,
    where names come from, and whether a run is one image or a folder.

    There are only two engines underneath. SINGLE_DISH and DISH_BATCH share the
    dish-detection engine; GRIDDED_PLATE and PLATE_BATCH share the gridding
    engine. The batch modes are the single modes run over a folder.
    """

    SINGLE_DISH = "single_dish"
    """One agar plate, one colony. No grid. Used for spot checks and for tuning
    settings before a long run."""

    DISH_BATCH = "dish_batch"
    """A folder of plates, one colony each. No grid."""

    GRIDDED_PLATE = "gridded_plate"
    """One large plate holding many colonies. Needs a grid."""

    PLATE_BATCH = "plate_batch"
    """A folder of large plates, all the same layout. Set the grid once and
    apply it to all. This is the main GWAS workflow."""

    @property
    def label(self) -> str:
        return {
            ImagingMode.SINGLE_DISH: "Single dish",
            ImagingMode.DISH_BATCH: "Dish batch",
            ImagingMode.GRIDDED_PLATE: "Gridded plate",
            ImagingMode.PLATE_BATCH: "Plate batch",
        }[self]

    @property
    def description(self) -> str:
        return {
            ImagingMode.SINGLE_DISH: "One agar plate, one colony. No grid needed.",
            ImagingMode.DISH_BATCH: "A folder of plates, one colony each. No grid needed.",
            ImagingMode.GRIDDED_PLATE: "One large plate, many colonies. Place a grid.",
            ImagingMode.PLATE_BATCH: "Many large plates, same layout. Set the grid once.",
        }[self]

    @property
    def needs_grid(self) -> bool:
        """True when the Plates tab must show a grid editor."""
        return self in (ImagingMode.GRIDDED_PLATE, ImagingMode.PLATE_BATCH)

    @property
    def is_batch(self) -> bool:
        """True when the mode runs over a folder rather than one image."""
        return self in (ImagingMode.DISH_BATCH, ImagingMode.PLATE_BATCH)

    @property
    def is_round(self) -> bool:
        """True for round plates (the dish engine), false for rectangular ones.

        The interface presents two plate shapes - round and rectangular/square -
        plus a batch checkbox. The four modes below are those two choices times
        the batch flag; this property recovers the shape from the mode.
        """
        return not self.needs_grid

    @property
    def shape_label(self) -> str:
        return "Round plates" if self.is_round else "Rectangular / square plates"

    @classmethod
    def for_shape(cls, round_plate: bool, batch: bool) -> ImagingMode:
        """The mode for a plate shape and whether a whole folder is processed."""
        if round_plate:
            return cls.DISH_BATCH if batch else cls.SINGLE_DISH
        return cls.PLATE_BATCH if batch else cls.GRIDDED_PLATE

    @property
    def engine(self) -> str:
        """Which of the two engines handles this mode."""
        return "grid" if self.needs_grid else "dish"


# --------------------------------------------------------------------------
# Plate layout
# --------------------------------------------------------------------------


@dataclass
class PlateLayout:
    """Logical plate shape. Only meaningful in the grid modes."""

    rows: int = 8
    cols: int = 12
    margin: float = 0.02
    """Fraction of the image kept clear at each edge when a grid is first
    created."""

    flip_columns: bool = False
    """Reverse column numbering, for plates photographed from the other side.
    False: A01 at the left. True: A01 at the right."""

    def well_id(self, row: int, col: int) -> str:
        """Well name such as ``A01`` or ``D06``.

        Rows are letters from A. Columns are 1-based and zero-padded to two
        digits, which keeps ``A02`` sorting before ``A10``.
        """
        if not 0 <= row < self.rows:
            raise ValueError(f"row {row} outside 0..{self.rows - 1}")
        if not 0 <= col < self.cols:
            raise ValueError(f"col {col} outside 0..{self.cols - 1}")
        letter = chr(ord("A") + row)
        number = (self.cols - col) if self.flip_columns else (col + 1)
        return f"{letter}{number:02d}"

    def iter_wells(self) -> Iterator[tuple[int, int, str]]:
        """Every well in row-major order, as ``(row, col, well_id)``."""
        for r in range(self.rows):
            for c in range(self.cols):
                yield r, c, self.well_id(r, c)

    @property
    def n_wells(self) -> int:
        return self.rows * self.cols


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


@dataclass
class RunRecord:
    """One entry in the project's run log.

    Written every time a pipeline step completes, so a result can always be
    traced to the settings, software version and machine that produced it.
    """

    step: str
    started: str
    finished: str
    n_inputs: int
    n_outputs: int
    n_errors: int
    app_version: str
    params_hash: str
    machine: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "started": self.started,
            "finished": self.finished,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "n_errors": self.n_errors,
            "app_version": self.app_version,
            "params_hash": self.params_hash,
            "machine": self.machine,
            "notes": self.notes,
        }


def _machine_fingerprint() -> str:
    """A short, non-identifying description of where a run happened."""
    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover - some sandboxes have no user
        user = "unknown"
    return f"{user}@{socket.gethostname()} {platform.system()} {platform.machine()}"


# --------------------------------------------------------------------------
# Project
# --------------------------------------------------------------------------


@dataclass
class Project:
    """
    A FungiCapture project.

    Layout on disk::

        my_project/
        ├── fungicapture.toml     settings, plate map reference, run log
        ├── images/               source photographs (or a link to them)
        ├── crops/                one image per colony
        ├── masks/                segmentation masks and overlays
        ├── vectors/              feature CSVs
        ├── qc/                   QC panels and PDF reports
        └── phenotypes/           GWAS-ready tables
    """

    root: Path
    name: str = "untitled"
    mode: ImagingMode = ImagingMode.PLATE_BATCH
    layout: PlateLayout = field(default_factory=PlateLayout)
    params: AnalysisParams = field(default_factory=AnalysisParams)

    image_dir: Path | None = None
    """Where the source photographs live. May sit outside the project folder,
    so a large image archive does not have to be copied."""

    plate_map_path: Path | None = None
    """CSV mapping well ID (grid modes) or filename (dish modes) to strain."""

    media_label: str = "SDA"
    """Growth medium written into exported filenames, e.g. SDA.

    Kept to 2-4 upper-case letters. The canonical colony ID is
    ``ACCESSION_MEDIA_WELL``, and the pattern that reads it back expects the
    middle token in that form. A label like 'medium 1' would be written into
    filenames and then fail to parse, leaving every colony with no medium - so
    it is normalised here rather than failing quietly later."""

    def normalise_media_label(self) -> list[str]:
        """Clean the medium label and report anything that had to change."""
        original = self.media_label
        cleaned = "".join(ch for ch in original if ch.isalpha()).upper()[:4]
        problems: list[str] = []

        if not cleaned:
            cleaned = "MED"
            problems.append(
                f"Medium label {original!r} has no letters, so 'MED' was used "
                "instead. Use 2-4 letters such as SDA, PDA or MEX."
            )
        elif len(cleaned) < 2:
            cleaned = f"{cleaned}X"
            problems.append(
                f"Medium label {original!r} is too short; using {cleaned!r}. "
                "Two to four letters work best."
            )
        elif cleaned != original:
            problems.append(
                f"Medium label {original!r} was adjusted to {cleaned!r} so it "
                "can be read back out of the exported filenames."
            )

        self.media_label = cleaned
        return problems

    folder_overrides: dict[str, str] = field(default_factory=dict)
    """Steps whose folder points somewhere other than the project default.

    Set by the folder row at the top of each tab. Keys are the names in
    ``DEFAULT_FOLDERS``."""

    schema_version: int = SCHEMA_VERSION
    created: str = ""
    runs: list[RunRecord] = field(default_factory=list)

    # ---------------- folders ----------------
    #
    # Each step's folders default to a sub-folder of the project, but any of
    # them can point somewhere else. That is what lets a step run on its own:
    # if you already have colony crops from an earlier run, or from a
    # completely different tool, point the segmentation step at that folder and
    # start there. Nothing forces the whole pipeline to run end to end.

    DEFAULT_FOLDERS: ClassVar[dict[str, str]] = {
        "crops": "crops",
        "masks": "masks",
        "vectors": "vectors",
        "qc": "qc",
        "phenotypes": "phenotypes",
    }

    @property
    def config_path(self) -> Path:
        return self.root / PROJECT_FILENAME

    def folder(self, name: str) -> Path:
        """Where a given kind of output lives, honouring any override."""
        if name not in self.DEFAULT_FOLDERS:
            raise KeyError(
                f"Unknown folder {name!r}. Expected one of "
                f"{', '.join(sorted(self.DEFAULT_FOLDERS))}."
            )
        override = self.folder_overrides.get(name)
        if override:
            path = Path(override).expanduser()
            return path if path.is_absolute() else (self.root / path)
        return self.root / self.DEFAULT_FOLDERS[name]

    def set_folder(self, name: str, path: str | Path | None) -> Path:
        """Point a step somewhere else, or back to the default with None."""
        if name not in self.DEFAULT_FOLDERS:
            raise KeyError(f"Unknown folder {name!r}")
        if path is None:
            self.folder_overrides.pop(name, None)
        else:
            self.folder_overrides[name] = str(Path(path).expanduser())
        return self.folder(name)

    def is_default_folder(self, name: str) -> bool:
        return name not in self.folder_overrides

    @property
    def crops_dir(self) -> Path:
        return self.folder("crops")

    @property
    def masks_dir(self) -> Path:
        return self.folder("masks")

    @property
    def vectors_dir(self) -> Path:
        return self.folder("vectors")

    @property
    def per_image_vectors_dir(self) -> Path:
        return self.vectors_dir / "per_image_vectors"

    @property
    def qc_dir(self) -> Path:
        return self.folder("qc")

    @property
    def phenotypes_dir(self) -> Path:
        return self.folder("phenotypes")

    @property
    def all_dirs(self) -> list[Path]:
        return [
            self.crops_dir,
            self.masks_dir,
            self.vectors_dir,
            self.per_image_vectors_dir,
            self.qc_dir,
            self.phenotypes_dir,
        ]

    # ---------------- lifecycle ----------------

    @classmethod
    def create(
        cls,
        root: str | Path,
        name: str,
        mode: ImagingMode,
        *,
        image_dir: str | Path | None = None,
        layout: PlateLayout | None = None,
        params: AnalysisParams | None = None,
        exist_ok: bool = False,
    ) -> Project:
        """Make a new project folder and write its settings file."""
        root = Path(root).expanduser().resolve()
        if root.exists() and any(root.iterdir()) and not exist_ok:
            if (root / PROJECT_FILENAME).exists():
                raise FileExistsError(
                    f"A project already exists at {root}. Use Project.load() to open it."
                )
        root.mkdir(parents=True, exist_ok=True)

        project = cls(
            root=root,
            name=name,
            mode=mode,
            layout=layout or PlateLayout(),
            params=params or AnalysisParams(),
            image_dir=Path(image_dir).expanduser().resolve() if image_dir else None,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        project.ensure_dirs()
        project.save()
        return project

    @classmethod
    def load(cls, path: str | Path) -> Project:
        """Open a project from its folder or its ``fungicapture.toml``."""
        path = Path(path).expanduser().resolve()
        config = path / PROJECT_FILENAME if path.is_dir() else path
        if not config.exists():
            raise FileNotFoundError(f"No {PROJECT_FILENAME} found at {path}")

        with open(config, "rb") as fh:
            data = tomllib.load(fh)

        version = data.get("schema_version", 1)
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"This project was written by a newer FungiCapture "
                f"(schema {version}, this build understands {SCHEMA_VERSION}). "
                f"Please update the application."
            )

        root = config.parent
        proj = data.get("project", {})
        layout_raw = data.get("layout", {})
        layout_fields = set(PlateLayout.__dataclass_fields__)

        def _abs(value: str | None) -> Path | None:
            if not value:
                return None
            p = Path(value)
            return p if p.is_absolute() else (root / p).resolve()

        return cls(
            root=root,
            name=proj.get("name", root.name),
            mode=ImagingMode(proj.get("mode", ImagingMode.PLATE_BATCH.value)),
            layout=PlateLayout(
                **{k: v for k, v in layout_raw.items() if k in layout_fields}
            ),
            params=AnalysisParams.from_dict(data.get("params", {})),
            image_dir=_abs(proj.get("image_dir")),
            plate_map_path=_abs(proj.get("plate_map")),
            media_label=proj.get("media_label", "media"),
            # Folder overrides are stored relative to the project root when they
            # sit inside it (see ``save``), so that moving or sharing the whole
            # project folder keeps every output pointing at the new location.
            # They are resolved back to absolute here so the rest of the
            # application, and the folder boxes in the interface, always work
            # with a real path.
            folder_overrides={
                str(k): str(_abs(v))
                for k, v in (data.get("folders") or {}).items()
                if v
            },
            schema_version=version,
            created=proj.get("created", ""),
            runs=[RunRecord(**r) for r in data.get("runs", [])],
        )

    def save(self) -> Path:
        """Write the settings file. Paths inside the project stay relative, so
        the whole folder can be moved or shared without breaking."""
        self.root.mkdir(parents=True, exist_ok=True)

        def _rel(p: Path | None) -> str | None:
            if p is None:
                return None
            try:
                return str(p.relative_to(self.root))
            except ValueError:
                return str(p)

        document: dict[str, Any] = {
            "schema_version": self.schema_version,
            "project": {
                "name": self.name,
                "mode": self.mode.value,
                "media_label": self.media_label,
                "created": self.created
                or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            "layout": {
                "rows": self.layout.rows,
                "cols": self.layout.cols,
                "margin": self.layout.margin,
                "flip_columns": self.layout.flip_columns,
            },
            "params": self.params.to_dict(),
            # Keep overrides that live inside the project relative, so the whole
            # folder can be moved or shared without the outputs breaking. An
            # override pointing somewhere else entirely stays absolute.
            "folders": {
                str(k): _rel(Path(str(v)).expanduser())
                for k, v in self.folder_overrides.items()
                if v
            },
            "runs": [r.to_dict() for r in self.runs],
        }
        for key, value in (
            ("image_dir", _rel(self.image_dir)),
            ("plate_map", _rel(self.plate_map_path)),
        ):
            if value is not None:
                document["project"][key] = value

        document = _strip_none(document)
        with open(self.config_path, "wb") as fh:
            tomli_w.dump(document, fh)
        return self.config_path

    def ensure_dirs(self) -> None:
        for d in self.all_dirs:
            d.mkdir(parents=True, exist_ok=True)

    # ---------------- provenance ----------------

    def params_hash(self) -> str:
        """Short stable hash of the analysis settings.

        Written into every output. If two result files carry different hashes,
        they were not produced by the same settings and must not be pooled.
        """
        blob = repr(_sorted_repr(self.params.to_dict())).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def log_run(
        self,
        step: str,
        started: datetime,
        n_inputs: int,
        n_outputs: int,
        n_errors: int = 0,
        notes: str = "",
        app_version: str = "0.1.0.dev0",
    ) -> RunRecord:
        """Record a completed pipeline step and save the project."""
        record = RunRecord(
            step=step,
            started=started.astimezone(timezone.utc).isoformat(timespec="seconds"),
            finished=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_errors=n_errors,
            app_version=app_version,
            params_hash=self.params_hash(),
            machine=_machine_fingerprint(),
            notes=notes,
        )
        self.runs.append(record)
        self.save()
        return record

    def provenance(self, app_version: str = "0.1.0.dev0") -> dict[str, Any]:
        """The block stamped into every CSV and PDF."""
        return {
            "fungicapture_version": app_version,
            "project_name": self.name,
            "imaging_mode": self.mode.value,
            "params_hash": self.params_hash(),
            "lab_illuminant": self.params.colour.lab_illuminant,
            "ring_fraction": self.params.ring.fraction,
            "glcm_levels": self.params.glcm.levels,
            "lbp_points": self.params.lbp.n_points,
            "lbp_radius": self.params.lbp.radius,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # ---------------- input discovery ----------------

    def find_images(self, directory: Path | None = None) -> list[Path]:
        """Source photographs, sorted, excluding anything the pipeline made.

        Masks, overlays and QC panels live under the project folder and must
        never be mistaken for inputs, which is a mistake the original scripts
        guarded against only by filename hints.
        """
        base = directory or self.image_dir
        if base is None:
            raise ValueError("No image directory set on this project.")
        base = Path(base)
        if not base.exists():
            raise FileNotFoundError(f"Image directory does not exist: {base}")

        derived = {d.resolve() for d in (self.masks_dir, self.qc_dir, self.crops_dir)}
        out: list[Path] = []
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if any(parent.resolve() in derived for parent in p.parents):
                continue
            out.append(p)
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Project(name={self.name!r}, mode={self.mode.value!r}, "
            f"root={str(self.root)!r}, runs={len(self.runs)})"
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _strip_none(obj: Any) -> Any:
    """TOML cannot store None. Drop those keys rather than writing empty
    strings, which would be read back as a real value."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_strip_none(v) for v in obj if v is not None]
    return obj


def _sorted_repr(obj: Any) -> Any:
    """Recursively sort dictionaries so the hash does not depend on key
    ordering."""
    if isinstance(obj, dict):
        return {k: _sorted_repr(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_sorted_repr(v) for v in obj]
    return obj
