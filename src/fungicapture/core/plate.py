"""
Plate geometry: turning one photograph of a multi-well plate into one image
per colony.

Two layout models, both stored in **normalised coordinates** (0 to 1 across the
image) rather than pixels. That means a layout set on a downscaled preview
applies unchanged to the full-resolution original, and a layout can be reused
across plates photographed at different resolutions.

Grid
    A set of straight horizontal and vertical lines. Simple and fast, but it
    assumes the plate is rectangular and not rotated. Moving one line changes
    two neighbouring wells.

Boxes
    One independent rectangle per well. Handles a crooked plate or unevenly
    spaced colonies, at the cost of more to adjust.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from .project import PlateLayout

# --------------------------------------------------------------------------
# Grid layout
# --------------------------------------------------------------------------


@dataclass
class GridGeometry:
    """
    Straight grid lines in normalised coordinates.

    ``row_lines`` has ``rows + 1`` entries and ``col_lines`` has ``cols + 1``;
    the extra entry is the closing edge. Both are kept sorted.
    """

    row_lines: list[float] = field(default_factory=list)
    col_lines: list[float] = field(default_factory=list)

    @classmethod
    def regular(cls, rows: int, cols: int, margin: float = 0.02) -> GridGeometry:
        """An evenly spaced grid inset from the image edge by ``margin``."""
        return cls(
            row_lines=np.linspace(margin, 1.0 - margin, rows + 1).tolist(),
            col_lines=np.linspace(margin, 1.0 - margin, cols + 1).tolist(),
        )

    @property
    def n_rows(self) -> int:
        return max(0, len(self.row_lines) - 1)

    @property
    def n_cols(self) -> int:
        return max(0, len(self.col_lines) - 1)

    def cell(self, row: int, col: int) -> tuple[float, float, float, float]:
        """One cell as ``(x1, y1, x2, y2)`` in normalised coordinates."""
        return (
            self.col_lines[col],
            self.row_lines[row],
            self.col_lines[col + 1],
            self.row_lines[row + 1],
        )

    def move_line(self, kind: str, index: int, value: float, gap: float = 0.005) -> None:
        """
        Move one grid line, keeping it between its neighbours.

        ``gap`` stops two lines being dragged on top of each other, which would
        produce a zero-width well and a crash further down.
        """
        lines = self.row_lines if kind == "row" else self.col_lines
        if not 0 <= index < len(lines):
            raise IndexError(f"No {kind} line at index {index}")
        low = lines[index - 1] + gap if index > 0 else 0.0
        high = lines[index + 1] - gap if index < len(lines) - 1 else 1.0
        lines[index] = float(min(max(value, low), high))

    def move_lines(
        self, kind: str, indices: Sequence[int], delta: float, gap: float = 0.005
    ) -> None:
        """
        Move several grid lines of one kind together, keeping their spacing.

        Every selected line shifts by the same ``delta`` so the block stays
        rigid. The shift is clamped so no selected line crosses a *non-selected*
        neighbour (or the image edge); because they all move by the same amount,
        selected lines can never cross each other. This is what lets a whole
        block of the grid be nudged in one drag.
        """
        lines = self.row_lines if kind == "row" else self.col_lines
        chosen = sorted(i for i in set(indices) if 0 <= i < len(lines))
        if not chosen:
            return
        selected = set(chosen)

        allowed_low = -1.0  # most negative delta permitted
        allowed_high = 1.0  # most positive delta permitted
        for i in chosen:
            low = 0.0
            for j in range(i - 1, -1, -1):
                if j not in selected:
                    low = lines[j] + gap
                    break
            high = 1.0
            for j in range(i + 1, len(lines)):
                if j not in selected:
                    high = lines[j] - gap
                    break
            allowed_low = max(allowed_low, low - lines[i])
            allowed_high = min(allowed_high, high - lines[i])

        step = min(max(delta, allowed_low), allowed_high)
        for i in chosen:
            lines[i] = float(lines[i] + step)

    def insert_line(self, kind: str) -> int:
        """Split the widest gap in two. Returns the new line's index."""
        lines = self.row_lines if kind == "row" else self.col_lines
        if len(lines) < 2:
            raise ValueError("Need at least two lines before inserting another.")
        widths = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
        widest = int(np.argmax(widths))
        position = 0.5 * (lines[widest] + lines[widest + 1])
        lines.insert(widest + 1, position)
        return widest + 1

    def remove_line(self, kind: str, index: int) -> bool:
        """Remove an interior line. Outer edges cannot be removed."""
        lines = self.row_lines if kind == "row" else self.col_lines
        if not 0 < index < len(lines) - 1:
            return False
        lines.pop(index)
        return True

    def to_dict(self) -> dict:
        return {"row_lines": list(self.row_lines), "col_lines": list(self.col_lines)}

    @classmethod
    def from_dict(cls, data: dict) -> GridGeometry:
        return cls(
            row_lines=[float(v) for v in data.get("row_lines", [])],
            col_lines=[float(v) for v in data.get("col_lines", [])],
        )


# --------------------------------------------------------------------------
# Box layout
# --------------------------------------------------------------------------


@dataclass
class BoxGeometry:
    """
    One rectangle per well, in normalised coordinates.

    Boxes are stored row-major: ``index = row * cols + col``. Keeping that
    order fixed is what lets a box be mapped back to a well ID.
    """

    rows: int = 8
    cols: int = 12
    boxes: list[list[float]] = field(default_factory=list)
    freeform: bool = False
    """True for a hand-placed set of boxes with no fixed row/column count -
    used by the dish modes, where the user starts from one large box and adds
    one per colony. In this mode the row*col invariant does not apply and boxes
    are named by position in the list (see ``freeform_well_id``)."""

    @classmethod
    def single(cls, margin: float = 0.08) -> BoxGeometry:
        """One large box covering most of the image.

        The starting point for a dish plate whose colonies are boxed by hand:
        one colony fills the plate, and the user adds more boxes if there are
        several colonies on the dish.
        """
        return cls(
            rows=1,
            cols=1,
            boxes=[[margin, margin, 1.0 - margin, 1.0 - margin]],
            freeform=True,
        )

    @classmethod
    def from_grid(
        cls, grid: GridGeometry, rows: int, cols: int, fill: float = 0.5
    ) -> BoxGeometry:
        """
        Place one box inside each grid cell, centred, at ``fill`` of the cell
        size.

        Starting smaller than the cell is deliberate: a box that already
        touched its neighbours would be awkward to grab and drag.
        """
        boxes: list[list[float]] = []
        for r in range(rows):
            for c in range(cols):
                x1, y1, x2, y2 = grid.cell(r, c)
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                half_w = 0.5 * fill * (x2 - x1)
                half_h = 0.5 * fill * (y2 - y1)
                boxes.append([cx - half_w, cy - half_h, cx + half_w, cy + half_h])
        return cls(rows=rows, cols=cols, boxes=boxes)

    @classmethod
    def regular(
        cls, rows: int, cols: int, margin: float = 0.02, fill: float = 0.5
    ) -> BoxGeometry:
        return cls.from_grid(GridGeometry.regular(rows, cols, margin), rows, cols, fill)

    def ensure_size(self) -> None:
        """Pad or trim so there is exactly one box per well.

        A no-op for a freeform layout, where the number of boxes is whatever
        the user placed and there is no ``rows * cols`` target to match.
        """
        if self.freeform:
            return
        target = self.rows * self.cols
        while len(self.boxes) < target:
            template = self.boxes[-1] if self.boxes else [0.4, 0.4, 0.6, 0.6]
            self.boxes.append(list(template))
        if len(self.boxes) > target:
            del self.boxes[target:]

    def add_box(self, box: Sequence[float] | None = None) -> int:
        """Append one box and return its index.

        With no box given, a small one is placed near the centre, nudged by a
        little each time so successive boxes do not stack exactly on top of one
        another and stay easy to grab.
        """
        if box is None:
            step = 0.035 * (len(self.boxes) % 6)
            cx, cy = 0.40 + step, 0.40 + step
            box = [cx, cy, min(cx + 0.18, 1.0), min(cy + 0.18, 1.0)]
        x1, y1, x2, y2 = (float(v) for v in box)
        self.boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        return len(self.boxes) - 1

    def remove(self, index: int) -> bool:
        """Delete one box. Returns True when a box was removed."""
        if 0 <= index < len(self.boxes):
            del self.boxes[index]
            return True
        return False

    def index_at(self, x: float, y: float) -> int | None:
        """Which box contains this point, topmost first."""
        for index in range(len(self.boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self.boxes[index]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return index
        return None

    def move(self, index: int, dx: float, dy: float) -> None:
        """Slide a box, keeping it inside the image."""
        x1, y1, x2, y2 = self.boxes[index]
        w, h = x2 - x1, y2 - y1
        nx1 = min(max(x1 + dx, 0.0), 1.0 - w)
        ny1 = min(max(y1 + dy, 0.0), 1.0 - h)
        self.boxes[index] = [nx1, ny1, nx1 + w, ny1 + h]

    def move_many(self, indices: Sequence[int], dx: float, dy: float) -> None:
        """
        Slide several boxes together by the same offset, keeping them rigid.

        The offset is clamped once for the whole group so the block stops at the
        image edge without deforming - if each box were clamped on its own, the
        ones already against the edge would lag behind and the layout would
        smear.
        """
        chosen = [i for i in set(indices) if 0 <= i < len(self.boxes)]
        if not chosen:
            return
        min_x = min(self.boxes[i][0] for i in chosen)
        max_x = max(self.boxes[i][2] for i in chosen)
        min_y = min(self.boxes[i][1] for i in chosen)
        max_y = max(self.boxes[i][3] for i in chosen)
        dx = min(max(dx, -min_x), 1.0 - max_x)
        dy = min(max(dy, -min_y), 1.0 - max_y)
        for i in chosen:
            x1, y1, x2, y2 = self.boxes[i]
            self.boxes[i] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    def resize_many(
        self,
        indices: Sequence[int],
        edge: str,
        dx: float,
        dy: float,
        minimum: float = 0.005,
    ) -> None:
        """
        Drag the same edge or corner of several boxes at once.

        The named edges of every selected box shift by the same ``(dx, dy)``.
        The shift is clamped for the group so the narrowest box never collapses
        below ``minimum`` and nothing leaves the image, which keeps every box in
        the selection valid.
        """
        chosen = [i for i in set(indices) if 0 <= i < len(self.boxes)]
        if not chosen:
            return

        if "left" in edge:
            lo = max(-self.boxes[i][0] for i in chosen)
            hi = min(self.boxes[i][2] - minimum - self.boxes[i][0] for i in chosen)
            dx_left = min(max(dx, lo), hi)
        if "right" in edge:
            lo = max(self.boxes[i][0] + minimum - self.boxes[i][2] for i in chosen)
            hi = min(1.0 - self.boxes[i][2] for i in chosen)
            dx_right = min(max(dx, lo), hi)
        if "top" in edge:
            lo = max(-self.boxes[i][1] for i in chosen)
            hi = min(self.boxes[i][3] - minimum - self.boxes[i][1] for i in chosen)
            dy_top = min(max(dy, lo), hi)
        if "bottom" in edge:
            lo = max(self.boxes[i][1] + minimum - self.boxes[i][3] for i in chosen)
            hi = min(1.0 - self.boxes[i][3] for i in chosen)
            dy_bottom = min(max(dy, lo), hi)

        for i in chosen:
            x1, y1, x2, y2 = self.boxes[i]
            if "left" in edge:
                x1 += dx_left
            if "right" in edge:
                x2 += dx_right
            if "top" in edge:
                y1 += dy_top
            if "bottom" in edge:
                y2 += dy_bottom
            self.boxes[i] = [x1, y1, x2, y2]

    def resize(self, index: int, edge: str, x: float, y: float, minimum: float = 0.005) -> None:
        """Drag one edge or corner. ``edge`` may combine 'left'/'right' with
        'top'/'bottom', e.g. ``"topleft"``."""
        x1, y1, x2, y2 = self.boxes[index]
        if "left" in edge:
            x1 = min(max(x, 0.0), x2 - minimum)
        if "right" in edge:
            x2 = max(min(x, 1.0), x1 + minimum)
        if "top" in edge:
            y1 = min(max(y, 0.0), y2 - minimum)
        if "bottom" in edge:
            y2 = max(min(y, 1.0), y1 + minimum)
        self.boxes[index] = [x1, y1, x2, y2]

    def set_box(self, index: int, box: Sequence[float]) -> None:
        x1, y1, x2, y2 = (float(v) for v in box)
        self.boxes[index] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "boxes": [list(b) for b in self.boxes],
            "freeform": self.freeform,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BoxGeometry:
        return cls(
            rows=int(data.get("rows", 8)),
            cols=int(data.get("cols", 12)),
            boxes=[[float(v) for v in b] for b in data.get("boxes", [])],
            freeform=bool(data.get("freeform", False)),
        )


# --------------------------------------------------------------------------
# Cropping
# --------------------------------------------------------------------------


@dataclass
class WellCrop:
    """One extracted well, with everything needed to name and audit it."""

    row: int
    col: int
    well_id: str
    image: np.ndarray
    bbox_px: tuple[int, int, int, int]
    """``(x1, y1, x2, y2)`` in pixels of the source image."""
    touches_edge: bool
    """True when the well runs off the side of the photograph."""


def crop_normalised(
    image: np.ndarray, box: Sequence[float]
) -> tuple[np.ndarray | None, tuple[int, int, int, int], bool]:
    """
    Cut a normalised rectangle out of an image.

    Returns ``(crop, bbox_px, touches_edge)``. ``crop`` is None only when the
    rectangle has no area at all inside the image.
    """
    height, width = image.shape[:2]
    x1n, y1n, x2n, y2n = box

    x1 = int(round(min(x1n, x2n) * width))
    x2 = int(round(max(x1n, x2n) * width))
    y1 = int(round(min(y1n, y2n) * height))
    y2 = int(round(max(y1n, y2n) * height))

    cx1, cx2 = max(0, min(width, x1)), max(0, min(width, x2))
    cy1, cy2 = max(0, min(height, y1)), max(0, min(height, y2))

    touches_edge = (x1 < 0) or (y1 < 0) or (x2 > width) or (y2 > height)
    if cx2 <= cx1 or cy2 <= cy1:
        return None, (cx1, cy1, cx2, cy2), True
    return image[cy1:cy2, cx1:cx2].copy(), (cx1, cy1, cx2, cy2), touches_edge


def iter_grid_crops(
    image: np.ndarray,
    grid: GridGeometry,
    layout: PlateLayout,
    *,
    skip_edge_wells: bool = False,
) -> Iterator[WellCrop]:
    """
    Every well of a grid layout.

    ``skip_edge_wells`` reproduces the original ``require_enclosed=True``
    behaviour. It is **off by default** here: the original dropped those wells
    silently, and a user could lose a whole outer row without being told. The
    caller now decides, and ``touches_edge`` is reported either way.
    """
    for row in range(min(grid.n_rows, layout.rows)):
        for col in range(min(grid.n_cols, layout.cols)):
            crop, bbox, edge = crop_normalised(image, grid.cell(row, col))
            if crop is None or (skip_edge_wells and edge):
                continue
            yield WellCrop(row, col, layout.well_id(row, col), crop, bbox, edge)


def iter_box_crops(
    image: np.ndarray,
    boxes: BoxGeometry,
    layout: PlateLayout,
    *,
    skip_edge_wells: bool = False,
) -> Iterator[WellCrop]:
    """Every well of a box layout."""
    boxes.ensure_size()
    for index, box in enumerate(boxes.boxes):
        row, col = divmod(index, boxes.cols)
        if row >= layout.rows or col >= layout.cols:
            continue
        crop, bbox, edge = crop_normalised(image, box)
        if crop is None or (skip_edge_wells and edge):
            continue
        yield WellCrop(row, col, layout.well_id(row, col), crop, bbox, edge)


def freeform_well_id(index: int) -> str:
    """Name a hand-placed box by its position in the list: ``A01``, ``A02``, …

    Numbers run 01 to 99, then the letter advances (``B01``), which keeps every
    name inside the ``[A-H]\\d{2}`` shape the rest of the pipeline recognises
    while still allowing far more colonies than a dish would ever hold.
    """
    letter = chr(ord("A") + min(index // 99, 7))
    return f"{letter}{index % 99 + 1:02d}"


def iter_freeform_box_crops(
    image: np.ndarray,
    boxes: BoxGeometry,
    *,
    skip_edge_wells: bool = False,
) -> Iterator[WellCrop]:
    """Every hand-placed box, named by list position rather than a grid cell.

    Used by the dish modes, where boxes are added one per colony and there is
    no fixed row/column layout to map them onto.
    """
    for index, box in enumerate(boxes.boxes):
        crop, bbox, edge = crop_normalised(image, box)
        if crop is None or (skip_edge_wells and edge):
            continue
        yield WellCrop(0, index, freeform_well_id(index), crop, bbox, edge)


def crop_filename(
    well_id: str,
    strain: str | None,
    media: str,
    *,
    unknown: str = "unknown",
    extension: str = ".png",
) -> str:
    """
    Build an export filename: ``STRAIN_MEDIA_WELL.png``.

    Matches the convention the rest of the pipeline parses, so a crop written
    here is recognised by the feature extractor without extra configuration.
    """
    safe = (strain or unknown).strip().replace(" ", "-").replace("/", "-")
    return f"{safe}_{media}_{well_id}{extension}"
