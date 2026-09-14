"""
The plate canvas: dragging a grid over a photograph.

Built on ``QGraphicsView`` rather than the original's ``QLabel``. That change
buys three things a QLabel cannot give:

* real zoom and pan, with the scene graph doing the transforms
* proper hit-testing, so grabbing a line does not mean hand-computing distances
* sane handling of a 32 MB photograph, because the view draws a scaled proxy
  while the geometry stays in normalised coordinates over the full-resolution
  original

Everything the user drags is stored normalised (0 to 1). The photograph's real
pixel size never enters the interaction code, so a layout set on one plate
applies unchanged to another shot at a different resolution.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from ...core.plate import BoxGeometry, GridGeometry, freeform_well_id

# Above this many pixels the view draws a downscaled proxy. The full-resolution
# array is kept for cropping; only the display is reduced.
MAX_DISPLAY_PIXELS = 12_000_000

GRAB_TOLERANCE_PX = 8
EDGE_TOLERANCE_PX = 10


def numpy_to_pixmap(rgb: np.ndarray) -> QPixmap:
    """Convert an RGB array to a QPixmap, downscaling if it is very large.

    The downscale uses OpenCV's ``INTER_AREA``, which is the right filter for
    shrinking and is many times faster than ``skimage.transform.resize`` on a
    50-megapixel plate photograph. The old skimage path could take several
    seconds *on the UI thread* for every image, which is what made switching
    between photographs feel like the program had hung.
    """
    array = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    height, width = array.shape[:2]

    if height * width > MAX_DISPLAY_PIXELS:
        scale = (MAX_DISPLAY_PIXELS / (height * width)) ** 0.5
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        try:
            import cv2

            array = np.ascontiguousarray(
                cv2.resize(array, new_size, interpolation=cv2.INTER_AREA)
            )
        except Exception:
            # Fallback: plain strided subsampling. Rougher, but instant and
            # dependency-free, so the view always shows *something* fast.
            step = max(1, int(round(1 / scale)))
            array = np.ascontiguousarray(array[::step, ::step])
        height, width = array.shape[:2]

    image = QImage(array.data, width, height, 3 * width, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class PlateCanvas(QGraphicsView):
    """
    Shows a plate photograph with an editable grid or set of boxes.

    Signals
    -------
    layout_changed
        Emitted whenever the user moves anything, so the window can enable
        Save and mark the project dirty.
    selection_changed(int)
        The index of the selected box, or -1.
    """

    layout_changed = Signal()
    selection_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor("#202020")))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.image: np.ndarray | None = None
        self._pixmap_item = None
        self._image_rect = QRectF()

        self.mode = "grid"
        self.grid: GridGeometry | None = None
        self.boxes: BoxGeometry | None = None
        self.show_labels = True
        self.flip_columns = False
        self.low_confidence_wells: set[int] = set()

        self._drag_kind = ""  # "", "lines", "boxes" or "rubber"
        self._drag_edge = ""
        self._drag_last: tuple[float, float] | None = None
        self._hover: tuple[str, int] | None = None

        # Selection. ``_selected`` is the primary box (kept for the label
        # highlight and the ``selected_box`` property); the two sets below hold
        # a multi-selection so several boxes or lines can be moved and resized
        # together.
        self._selected = -1
        self._selected_boxes: set[int] = set()
        self._selected_lines: set[tuple[str, int]] = set()

        # Rubber-band selection (drag over empty space to select many at once).
        self._rubber_origin: QPointF | None = None
        self._rubber_rect: QRectF | None = None
        self._rubber_additive = False

    # ---------------- image ----------------

    def set_image(self, rgb: np.ndarray) -> None:
        self.image = rgb
        self._scene.clear()
        pixmap = numpy_to_pixmap(rgb)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._image_rect = QRectF(0, 0, pixmap.width(), pixmap.height())
        self._scene.setSceneRect(self._image_rect)
        self.fit_to_window()
        self.viewport().update()

    def fit_to_window(self) -> None:
        if not self._image_rect.isEmpty():
            self.fitInView(self._image_rect, Qt.KeepAspectRatio)

    def zoom(self, factor: float) -> None:
        self.scale(factor, factor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Ctrl+wheel zooms; plain wheel scrolls, as users expect."""
        if event.modifiers() & Qt.ControlModifier:
            self.zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
            event.accept()
        else:
            super().wheelEvent(event)

    # ---------------- layout ----------------

    def set_grid(self, grid: GridGeometry) -> None:
        self.mode = "grid"
        self.grid = grid
        self._clear_selection()
        self.viewport().update()

    def set_boxes(self, boxes: BoxGeometry) -> None:
        self.mode = "box"
        self.boxes = boxes
        boxes.ensure_size()
        self._clear_selection()
        self.viewport().update()

    def add_box(self) -> None:
        """Place one more box (dish modes: one box per colony)."""
        if self.mode != "box" or self.boxes is None:
            return
        index = self.boxes.add_box()
        self._selected_boxes = {index}
        self._selected = index
        self.selection_changed.emit(index)
        self.layout_changed.emit()
        self.viewport().update()

    def delete_selected_boxes(self) -> None:
        """Remove the selected boxes. Only for a freeform (dish) layout, where
        a fixed row*col count is not being kept."""
        if (
            self.mode != "box"
            or self.boxes is None
            or not self.boxes.freeform
            or not self._selected_boxes
        ):
            return
        for index in sorted(self._selected_boxes, reverse=True):
            self.boxes.remove(index)
        self._clear_selection()
        self.layout_changed.emit()
        self.viewport().update()

    def _clear_selection(self) -> None:
        self._selected = -1
        self._selected_boxes.clear()
        self._selected_lines.clear()
        self.selection_changed.emit(-1)

    # ---------------- coordinates ----------------

    def _to_scene(self, x: float, y: float) -> QPointF:
        return QPointF(x * self._image_rect.width(), y * self._image_rect.height())

    def _to_normalised(self, point: QPointF) -> tuple[float, float]:
        if self._image_rect.isEmpty():
            return 0.0, 0.0
        return (
            min(max(point.x() / self._image_rect.width(), 0.0), 1.0),
            min(max(point.y() / self._image_rect.height(), 0.0), 1.0),
        )

    def _scene_tolerance(self, pixels: int) -> float:
        """Convert a screen tolerance into scene units at the current zoom.

        Without this, grab targets would shrink as you zoom in - the opposite
        of what a user expects.
        """
        scale = self.transform().m11() or 1.0
        return pixels / scale

    # ---------------- painting ----------------

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        if self.image is None or self._image_rect.isEmpty():
            return
        painter.save()
        if self.mode == "grid" and self.grid is not None:
            self._draw_grid(painter)
        elif self.mode == "box" and self.boxes is not None:
            self._draw_boxes(painter)
        if self._rubber_rect is not None:
            self._draw_rubber_band(painter)
        painter.restore()

    def _draw_rubber_band(self, painter: QPainter) -> None:
        assert self._rubber_rect is not None
        pen = QPen(QColor("#66ccff"), self._scene_tolerance(1.0))
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(102, 204, 255, 40)))
        painter.drawRect(self._rubber_rect)
        painter.setBrush(Qt.NoBrush)

    def _draw_grid(self, painter: QPainter) -> None:
        assert self.grid is not None
        width = self._scene_tolerance(1.5)
        normal = QPen(QColor("#33dd55"), width)
        highlighted = QPen(QColor("#ffdd33"), width * 2.2)

        for index, x in enumerate(self.grid.col_lines):
            active = ("col", index) in self._selected_lines or self._hover == ("col", index)
            painter.setPen(highlighted if active else normal)
            painter.drawLine(self._to_scene(x, 0.0), self._to_scene(x, 1.0))

        for index, y in enumerate(self.grid.row_lines):
            active = ("row", index) in self._selected_lines or self._hover == ("row", index)
            painter.setPen(highlighted if active else normal)
            painter.drawLine(self._to_scene(0.0, y), self._to_scene(1.0, y))

        if self.show_labels:
            self._draw_labels(
                painter,
                [
                    (
                        self._grid_label(r, c),
                        0.5 * (self.grid.col_lines[c] + self.grid.col_lines[c + 1]),
                        0.5 * (self.grid.row_lines[r] + self.grid.row_lines[r + 1]),
                    )
                    for r in range(self.grid.n_rows)
                    for c in range(self.grid.n_cols)
                ],
            )

    def _grid_label(self, row: int, col: int) -> str:
        assert self.grid is not None
        number = (self.grid.n_cols - col) if self.flip_columns else (col + 1)
        return f"{chr(ord('A') + row)}{number:02d}"

    def _draw_boxes(self, painter: QPainter) -> None:
        assert self.boxes is not None
        width = self._scene_tolerance(1.5)
        normal = QPen(QColor("#33dd55"), width)
        selected = QPen(QColor("#ffdd33"), width * 2.2)
        # Amber marks a well the auto-fit was unsure about, so the eye goes
        # straight to the ones that need checking.
        uncertain = QPen(QColor("#ff9933"), width * 2.0)

        for index, (x1, y1, x2, y2) in enumerate(self.boxes.boxes):
            if index in self._selected_boxes:
                painter.setPen(selected)
            elif index in self.low_confidence_wells:
                painter.setPen(uncertain)
            else:
                painter.setPen(normal)
            top_left = self._to_scene(x1, y1)
            bottom_right = self._to_scene(x2, y2)
            painter.drawRect(QRectF(top_left, bottom_right))

        if self.show_labels:
            labels = []
            for index, (x1, y1, x2, y2) in enumerate(self.boxes.boxes):
                if self.boxes.freeform:
                    text = freeform_well_id(index)
                else:
                    row, col = divmod(index, self.boxes.cols)
                    number = (self.boxes.cols - col) if self.flip_columns else (col + 1)
                    text = f"{chr(ord('A') + row)}{number:02d}"
                labels.append((text, 0.5 * (x1 + x2), 0.5 * (y1 + y2)))
            self._draw_labels(painter, labels)

    def _draw_labels(self, painter: QPainter, labels) -> None:
        font = QFont()
        font.setPointSizeF(max(6.0, self._scene_tolerance(11)))
        painter.setFont(font)
        painter.setPen(QPen(QColor("#ffffff")))
        for text, x, y in labels:
            painter.drawText(self._to_scene(x, y), text)

    # ---------------- interaction ----------------

    def mousePressEvent(self, event) -> None:
        if self.image is None or event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        point = self.mapToScene(event.position().toPoint())
        x, y = self._to_normalised(point)
        additive = bool(event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier))

        if self.mode == "grid" and self.grid is not None:
            self._press_grid(point, x, y, additive)
        elif self.mode == "box" and self.boxes is not None:
            self._press_box(point, x, y, additive)
        self.viewport().update()

    def _press_grid(self, point, x, y, additive: bool) -> None:
        hit = self._nearest_line(point)
        if hit is None:
            # Empty space: start a rubber band to select several lines.
            self._begin_rubber_band(point, additive)
            return
        # Toggle / set the selection.
        if additive:
            if hit in self._selected_lines:
                self._selected_lines.discard(hit)
            else:
                self._selected_lines.add(hit)
        elif hit not in self._selected_lines:
            self._selected_lines = {hit}
        # Drag the whole current selection when the grabbed line is part of it.
        if hit in self._selected_lines:
            self._drag_kind = "lines"
            self._drag_last = (x, y)

    def _press_box(self, point, x, y, additive: bool) -> None:
        assert self.boxes is not None
        index = self.boxes.index_at(x, y)
        if index is None:
            self._begin_rubber_band(point, additive)
            if not additive:
                self._selected_boxes.clear()
                self._selected = -1
                self.selection_changed.emit(-1)
            return

        if additive:
            if index in self._selected_boxes:
                self._selected_boxes.discard(index)
            else:
                self._selected_boxes.add(index)
            self._selected = index if index in self._selected_boxes else (
                next(iter(self._selected_boxes)) if self._selected_boxes else -1
            )
            self.selection_changed.emit(self._selected)
            # A modifier-click only edits the selection; it does not start a drag.
            return

        # Plain click: keep the group if the box is already in it, otherwise
        # select just this one. Either way, the drag moves the whole selection.
        if index not in self._selected_boxes:
            self._selected_boxes = {index}
        self._selected = index
        self.selection_changed.emit(self._selected)
        self._drag_edge = self._edge_at(index, x, y)
        self._drag_kind = "boxes"
        self._drag_last = (x, y)

    def _begin_rubber_band(self, point: QPointF, additive: bool) -> None:
        self._drag_kind = "rubber"
        self._rubber_origin = point
        self._rubber_rect = QRectF(point, point)
        self._rubber_additive = additive

    def mouseMoveEvent(self, event) -> None:
        if self.image is None:
            super().mouseMoveEvent(event)
            return

        point = self.mapToScene(event.position().toPoint())
        x, y = self._to_normalised(point)

        if not self._drag_kind:
            if self.mode == "grid" and self.grid is not None:
                previous = self._hover
                self._hover = self._nearest_line(point)
                if previous != self._hover:
                    self.setCursor(
                        Qt.SizeVerCursor
                        if self._hover and self._hover[0] == "row"
                        else Qt.SizeHorCursor
                        if self._hover
                        else Qt.ArrowCursor
                    )
                    self.viewport().update()
            super().mouseMoveEvent(event)
            return

        if self._drag_kind == "rubber":
            if self._rubber_origin is not None:
                self._rubber_rect = QRectF(self._rubber_origin, point).normalized()
                self.viewport().update()
            return

        if self._drag_last is None:
            return
        dx = x - self._drag_last[0]
        dy = y - self._drag_last[1]

        if self._drag_kind == "lines" and self.grid is not None:
            col_indices = [i for k, i in self._selected_lines if k == "col"]
            row_indices = [i for k, i in self._selected_lines if k == "row"]
            if col_indices:
                self.grid.move_lines("col", col_indices, dx)
            if row_indices:
                self.grid.move_lines("row", row_indices, dy)
        elif self._drag_kind == "boxes" and self.boxes is not None:
            if self._drag_edge == "move":
                self.boxes.move_many(self._selected_boxes, dx, dy)
            else:
                self.boxes.resize_many(self._selected_boxes, self._drag_edge, dx, dy)

        self._drag_last = (x, y)
        self.layout_changed.emit()
        self.viewport().update()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_kind == "rubber":
            self._apply_rubber_band()
        self._drag_kind = ""
        self._drag_edge = ""
        self._drag_last = None
        self._rubber_origin = None
        self._rubber_rect = None
        self.viewport().update()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        """Escape clears the selection; Ctrl+A selects everything."""
        if event.key() == Qt.Key_Escape:
            self._clear_selection()
            self.viewport().update()
            return
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected_boxes()
            return
        if event.key() == Qt.Key_A and event.modifiers() & Qt.ControlModifier:
            if self.mode == "box" and self.boxes is not None:
                self._selected_boxes = set(range(len(self.boxes.boxes)))
                self._selected = 0 if self.boxes.boxes else -1
                self.selection_changed.emit(self._selected)
            elif self.mode == "grid" and self.grid is not None:
                self._selected_lines = {
                    ("col", i) for i in range(len(self.grid.col_lines))
                } | {("row", i) for i in range(len(self.grid.row_lines))}
            self.viewport().update()
            return
        super().keyPressEvent(event)

    def _apply_rubber_band(self) -> None:
        """Turn the drawn rectangle into a selection of boxes or lines."""
        if self._rubber_rect is None or self._rubber_rect.isEmpty():
            return
        rx1, ry1 = self._to_normalised(self._rubber_rect.topLeft())
        rx2, ry2 = self._to_normalised(self._rubber_rect.bottomRight())
        rx1, rx2 = min(rx1, rx2), max(rx1, rx2)
        ry1, ry2 = min(ry1, ry2), max(ry1, ry2)

        if self.mode == "box" and self.boxes is not None:
            hits = {
                i
                for i, (x1, y1, x2, y2) in enumerate(self.boxes.boxes)
                if not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)
            }
            if self._rubber_additive:
                self._selected_boxes |= hits
            else:
                self._selected_boxes = hits
            self._selected = (
                self._selected if self._selected in self._selected_boxes
                else (min(self._selected_boxes) if self._selected_boxes else -1)
            )
            self.selection_changed.emit(self._selected)
        elif self.mode == "grid" and self.grid is not None:
            hits = {
                ("col", i)
                for i, xv in enumerate(self.grid.col_lines)
                if rx1 <= xv <= rx2
            } | {
                ("row", i)
                for i, yv in enumerate(self.grid.row_lines)
                if ry1 <= yv <= ry2
            }
            if self._rubber_additive:
                self._selected_lines |= hits
            else:
                self._selected_lines = hits

    def _nearest_line(self, point: QPointF) -> tuple[str, int] | None:
        """Which grid line is close enough to grab."""
        assert self.grid is not None
        tolerance = self._scene_tolerance(GRAB_TOLERANCE_PX)
        best: tuple[str, int] | None = None
        best_distance = tolerance

        for index, x in enumerate(self.grid.col_lines):
            distance = abs(self._to_scene(x, 0).x() - point.x())
            if distance < best_distance:
                best_distance, best = distance, ("col", index)

        for index, y in enumerate(self.grid.row_lines):
            distance = abs(self._to_scene(0, y).y() - point.y())
            if distance < best_distance:
                best_distance, best = distance, ("row", index)
        return best

    def _edge_at(self, index: int, x: float, y: float) -> str:
        """Whether the click grabbed an edge, a corner, or the body of a box."""
        assert self.boxes is not None
        x1, y1, x2, y2 = self.boxes.boxes[index]
        tolerance_x = self._scene_tolerance(EDGE_TOLERANCE_PX) / max(
            1.0, self._image_rect.width()
        )
        tolerance_y = self._scene_tolerance(EDGE_TOLERANCE_PX) / max(
            1.0, self._image_rect.height()
        )

        parts = ""
        if abs(y - y1) < tolerance_y:
            parts += "top"
        elif abs(y - y2) < tolerance_y:
            parts += "bottom"
        if abs(x - x1) < tolerance_x:
            parts += "left"
        elif abs(x - x2) < tolerance_x:
            parts += "right"
        return parts or "move"

    @property
    def selected_box(self) -> int:
        return self._selected
