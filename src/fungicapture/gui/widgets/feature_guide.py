"""
A popup that explains, in short, what every measured feature means.

The feature table has several hundred columns. Rather than leave a user to
guess what ``inside_glcm_correlation_mean`` is, this dialog lists every feature
grouped by family with a one-line description, and a search box to jump to one.

Descriptions come from :mod:`fungicapture.core.feature_glossary`, which builds
them from the feature name itself, so the guide always matches what is actually
measured.
"""

from __future__ import annotations

from html import escape

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QTextBrowser,
    QVBoxLayout,
)

from ...core.feature_glossary import (
    REPRESENTATIVE_FEATURES,
    group_features,
)


class FeatureGuideDialog(QDialog):
    """Scrollable, searchable glossary of every feature."""

    def __init__(self, columns=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("What the features mean")
        self.resize(720, 660)

        # Real feature columns when the feature step has run; otherwise a
        # representative set so the guide is still useful beforehand.
        names = [
            c for c in (columns or [])
            if c.lower() not in {"filename", "stem", "image_path", "mask_path"}
        ]
        self._have_real = bool(names)
        self._groups = group_features(names or REPRESENTATIVE_FEATURES)

        layout = QVBoxLayout(self)

        intro = QLabel(
            f"Short descriptions of the {sum(len(v) for _, v in self._groups)} "
            "features"
            + ("" if self._have_real else
               " (representative set — run the feature step to list every one)")
            + ". Type to filter."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#555;")
        layout.addWidget(intro)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search features, e.g. 'glcm', 'lab', 'zonation'…")
        self.search.textChanged.connect(self._render)
        layout.addWidget(self.search)

        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(False)
        layout.addWidget(self.view, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self._render("")

    def _render(self, needle: str) -> None:
        needle = needle.strip().lower()
        parts: list[str] = [
            "<style>"
            "h3{margin:14px 0 4px 0;color:#234;border-bottom:1px solid #ddd;}"
            "table{border-collapse:collapse;width:100%;}"
            "td{padding:2px 8px 2px 0;vertical-align:top;}"
            ".name{color:#0a5;font-family:monospace;white-space:nowrap;}"
            ".desc{color:#333;}"
            "</style>"
        ]
        shown = 0
        for title, items in self._groups:
            rows = [
                (name, desc)
                for name, desc in items
                if not needle or needle in name.lower() or needle in desc.lower()
            ]
            if not rows:
                continue
            parts.append(f"<h3>{escape(title)} &nbsp;<span style='color:#999'>"
                         f"({len(rows)})</span></h3><table>")
            for name, desc in rows:
                shown += 1
                parts.append(
                    f"<tr><td class='name'>{escape(name)}</td>"
                    f"<td class='desc'>{escape(desc)}</td></tr>"
                )
            parts.append("</table>")

        if shown == 0:
            parts.append("<p style='color:#888'>No features match that search.</p>")
        self.view.setHtml("".join(parts))
