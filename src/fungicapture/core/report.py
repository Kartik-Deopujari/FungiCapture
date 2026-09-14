"""
PDF reports.

Three modes, because three different jobs need three different documents:

**full**
    One page per colony plus a summary. This is what goes into a paper's
    supplementary material - a complete visual audit of every measurement.

**failures**
    Only the colonies marked as failed. Short enough to email to a
    collaborator with "these are the ones I am worried about".

**contact_sheet**
    A dense grid of thumbnails with mask outlines, 24 per page. For scanning a
    whole plate at a glance and spotting the odd one out.

Every report carries a provenance header: project, date, application version,
parameter hash and the key settings. A QC document without that is not
evidence, because you cannot tell later which settings produced it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .params import AnalysisParams
from .validate import PanelData, VerdictStore, make_panel

REPORT_MODES = ("full", "failures", "contact_sheet")


@dataclass
class ReportContext:
    """Header information stamped onto every page."""

    project_name: str
    provenance: dict[str, Any]
    mode: str
    n_colonies: int
    verdict_counts: dict[str, int]
    extra_notes: list[str]

    def header_lines(self) -> list[str]:
        p = self.provenance
        counts = self.verdict_counts
        return [
            f"FungiCapture QC report - {self.project_name}",
            f"mode: {self.mode}   colonies: {self.n_colonies}   "
            f"pass: {counts.get('pass', 0)}  fail: {counts.get('fail', 0)}  "
            f"unreviewed: {counts.get('unreviewed', 0)}",
            f"version: {p.get('fungicapture_version', '?')}   "
            f"params hash: {p.get('params_hash', '?')}   "
            f"generated: {p.get('generated_utc', '?')}",
            f"ring fraction: {p.get('ring_fraction')}   "
            f"GLCM levels: {p.get('glcm_levels')}   "
            f"LBP: P={p.get('lbp_points')} R={p.get('lbp_radius')}   "
            f"Lab white: {p.get('lab_illuminant')}",
        ]


def _summary_page(pdf, context: ReportContext) -> None:
    """The first page: what this document is and how it was made."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(11.7, 8.3))
    figure.text(0.06, 0.93, "FungiCapture QC report", fontsize=22, weight="bold")
    figure.text(0.06, 0.885, context.project_name, fontsize=13, color="#555555")

    y = 0.80
    for line in context.header_lines()[1:]:
        figure.text(0.06, y, line, fontsize=9.5, family="monospace")
        y -= 0.033

    y -= 0.03
    figure.text(0.06, y, "What to look for", fontsize=13, weight="bold")
    y -= 0.04
    guidance = [
        "1. Does the green mask follow the colony edge, including any fuzzy margin?",
        "2. Does the cyan ring sit on agar, not on a neighbouring colony?",
        "3. Is the L* map dark where the colony looks dark to your eye?",
        "4. Do the dominant colour swatches match what you see in the photograph?",
        "5. Does the radial profile show rings only where the colony really has them?",
        "",
        "A colony that fails any of these should be marked 'fail' in the Validate",
        "tab. Failed colonies are excluded from phenotype scoring, and the reason",
        "you type is kept with the record.",
    ]
    for line in guidance:
        figure.text(0.07, y, line, fontsize=10)
        y -= 0.030

    if context.extra_notes:
        y -= 0.02
        figure.text(0.06, y, "Notes from this run", fontsize=13, weight="bold")
        y -= 0.04
        for note in context.extra_notes[:12]:
            figure.text(0.07, y, f"- {note[:110]}", fontsize=9, color="#8a5a00")
            y -= 0.026

    figure.text(
        0.06, 0.04,
        "Segmentation used the Segment Anything Model 3 (SAM 3), released by Meta AI\n"
        "under the SAM License. FungiCapture is released under AGPL-3.0.",
        fontsize=7.5, color="#777777",
    )
    pdf.savefig(figure)
    plt.close(figure)


def write_report(
    path: str | Path,
    panels: Iterable[PanelData],
    params: AnalysisParams,
    context: ReportContext,
    *,
    mode: str = "full",
    verdicts: VerdictStore | None = None,
    progress: Callable[[int, int], bool] | None = None,
) -> Path:
    """
    Build a PDF.

    ``progress`` is called as ``progress(done, total)`` and may return False to
    cancel - a 400-colony report takes a while, and a user who cannot stop it
    will force-quit the application instead.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if mode not in REPORT_MODES:
        raise ValueError(f"Unknown report mode {mode!r}. Use one of {REPORT_MODES}.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = list(panels)

    if mode == "failures" and verdicts is not None:
        failed = verdicts.failed_keys()
        panels = [p for p in panels if p.key in failed]

    total = len(panels)
    with PdfPages(path) as pdf:
        _summary_page(pdf, context)

        if mode == "contact_sheet":
            _contact_sheet(pdf, panels, verdicts, progress)
        else:
            for index, data in enumerate(panels, start=1):
                verdict = verdicts.get(data.key) if verdicts else None
                extra = ""
                if verdict and verdict.status != "unreviewed":
                    extra = f"[{verdict.status.upper()}]"
                    if verdict.note:
                        extra += f" {verdict.note}"
                figure = make_panel(data, params, title_extra=extra)
                pdf.savefig(figure)
                plt.close(figure)
                if progress and not progress(index, total):
                    break

        info = pdf.infodict()
        info["Title"] = f"FungiCapture QC report - {context.project_name}"
        info["Subject"] = f"mode={mode}, params_hash={context.provenance.get('params_hash')}"
        info["Creator"] = "FungiCapture"

    return path


def _contact_sheet(
    pdf,
    panels: Sequence[PanelData],
    verdicts: VerdictStore | None,
    progress: Callable[[int, int], bool] | None,
    per_page: int = 24,
    columns: int = 6,
) -> None:
    """Thumbnails with mask outlines, for scanning a plate quickly."""
    import matplotlib.pyplot as plt

    rows = math.ceil(per_page / columns)
    total = len(panels)

    for start in range(0, total, per_page):
        chunk = panels[start : start + per_page]
        figure, axes = plt.subplots(rows, columns, figsize=(11.7, 8.3))
        axes = np.atleast_1d(axes).ravel()

        for axis in axes:
            axis.axis("off")

        for axis, data in zip(axes, chunk):
            axis.imshow(data.rgb)
            axis.contour(data.mask.astype(float), levels=[0.5],
                         colors="lime", linewidths=0.8)

            status = verdicts.get(data.key).status if verdicts else "unreviewed"
            colour = {"pass": "#228833", "fail": "#cc3311"}.get(status, "#999999")
            axis.set_title(data.key[:22], fontsize=6.5, color=colour)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color(colour)
                spine.set_linewidth(1.5 if status != "unreviewed" else 0.5)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.axis("on")

        figure.suptitle(
            f"Contact sheet - colonies {start + 1} to {min(start + per_page, total)} "
            f"of {total}   (green outline = mask; border = pass/fail)",
            fontsize=10,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(figure)
        plt.close(figure)

        if progress and not progress(min(start + per_page, total), total):
            break


def build_context(
    project,
    mode: str,
    n_colonies: int,
    verdicts: VerdictStore | None = None,
    extra_notes: Sequence[str] = (),
) -> ReportContext:
    """Assemble the header block from a project."""
    return ReportContext(
        project_name=project.name,
        provenance=project.provenance(),
        mode=mode,
        n_colonies=n_colonies,
        verdict_counts=verdicts.status_counts() if verdicts else {},
        extra_notes=list(extra_notes),
    )
