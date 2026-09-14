"""
Tabs 2, 3 and 5 - Segment, Features and Phenotypes.

These share a shape: choose a few settings, press a button, watch a long job
run, read the result. So they share a base class rather than repeating the same
layout code three times.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.project import Project
from ..widgets.folder_row import FolderPanel, count_images


class StepTab(QWidget):
    """A folder panel, a settings panel, a Run button, and a log.

    Every step names its own input and output folder, so any step can be run on
    its own against files produced earlier or by another tool. Nothing forces
    the whole pipeline to run end to end.
    """

    request_run = Signal(str, dict)
    status_message = Signal(str)

    step_name = ""
    run_label = "Run"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Project | None = None

        layout = QVBoxLayout(self)

        self.folders = FolderPanel()
        self.build_folders()
        self.folders.finish()
        self.folders.changed.connect(self._folders_changed)
        layout.addWidget(self.folders)

        self.settings_box = QGroupBox("Settings")
        self.form = QFormLayout(self.settings_box)
        layout.addWidget(self.settings_box)

        self.build_settings()

        row = QHBoxLayout()
        self.run_button = QPushButton(self.run_label)
        self.run_button.clicked.connect(self._run)
        row.addWidget(self.run_button)
        row.addStretch()
        layout.addLayout(row)

        layout.addWidget(QLabel("Log"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

    def build_folders(self) -> None:  # pragma: no cover - overridden
        pass

    def build_settings(self) -> None:  # pragma: no cover - overridden
        pass

    def collect(self) -> dict:  # pragma: no cover - overridden
        return {}

    def folder_options(self) -> dict:
        """Folder choices, as keyword arguments for the pipeline step."""
        return {}

    def _folders_changed(self) -> None:
        """Remember a changed folder in the project, so it survives a restart."""
        self.folders.refresh_status()
        if self.project is None:
            return
        for key, field in self.folders._fields.items():
            if key in self.project.DEFAULT_FOLDERS:
                self.project.set_folder(key, field.value())
        self.project.save()

    def set_project(self, project: Project) -> None:
        self.project = project
        self.sync_folders()

    def sync_folders(self) -> None:
        """Show the project's folders in the boxes."""
        if self.project is None:
            return
        for key, field in self.folders._fields.items():
            if key in self.project.DEFAULT_FOLDERS:
                field.set_default(self.project.root / self.project.DEFAULT_FOLDERS[key])
                field.set_value(self.project.folder_overrides.get(key))
        self.folders.refresh_status()

    def _run(self) -> None:
        if self.project is None:
            return
        self.log.clear()
        options = dict(self.collect())
        options.update(self.folder_options())
        self.request_run.emit(self.step_name, options)

    def append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def show_result(self, result) -> None:
        """Print what happened, including warnings. Warnings are the useful
        part - a step that 'succeeded' while quietly falling back to a
        different backend is exactly what a user needs told."""
        self.append_log(result.summary())
        for warning in result.warnings:
            self.append_log(f"  note: {warning}")
        for name, error in result.errors[:20]:
            self.append_log(f"  error: {Path(name).name}: {error}")
        if len(result.errors) > 20:
            self.append_log(f"  ... and {len(result.errors) - 20} more errors")


class SegmentTab(StepTab):
    """Tab 2 - find the colony outline in every crop."""

    step_name = "segment"
    run_label = "Run segmentation"

    def build_folders(self) -> None:
        self.folders.add(
            "crops", "Colony images:",
            tooltip=(
                "Folder of one-colony images to segment. Defaults to the crops "
                "this project made, but you can point it at crops from an "
                "earlier run or another tool and start here."
            ),
        )
        self.folders.add(
            "masks", "Save masks to:",
            tooltip="Where the outlines are written.",
        )
        self.folders.refresh_status(self._count)

    def _count(self) -> str:
        folder = self.folders["crops"].effective()
        if folder is None:
            return ""
        n = count_images(folder)
        if not Path(folder).exists():
            return f"That folder does not exist yet: {folder}"
        return f"{n} colony image(s) found in {folder}"

    def folder_options(self) -> dict:
        return {
            "input_dir": self.folders["crops"].effective(),
            "output_dir": self.folders["masks"].effective(),
        }

    def build_settings(self) -> None:
        # What will actually run, stated plainly. The old version said nothing,
        # so a machine with no PyTorch silently used the classical segmenter
        # and looked broken.
        self.env_status = QLabel("checking…")
        self.env_status.setWordWrap(True)
        setup_button = QPushButton("Segmentation setup…")
        setup_button.setToolTip(
            "Check what hardware and software is available, and install the "
            "SAM 3 model if it is missing."
        )
        setup_button.clicked.connect(self._open_setup)
        status_row = QHBoxLayout()
        status_row.addWidget(self.env_status, 1)
        status_row.addWidget(setup_button)
        status_widget = QWidget()
        status_widget.setLayout(status_row)
        self.form.addRow("Status:", status_widget)

        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "mps", "cpu"])
        self.device.setToolTip(
            "auto picks the GPU when one is usable and the processor "
            "otherwise. Choose explicitly to override."
        )
        self.form.addRow("Run on:", self.device)

        self.backend = QComboBox()
        self.backend.addItems(["auto", "sam3", "classical"])
        self.backend.setToolTip(
            "auto uses SAM 3 when the weights are present and falls back to the "
            "classical segmenter otherwise, telling you which it used."
        )
        self.form.addRow("Backend:", self.backend)

        self.weights = QLabel("not found")
        weights_button = QPushButton("Install weights…")
        weights_button.clicked.connect(self._install_weights)
        row = QHBoxLayout()
        row.addWidget(self.weights, 1)
        row.addWidget(weights_button)
        container = QWidget()
        container.setLayout(row)
        self.form.addRow("SAM 3 weights:", container)

        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.01, 0.99)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(0.25)
        self.form.addRow("Confidence:", self.confidence)

        self.overlays = QCheckBox("Save overlay images for quick checking")
        self.overlays.setChecked(True)
        self.form.addRow("", self.overlays)

        self.prompts = QPlainTextEdit()
        self.prompts.setMaximumHeight(120)
        self.prompts.setToolTip(
            "One prompt per line. Blank line starts a new group. Groups are "
            "merged with OR, so the colony core and its filamentous margin can "
            "be described separately."
        )
        self.form.addRow("Prompts:", self.prompts)

        self._refresh_weights()

    def _refresh_weights(self) -> None:
        from ...core.segment import find_model

        path = find_model()
        self.weights.setText(str(path) if path else "not found — will use classical")

    def _install_weights(self) -> None:
        from ...core.segment import SAM3_LICENCE_NOTE, SAM3_MODEL_PAGE, install_model

        QMessageBox.information(
            self, "SAM 3 weights",
            f"{SAM3_LICENCE_NOTE}\n\nDownload page:\n{SAM3_MODEL_PAGE}\n\n"
            "Once you have the file, choose it on the next screen.",
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Select the SAM 3 weights file", "", "Model weights (*.pt)"
        )
        if not path:
            return
        try:
            installed = install_model(path)
        except Exception as error:
            QMessageBox.critical(self, "Could not install the weights", str(error))
            return
        QMessageBox.information(self, "Installed", f"Weights cached at:\n{installed}")
        self._refresh_weights()

    def _open_setup(self) -> None:
        from .setup_dialog import SegmentationSetupDialog

        dialog = SegmentationSetupDialog(self)
        dialog.exec()
        self.refresh_environment()
        self._refresh_weights()

    def refresh_environment(self) -> None:
        """Show in one line what segmentation will actually do."""
        try:
            from ...core.setup_env import check_environment

            report = check_environment()
        except Exception as error:  # pragma: no cover
            self.env_status.setText(f"Could not check: {error}")
            return

        text = report.summary()
        if report.can_run_sam3:
            self.env_status.setStyleSheet("color:#1a6;")
        else:
            self.env_status.setStyleSheet("color:#a70;")
            text += "  —  press Segmentation setup to fix"
        self.env_status.setText(text)

    def set_project(self, project: Project) -> None:
        super().set_project(project)
        self.refresh_environment()
        self.device.setCurrentText(project.params.segmentation.device)
        groups = project.params.segmentation.prompt_groups
        self.prompts.setPlainText(
            "\n\n".join("\n".join(prompts) for prompts in groups.values())
        )
        self.confidence.setValue(project.params.segmentation.confidence)

    def collect(self) -> dict:
        if self.project is not None:
            from dataclasses import replace

            groups: dict[str, tuple[str, ...]] = {}
            for index, block in enumerate(self.prompts.toPlainText().split("\n\n")):
                lines = tuple(l.strip() for l in block.splitlines() if l.strip())
                if lines:
                    name = (
                        list(self.project.params.segmentation.prompt_groups)[index]
                        if index
                        < len(self.project.params.segmentation.prompt_groups)
                        else f"group_{index + 1}"
                    )
                    groups[name] = lines
            if groups:
                self.project.params = self.project.params.with_changes(
                    segmentation=replace(
                        self.project.params.segmentation,
                        prompt_groups=groups,
                        confidence=self.confidence.value(),
                        device=self.device.currentText(),
                    )
                )
            else:
                self.project.params = self.project.params.with_changes(
                    segmentation=replace(
                        self.project.params.segmentation,
                        confidence=self.confidence.value(),
                        device=self.device.currentText(),
                    )
                )
            self.project.save()

        return {
            "backend": self.backend.currentText(),
            "save_overlays": self.overlays.isChecked(),
        }


class FeaturesTab(StepTab):
    """Tab 3 - measure every colony."""

    step_name = "features"
    run_label = "Extract features"

    def build_folders(self) -> None:
        self.folders.add(
            "crops", "Colony images:",
            tooltip="The one-colony images the numbers are measured on.",
        )
        self.folders.add(
            "masks", "Masks:",
            tooltip=(
                "The outlines. A mask is matched to its crop by name, so "
                "AMF270_SDA_D06_mask.png goes with AMF270_SDA_D06.png."
            ),
        )
        self.folders.add(
            "vectors", "Save results to:",
            tooltip="Where all_colony_features.csv is written.",
        )
        self.folders.refresh_status(self._count)

    def _count(self) -> str:
        crops = self.folders["crops"].effective()
        masks = self.folders["masks"].effective()
        if crops is None or masks is None or self.project is None:
            return ""
        from ...core.pipeline import pair_crops_and_masks

        try:
            pairs = pair_crops_and_masks(self.project, crops, masks)
        except Exception:
            return ""
        n_crops = count_images(crops)
        if not pairs and n_crops:
            return (
                f"{n_crops} colony image(s) found, but none has a matching mask. "
                "Check the masks folder."
            )
        return f"{len(pairs)} colony/mask pair(s) ready"

    def folder_options(self) -> dict:
        return {
            "crops_dir": self.folders["crops"].effective(),
            "masks_dir": self.folders["masks"].effective(),
            "output_dir": self.folders["vectors"].effective(),
        }

    def build_settings(self) -> None:
        self.morphology = QCheckBox("Shape (area, circularity, solidity …)")
        self.intensity = QCheckBox("Brightness statistics")
        self.glcm = QCheckBox("GLCM texture")
        self.lbp = QCheckBox("LBP texture")
        self.colour = QCheckBox("Colour and melanization")
        for box in (
            self.morphology, self.intensity, self.glcm, self.lbp, self.colour
        ):
            box.setChecked(True)
            self.form.addRow("", box)

        self.histograms = QCheckBox(
            "Include colour histograms (many columns, excluded from PCA)"
        )
        self.histograms.setChecked(True)
        self.form.addRow("", self.histograms)

        self.perimeter = QComboBox()
        self.perimeter.addItems(["crofton", "legacy_4", "legacy_8"])
        self.perimeter.setToolTip(
            "crofton is accurate: a perfect circle scores circularity 0.99.\n"
            "legacy_8 is what the original scripts used and scores 0.62 on the "
            "same circle. Use it only to reproduce old results."
        )
        self.form.addRow("Perimeter method:", self.perimeter)

        self.ring_fraction = QDoubleSpinBox()
        self.ring_fraction.setRange(0.01, 0.50)
        self.ring_fraction.setSingleStep(0.01)
        self.ring_fraction.setValue(0.10)
        self.ring_fraction.setToolTip(
            "Ring width as a fraction of the colony's own radius."
        )
        self.form.addRow("Ring fraction:", self.ring_fraction)

    def set_project(self, project: Project) -> None:
        super().set_project(project)
        params = project.params
        self.morphology.setChecked(params.compute_morphology)
        self.intensity.setChecked(params.compute_intensity)
        self.glcm.setChecked(params.compute_glcm)
        self.lbp.setChecked(params.compute_lbp)
        self.colour.setChecked(params.compute_colour)
        self.perimeter.setCurrentText(params.morphology.perimeter_method)
        self.ring_fraction.setValue(params.ring.fraction)

    def collect(self) -> dict:
        if self.project is not None:
            from dataclasses import replace

            self.project.params = self.project.params.with_changes(
                compute_morphology=self.morphology.isChecked(),
                compute_intensity=self.intensity.isChecked(),
                compute_glcm=self.glcm.isChecked(),
                compute_lbp=self.lbp.isChecked(),
                compute_colour=self.colour.isChecked(),
                morphology=replace(
                    self.project.params.morphology,
                    perimeter_method=self.perimeter.currentText(),
                ),
                ring=replace(
                    self.project.params.ring, fraction=self.ring_fraction.value()
                ),
            )
            self.project.save()
        return {"include_histograms": self.histograms.isChecked()}


class PhenotypesTab(StepTab):
    """
    Tab 5 - GWAS phenotype scoring.

    Optional, as requested: the pipeline can stop at the feature table, or go
    on to produce GWAS-ready scores here.
    """

    step_name = "phenotypes"
    run_label = "Compute GWAS phenotypes"

    def build_folders(self) -> None:
        self.folders.add(
            "vectors", "Feature table:",
            tooltip=(
                "The all_colony_features.csv to score. Point this at a table "
                "from an earlier run to re-score it without redoing the "
                "measurements."
            ),
        )
        self.folders.add(
            "phenotypes", "Save scores to:",
            tooltip="Where the GWAS-ready tables and the diagnostics PDF go.",
        )
        self.folders.refresh_status(self._count)

    def _count(self) -> str:
        folder = self.folders["vectors"].effective()
        if folder is None:
            return ""
        candidate = Path(folder)
        if candidate.is_dir():
            candidate = candidate / "all_colony_features.csv"
        if not candidate.exists():
            return f"No feature table found at {candidate}"
        try:
            import pandas as pd

            n = len(pd.read_csv(candidate, usecols=[0]))
            return f"{n} colony row(s) in {candidate.name}"
        except Exception:
            return f"Found {candidate.name}"

    def folder_options(self) -> dict:
        return {
            "features_csv": self.folders["vectors"].effective(),
            "output_dir": self.folders["phenotypes"].effective(),
        }

    def build_settings(self) -> None:
        explanation = QLabel(
            "Combines the correlated colony features into a few pseudo-traits "
            "by PCA. Running "
            "GWAS on PC1 has more power than on any single feature when the "
            "features share a causal variant.\n\n"
            "Apply a minor-allele-frequency filter of at least 0.05 downstream: "
            "the power advantage is greatest above about 0.2."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color:#555;")
        self.form.addRow(explanation)

        self.stratify = QCheckBox("Score each growth medium separately")
        self.stratify.setChecked(True)
        self.stratify.setToolTip(
            "Colonies on MEX and SDA are not directly comparable. Pooling them "
            "would put a medium effect into PC1 and mask the genetics."
        )
        self.form.addRow("", self.stratify)

        self.exclude_failed = QCheckBox(
            "Exclude colonies marked 'fail' in the Validate tab"
        )
        self.exclude_failed.setChecked(True)
        self.form.addRow("", self.exclude_failed)

        self.n_pcs = QSpinBox()
        self.n_pcs.setRange(1, 10)
        self.n_pcs.setValue(2)
        self.form.addRow("Phenotype PCs to export:", self.n_pcs)

        self.min_nonmissing = QDoubleSpinBox()
        self.min_nonmissing.setRange(0.0, 1.0)
        self.min_nonmissing.setSingleStep(0.05)
        self.min_nonmissing.setValue(0.90)
        self.form.addRow("Minimum non-missing:", self.min_nonmissing)

        self.outlier_alpha = QDoubleSpinBox()
        self.outlier_alpha.setDecimals(4)
        self.outlier_alpha.setRange(0.0001, 0.5)
        self.outlier_alpha.setSingleStep(0.001)
        self.outlier_alpha.setValue(0.001)
        self.outlier_alpha.setToolTip(
            "Colonies are flagged, never deleted. You decide whether to drop "
            "them."
        )
        self.form.addRow("Outlier p-value:", self.outlier_alpha)

    def collect(self) -> dict:
        from ...core.phenotype import PhenotypeSettings

        return {
            "settings": PhenotypeSettings(
                min_nonmissing=self.min_nonmissing.value(),
                n_phenotype_pcs=self.n_pcs.value(),
                outlier_alpha=self.outlier_alpha.value(),
                stratify_by_media=self.stratify.isChecked(),
            ),
            "exclude_failed": self.exclude_failed.isChecked(),
        }

    def show_result(self, result) -> None:
        super().show_result(result)
        phenotypes = result.data.get("result")
        if phenotypes is None:
            return
        self.append_log("")
        for media, media_result in phenotypes.per_media.items():
            if media_result.explained_variance.size == 0:
                continue
            self.append_log(
                f"{media}: {media_result.n_samples} colonies, "
                f"PC1 explains {media_result.explained_variance[0] * 100:.1f}% "
                f"of the variance, {media_result.n_outliers} outlier(s)"
            )
            if not media_result.loadings.empty:
                top = (
                    media_result.loadings["PC1"]
                    .sort_values(key=abs, ascending=False)
                    .head(5)
                )
                self.append_log("  PC1 is driven mostly by:")
                for name, value in top.items():
                    self.append_log(f"    {name}: {value:+.3f}")
