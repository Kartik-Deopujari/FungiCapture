"""
FungiCapture — image-based phenotyping of fungal colonies.

From an agar plate photograph to GWAS-ready phenotype scores: shape,
brightness, texture and colour measured on the colony interior and an adaptive
boundary ring.

Layout
------
``fungicapture.core``
    The science. No graphical dependency, so it installs and runs headless on
    a compute cluster.
``fungicapture.gui``
    The PySide6 application. A thin layer that calls into ``core``.
``fungicapture.cli``
    The command-line runner, for batches and scripted re-analysis.

This file also matters practically: without it Python treats ``fungicapture``
as an implicit namespace package, which imports but reports ``__file__`` as
None and can be packaged incorrectly. Being an explicit package avoids both.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
