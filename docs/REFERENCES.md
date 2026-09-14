# FungiCapture — References and Attribution

**Version:** 2.0
**Date:** 14 September 2026
**Supersedes:** version 1.0 of 21 August 2026
**Purpose:** every method, algorithm and library used anywhere in FungiCapture,
with its original source. This protects you from accidental plagiarism and makes
your methods section easy to write.

> **Changes in version 2.0.** The TAMMiCol citation in version 1.0 named the
> wrong authors and the wrong issue — it has been replaced. The claim that
> TAMMiCol "does not segment" was also wrong and has been removed. The
> dependency licence table has been corrected (`rawpy` is MIT, not LGPL;
> PySide6 is a three-way expression; `imagecodecs` was missing entirely). The
> scikit-learn PCA solver description has been corrected. Section 10, "What is
> genuinely yours", has been rewritten — two of its six claims did not survive
> a prior-art check.

**How to use this file:**

1. It lives in the repository as `docs/REFERENCES.md`.
2. When you write your paper, copy the relevant entries into your methods
   section. Every algorithm you name must be cited.
3. **Verify every DOI yourself before submission.** Any citation you have not
   personally opened is a citation you should not submit.
4. `CITATION.cff` gives GitHub a "Cite this repository" button.

---

## 1. Segmentation models

### SAM 3 — the primary segmentation backend

> Meta AI (2025). *SAM 3: Segment Anything with Concepts.*
> arXiv:2511.16719. https://arxiv.org/abs/2511.16719

Used in `core/segment.py` through `SAM3SemanticPredictor` for text-promptable
("promptable concept") segmentation of colony bodies and filamentous margins.

**⚠ Licence obligation.** The SAM License requires you to acknowledge SAM
materials in any publication:

> Colony segmentation used the Segment Anything Model 3 (SAM 3), released by
> Meta AI under the SAM License.

### SAM 1 and SAM 2 — lineage

> Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L.,
> Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., Girshick, R.
> (2023). *Segment Anything.* Proceedings of the IEEE/CVF International
> Conference on Computer Vision (ICCV), 4015–4026. arXiv:2304.02643

> Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H.,
> Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K. V.,
> Carion, N., Wu, C.-Y., Girshick, R., Dollár, P., Feichtenhofer, C. (2024).
> *SAM 2: Segment Anything in Images and Videos.* arXiv:2408.00714

### YOLOv11 — TU_MyCo-Vision (micro pipeline, out of scope for this build)

> Khanam, R., Hussain, M. (2024). *YOLOv11: An Overview of the Key
> Architectural Enhancements.* arXiv:2410.17725

> Jocher, G., Qiu, J., Chaurasia, A. (2023). *Ultralytics YOLO* (Version 8.0.0)
> [Computer software]. https://github.com/ultralytics/ultralytics

**⚠ Licence:** Ultralytics is AGPL-3.0. See `docs/LICENSING.md`.

---

## 2. Texture analysis

### GLCM — Gray-Level Co-occurrence Matrix

> Haralick, R. M., Shanmugam, K., Dinstein, I. (1973). *Textural Features for
> Image Classification.* IEEE Transactions on Systems, Man, and Cybernetics,
> SMC-3(6), 610–621. doi:10.1109/TSMC.1973.4309314

The original paper for the properties computed: contrast, homogeneity, energy,
correlation and angular second moment (ASM).

Implementation: `skimage.feature.graycomatrix` and
`skimage.feature.graycoprops`.

**Note on `dissimilarity`:** this property is **not** among Haralick's original
14. It comes from the later remote-sensing literature and is standard in
scikit-image. Do not attribute it to Haralick.

**Note on `energy` and `ASM`:** `energy = √ASM` exactly. Both are in the default
property list. Report them as one quantity, or drop one.

### LBP — Local Binary Patterns

> Ojala, T., Pietikäinen, M., Harwood, D. (1996). *A comparative study of
> texture measures with classification based on featured distributions.*
> Pattern Recognition, 29(1), 51–59. doi:10.1016/0031-3203(95)00067-4

> Ojala, T., Pietikäinen, M., Mäenpää, T. (2002). *Multiresolution Gray-Scale
> and Rotation Invariant Texture Classification with Local Binary Patterns.*
> IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(7),
> 971–987. doi:10.1109/TPAMI.2002.1017623

**Cite the 2002 paper.** It introduces the **uniform** variant
(`method='uniform'`) and the multi-resolution (P, R) parameterisation used here
with P = 16 and R = 2, giving P + 2 = 18 histogram bins.

Implementation: `skimage.feature.local_binary_pattern`.

### Shannon entropy (computed on the LBP histogram)

> Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System
> Technical Journal, 27(3), 379–423. doi:10.1002/j.1538-7305.1948.tb01338.x

Reported in **bits** (base-2 logarithm); zero-probability bins are excluded.

---

## 3. Morphology and shape

### Region properties

The individual measures (area, perimeter, eccentricity, solidity, extent, major
and minor axis of the fitted ellipse, equivalent diameter) are classical. Cite
the implementation:

> van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F.,
> Warner, J. D., Yager, N., Gouillart, E., Yu, T. (2014). *scikit-image: image
> processing in Python.* PeerJ, 2, e453. doi:10.7717/peerj.453

Implementation: `skimage.measure.regionprops`, `skimage.measure.label`.

### Circularity and roughness

`circularity = 4πA / P²` and `roughness = P² / (4πA)` are standard shape factors
with no single origin paper. The mathematical basis is the isoperimetric
inequality. In a mycology context you may cite:

> Cox, E. P. (1927). *A method of assigning numerical and percentage values to
> the degree of roundness of sand grains.* Journal of Paleontology, 1(3),
> 179–183.

**Note:** `roughness = 1 / circularity` exactly. Report one, not both.

### Perimeter estimation — Crofton

**This is now the default, and the change matters scientifically.**

> Legland, D., Kiêu, K., Devaux, M.-F. (2007). *Computation of Minkowski
> measures on 2D and 3D binary images.* Image Analysis & Stereology, 26(2),
> 83–92. doi:10.5566/ias.v26.p83-92

The Crofton formula from integral geometry estimates perimeter from counts of
intersections with families of parallel lines, rather than by summing
stair-stepped pixel boundaries.

Implementation: `skimage.measure.perimeter_crofton(..., directions=4)`.

The legacy alternatives, retained as `perimeter_method="legacy_4"` and
`"legacy_8"`, are `skimage.measure.perimeter(..., neighborhood=4 | 8)`.

**Report this in your methods, and consider reporting it as a result.** The
original pipeline used `neighborhood=8`, which reports a perfect disc as
circularity 0.62 instead of 1.0. Critically, the bias is **not a constant
rescaling** — it depends on boundary orientation and thinness. Measured bias
factors: 1.05 for a square, 1.26 for a disc, 1.47 for a diamond, and 0.95 for a
disc with thin spikes. **Compact and filamentous colonies are therefore biased
in opposite directions** — which is exactly the comparison this tool exists to
make.

### Morphological dilation (used to build the boundary ring)

> Serra, J. (1982). *Image Analysis and Mathematical Morphology.* Academic
> Press, London. ISBN 0-12-637240-3

Implementation: `skimage.morphology.dilation` with `skimage.morphology.disk`.

**⚠ Prior art for the ring itself — corrected in version 2.0.** Version 1.0 of
this document stated that the adaptive boundary ring was an original
contribution requiring no citation. **That was wrong.** ColTapp (Bär *et al.*,
2020) already computes a halo band "based on a fixed number of pixels **or as a
multiple of colony radius**". Cite ColTapp when describing the ring, and
position FungiCapture on what it computes *inside* the band (the full feature
vector) rather than on the band itself. See §10.

### Distance transform (radial colour zonation)

> Maurer, C. R., Qi, R., Raghavan, V. (2003). *A linear time algorithm for
> computing exact Euclidean distance transforms of binary images in arbitrary
> dimensions.* IEEE Transactions on Pattern Analysis and Machine Intelligence,
> 25(2), 265–270. doi:10.1109/TPAMI.2003.1177156

Implementation: `scipy.ndimage.distance_transform_edt`.

### Otsu thresholding (classical fallback backend, and colony detection)

> Otsu, N. (1979). *A threshold selection method from gray-level histograms.*
> IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66.
> doi:10.1109/TSMC.1979.4310076

Implementation: `skimage.filters.threshold_otsu`. Used in `core/detect.py` and
in the classical segmentation backend of `core/segment.py`.

### Hough circle transform (Petri dish rim detection)

> Duda, R. O., Hart, P. E. (1972). *Use of the Hough transformation to detect
> lines and curves in pictures.* Communications of the ACM, 15(1), 11–15.
> doi:10.1145/361237.361242

> Hough, P. V. C. (1962). *Method and means for recognizing complex patterns.*
> U.S. Patent 3,069,654.

Implementation: `skimage.transform.hough_circle`, `hough_circle_peaks`, with
`skimage.filters.sobel` for edge detection.

### Active contours without edges (Chan–Vese) — legacy only

> Chan, T. F., Vese, L. A. (2001). *Active contours without edges.* IEEE
> Transactions on Image Processing, 10(2), 266–277. doi:10.1109/83.902291

> Márquez-Neila, P., Baumela, L., Álvarez, L. (2014). *A morphological approach
> to curvature-based evolution of curves and surfaces.* IEEE Transactions on
> Pattern Analysis and Machine Intelligence, 36(1), 2–17.
> doi:10.1109/TPAMI.2013.106

Cite **both** if this refinement step is used — the second paper is the
algorithm actually run by `skimage.segmentation.morphological_chan_vese`.

---

## 4. Colour science

### CIE L\*a\*b\* colour space

> Commission Internationale de l'Éclairage (1978). *Recommendations on Uniform
> Color Spaces, Color-Difference Equations, Psychometric Color Terms.*
> Supplement No. 2 to CIE Publication No. 15 (E-1.3.1) 1971/(TC-1.3), Bureau
> Central de la CIE, Paris.

> Commission Internationale de l'Éclairage (2004). *CIE 15:2004 Colorimetry,
> 3rd edition.* CIE Central Bureau, Vienna.

**State your reference white in the methods.** FungiCapture uses **D65 with the
2° standard observer**, recorded in `params.lab_illuminant` /
`params.lab_observer` and written into every output file. L\* values computed
against a different white are not comparable.

Implementation: `skimage.color.rgb2lab`.

### CIE76 colour difference (used to score the colour-card correction)

> Commission Internationale de l'Éclairage (1976). *Official recommendations on
> uniform color spaces, color-difference equations and metric color terms.*
> CIE Publication No. 15, Supplement No. 2.

`ΔE*₇₆ = √[(ΔL*)² + (Δa*)² + (Δb*)²]`. Used in `core/calibration.py` as the
residual after fitting, with a rejection threshold — a correction that fits
poorly is refused rather than applied silently.

### HSV colour space

> Smith, A. R. (1978). *Color gamut transform pairs.* ACM SIGGRAPH Computer
> Graphics, 12(3), 12–19. doi:10.1145/965139.807361

Implementation: `skimage.color.rgb2hsv`.

### Grayscale conversion

The luminance weights used by `skimage.color.rgb2gray`
(Y = 0.2125 R + 0.7154 G + 0.0721 B) come from ITU-R BT.709:

> International Telecommunication Union (2015). *Recommendation ITU-R BT.709-6:
> Parameter values for the HDTV standards for production and international
> programme exchange.*

**Worth stating in your paper:** green contributes seven times more than blue.
For a melanin-pigmented fungus that is not a neutral choice, and it is one
reason the grayscale-only pipeline could not measure pigmentation. Colour
features are computed from the colour image instead.

### k-means clustering (dominant colour extraction)

> Lloyd, S. P. (1982). *Least squares quantization in PCM.* IEEE Transactions
> on Information Theory, 28(2), 129–137. doi:10.1109/TIT.1982.1056489
> (Originally a 1957 Bell Labs technical report.)

> MacQueen, J. (1967). *Some methods for classification and analysis of
> multivariate observations.* Proceedings of the Fifth Berkeley Symposium on
> Mathematical Statistics and Probability, 1, 281–297.

> Arthur, D., Vassilvitskii, S. (2007). *k-means++: The advantages of careful
> seeding.* Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete
> Algorithms, 1027–1035.

scikit-learn's `KMeans` uses k-means++ initialisation by default, so cite
Arthur & Vassilvitskii as well as Lloyd. Clustering runs in Lab space, where
Euclidean distance approximates perceived colour difference.

### Linear colour correction (colour reference card)

The 3×4 augmented-matrix model fitted by ordinary least squares
(`numpy.linalg.lstsq`) is the standard linear colour-correction model. It is
textbook material rather than a single-paper method; cite a colour-science text
if a reviewer asks, for example:

> Westland, S., Ripamonti, C., Cheung, V. (2012). *Computational Colour Science
> Using MATLAB*, 2nd edition. Wiley. ISBN 978-0-470-66569-5

### Pigmentation and melanization — background reading

The three indices (`MI_lightness`, `MI_browning`, `MI_darkfraction`) are
**FungiCapture definitions, not community standards**. State each formula in
full, and state the `L* < 35` threshold.

> ⚠ **Terminology.** Until these indices are validated against an independent
> assay, describe them as **image-derived pigmentation indices**, not as
> melanin measurements. Colony darkness in a photograph can also arise from
> other pigments, colony thickness, sporulation, surface wetness, condensation
> and agar colour.

Background:

> Eisenman, H. C., Casadevall, A. (2012). *Synthesis and assembly of fungal
> melanin.* Applied Microbiology and Biotechnology, 93(3), 931–940.
> doi:10.1007/s00253-011-3777-2

> Lendenmann, M. H., Croll, D., McDonald, B. A. (2015). *Quantitative trait
> locus mapping of melanization in the plant pathogenic fungus Zymoseptoria
> tritici.* Fungal Genetics and Biology, 80, 53–67.
> doi:10.1016/j.fgb.2015.05.001

The Lendenmann paper is the closest published example of image-based
melanization used as a mapped quantitative trait. **Cite it; it is directly
relevant to your GWAS.** Use the journal year **2015**.

---

## 5. Statistics

### Skewness and kurtosis

> Joanes, D. N., Gill, C. A. (1998). *Comparing measures of sample skewness and
> kurtosis.* Journal of the Royal Statistical Society: Series D (The
> Statistician), 47(1), 183–189. doi:10.1111/1467-9884.00122

`bias=False` selects the **G1 / G2** bias-corrected estimators described in this
paper. State that in the methods, and state that kurtosis is reported as
**excess kurtosis** (Fisher definition, normal distribution = 0).

Implementation: `scipy.stats.skew`, `scipy.stats.kurtosis`.

### Principal Component Analysis

> Pearson, K. (1901). *On lines and planes of closest fit to systems of points
> in space.* Philosophical Magazine, Series 6, 2(11), 559–572.
> doi:10.1080/14786440109462720

> Hotelling, H. (1933). *Analysis of a complex of statistical variables into
> principal components.* Journal of Educational Psychology, 24(6), 417–441.
> doi:10.1037/h0071325

> Jolliffe, I. T., Cadima, J. (2016). *Principal component analysis: a review
> and recent developments.* Philosophical Transactions of the Royal Society A,
> 374(2065), 20150202. doi:10.1098/rsta.2015.0202
> — a good modern citation for a methods section.

**Corrected in version 2.0.** Version 1.0 stated that scikit-learn's `PCA`
"uses truncated SVD" and cited Halko *et al.* (2011). That is only sometimes
true. The solver is chosen by the `svd_solver` parameter, which defaults to
`"auto"`; `"auto"` selects the full LAPACK SVD for the matrix sizes typical
here, and only switches to the randomized solver for large matrices.
FungiCapture does not set `svd_solver`.

**Therefore: cite Halko *et al.* only if you have confirmed that the randomized
solver actually ran.** Otherwise state simply that PCA was computed by singular
value decomposition of the standardised feature matrix.

If you do need it:

> Halko, N., Martinsson, P. G., Tropp, J. A. (2011). *Finding structure with
> randomness: probabilistic algorithms for constructing approximate matrix
> decompositions.* SIAM Review, 53(2), 217–288. doi:10.1137/090771806

### PCA-based multiple-trait GWAS — the rationale for the phenotype scores

> Zhang, F., Gao, Y., *et al.* (2018). *PCA-Based Multiple-Trait GWAS Analysis:
> A Powerful Model for Exploring Pleiotropy.* Animals, 8(12), 239.
> doi:10.3390/ani8120239. PMID: 30562943

**⚠ Verify the full author list from the PubMed record before submission.** The
title, journal, volume, article number, DOI and PMID are confirmed; the complete
author list is not.

**What this paper does and does not justify.** It supports combining correlated
traits into pseudo-traits to raise power for detecting pleiotropic variants. It
does **not** establish that PC1 is the correct phenotype for any particular
feature set. FungiCapture accordingly exports PC1 **and** PC2, fits separately
per growth medium, reports loadings and explained variance, and raises a warning
when PC1 explains less than 15 % of variance. Present PC1 as one exploratory
composite phenotype, not as the phenotype.

Two caveats worth reporting from that paper: the power advantage is greatest
when the causal variant has minor allele frequency above about 0.2 (apply a MAF
filter of at least 0.05 downstream), and for two linked causal variants the
advantage holds while r²_LD is above roughly 0.7.

### Mahalanobis distance (multivariate outlier flagging)

> Mahalanobis, P. C. (1936). *On the generalised distance in statistics.*
> Proceedings of the National Institute of Sciences of India, 2(1), 49–55.

The chi-squared cut-off:

> Rousseeuw, P. J., Van Zomeren, B. C. (1990). *Unmasking multivariate outliers
> and leverage points.* Journal of the American Statistical Association,
> 85(411), 633–639. doi:10.1080/01621459.1990.10474920

**⚠ State the limitation honestly.** The chi-squared threshold assumes a stable
covariance estimate and approximately elliptical multivariate data. With many
components and few colonies that assumption is weak. FungiCapture **flags**
outliers (`is_outlier`, `mahalanobis_d2`, `outlier_p` columns) and never deletes
them; the decision to exclude a strain remains the analyst's. Describe the flags
as a screening aid, not as a test.

### Moore–Penrose pseudoinverse

> Penrose, R. (1955). *A generalized inverse for matrices.* Mathematical
> Proceedings of the Cambridge Philosophical Society, 51(3), 406–413.
> doi:10.1017/S0305004100030401

Implementation: `numpy.linalg.pinv`. Used because the covariance matrix of PC
scores is singular when components outnumber colonies. Note in the methods that
the pseudoinverse permits computation but does not repair the underlying
statistical assumption.

---

## 6. Software libraries

Cite these in a "Software" subsection of your methods, **with the version
numbers recorded in your run's params file**.

| Library | Citation | SPDX licence |
|---|---|---|
| **NumPy** | Harris, C. R., et al. (2020). *Array programming with NumPy.* Nature, 585, 357–362. doi:10.1038/s41586-020-2649-2 | `BSD-3-Clause` |
| **SciPy** | Virtanen, P., et al. (2020). *SciPy 1.0.* Nature Methods, 17, 261–272. doi:10.1038/s41592-019-0686-2 | `BSD-3-Clause` |
| **pandas** | McKinney, W. (2010). *Data structures for statistical computing in Python.* Proc. 9th Python in Science Conf., 56–61. doi:10.25080/Majora-92bf1922-00a | `BSD-3-Clause` |
| **scikit-image** | van der Walt, S., et al. (2014). PeerJ, 2, e453. doi:10.7717/peerj.453 | `BSD-3-Clause` |
| **scikit-learn** | Pedregosa, F., et al. (2011). JMLR, 12, 2825–2830 | `BSD-3-Clause` |
| **matplotlib** | Hunter, J. D. (2007). CiSE, 9(3), 90–95. doi:10.1109/MCSE.2007.55 | Matplotlib License (PSF-derived) |
| **OpenCV** (`opencv-python-headless`) | Bradski, G. (2000). *The OpenCV Library.* Dr. Dobb's Journal, 25(11), 120–125 | Wrapper `MIT`; OpenCV `Apache-2.0`; **bundles FFmpeg `LGPL-2.1`** |
| **Pillow** | Clark, A. and contributors. *Pillow.* https://python-pillow.org/ | `MIT-CMU` |
| **imagecodecs** | Gohlke, C. *imagecodecs.* https://github.com/cgohlke/imagecodecs | `BSD-3-Clause` + bundled codec licences |
| **rawpy** | Riechert, M. *rawpy.* https://github.com/letmaik/rawpy | **`MIT`** |
| └ **LibRaw** | https://www.libraw.org/ | `LGPL-2.1-only OR CDDL-1.0` |
| **platformdirs**, **tomli**, **tomli-w** | — | `MIT` |
| **PyTorch** | Paszke, A., et al. (2019). NeurIPS, 32, 8024–8035 | `BSD-3-Clause` |
| **Ultralytics** | Jocher, G., Qiu, J., Chaurasia, A. (2023). https://github.com/ultralytics/ultralytics | **`AGPL-3.0-only`** ⚠ |
| **PySide6 / Qt for Python** | The Qt Company. https://doc.qt.io/qtforpython/ | `LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only` |
| **PyInstaller** | https://pyinstaller.org/ | `GPL-2.0-or-later WITH Bootloader-exception` |

**Corrections made in version 2.0:** `rawpy` was listed as LGPL — it is MIT; the
LGPL/CDDL applies to LibRaw, the native library it wraps. PySide6 was listed as
LGPL-3.0 only. `imagecodecs`, `Pillow`, `platformdirs`, `tomli` and `tomli-w`
were missing entirely.

Full detail, including the obligations each licence creates, is in
`docs/DEPENDENCIES_AND_LICENSING.md`.

---

## 7. Prior tools to cite in the introduction

Citing competitors is not a weakness. It shows you know the field, and it is
what makes a novelty claim credible. Cite all of these.

### Colony size and plate gridding

> Lawless, C., Wilkinson, D. J., Young, A., Addinall, S. G., Lydall, D. A.
> (2010). *Colonyzer: automated quantification of micro-organism growth
> characteristics on solid agar.* BMC Bioinformatics, 11, 287.
> doi:10.1186/1471-2105-11-287

> Dittmar, J. C., Reid, R. J., Rothstein, R. (2010). *ScreenMill: a freely
> available software suite for growth measurement, analysis and visualization
> of high-throughput screen data.* BMC Bioinformatics, 11, 353.
> doi:10.1186/1471-2105-11-353

> Wagih, O., Parts, L. (2014). *gitter: a robust and accurate method for
> quantification of colony sizes from plate images.* G3: Genes, Genomes,
> Genetics, 4(3), 547–552. doi:10.1534/g3.113.009431

> Kritikos, G., Banzhaf, M., Herrera-Dominguez, L., Koumoutsi, A., Wartel, M.,
> Zietek, M., Typas, A. (2017). *A tool named Iris for versatile
> high-throughput phenotyping in microorganisms.* Nature Microbiology, 2,
> 17014. doi:10.1038/nmicrobiol.2017.14

> Kamrad, S., Rodríguez-López, M., Cotobal, C., Correia-Melo, C., Ralser, M.,
> Bähler, J. (2020). *Pyphe, a python toolbox for assessing microbial growth
> and cell viability in high-throughput colony screens.* eLife, 9, e55160.
> doi:10.7554/eLife.55160

> MCount: an automated colony counting tool for high-throughput microbiology.
> PLOS ONE (2025). doi:10.1371/journal.pone.0311242
> **⚠ Use 2025 for the journal article.** Version 1.0 of this document dated it
> 2024; that refers to the preprint stage. Verify the author list from the
> journal page.

### Colony morphology, texture and colour — the direct competitors

> **Tronnolone, H., Gardner, J. M., Sundstrom, J. F., Jiranek, V.,
> Oliver, S. G., Binder, B. J. (2018). *TAMMiCol: Tool for analysis of the
> morphology of microbial colonies.* PLOS Computational Biology, 14(12),
> e1006629. doi:10.1371/journal.pcbi.1006629**
>
> ⚠ **This citation was wrong in version 1.0 of this document.** It named
> "Davis, H. J., Miller, R. C., et al." and gave issue 14(3). Neither is
> correct. Author list and issue verified 14 September 2026.
>
> ⚠ **The description was also wrong.** Version 1.0 said TAMMiCol "takes a
> binary image as input — does not segment". TAMMiCol converts colony
> photographs to binary itself, selects thresholds, supports batch processing
> and performs quantitative morphology analysis, including on filamentous yeast
> colonies. Describe it as an automated classical-segmentation and morphology
> tool.

> Bär, J., Boumasmoud, M., Kouyos, R. D., Zinkernagel, A. S., Vulin, C. (2020).
> *Efficient microbial colony growth dynamics quantification with ColTapp, an
> automated image analysis application.* Scientific Reports, 10, 16084.
> doi:10.1038/s41598-020-72979-4
>
> **The most important tool to cite.** ColTapp measures colony colour (RGB and
> grayscale), image entropy and intensity SD as texture proxies, contour
> perimeter and its SD, spatial density metrics, and **halo bands defined either
> as a fixed pixel width or as a multiple of colony radius**, with explicit
> handling of perimeter sections affected by neighbouring colonies. Do not
> describe it as measuring only radius over time.

> Vidal-Diez de Ulzurrun, G., Huang, T.-Y., Chang, C.-W., Lin, H.-C.,
> Hsueh, Y.-P. (2019). *Fungal feature tracker (FFT): a tool for quantitatively
> characterizing the morphology and growth of filamentous fungi.* PLOS
> Computational Biology, 15(10), e1007428. doi:10.1371/journal.pcbi.1007428

> Goldschmidt, A., Kunert-Graf, J., Scott, A. C., *et al.* (2022). *Quantifying
> yeast colony morphologies with feature engineering from time-lapse
> photography.* Scientific Data, 9, 216. doi:10.1038/s41597-022-01340-3

> Mota, A., Schiele, A., Santos de Sousa, I., Cortesão, M. (2026).
> *Open-source digital tools for filamentous fungi analysis.* Journal of
> Microbiological Methods. (Preprint: SSRN 5912070, 12 December 2025.)
>
> **New in version 2.0 — missed by the August search.** Presents **Orbis v2**
> (colony segmentation and area by thresholding and contour detection) and
> **SporeQuant** (spore counting). Both free, open-source and web-based,
> designed for use without specialist hardware or expertise. Fungal,
> colony-scale, recent, and explicitly aimed at accessibility.

### Foundation models on microbial colonies — new in version 2.0

> **Colony Grounded SAM2: zero-shot detection and segmentation of bacterial
> colonies using foundation models.** Proceedings of SPIE Medical Imaging 2026,
> Vol. 13928, 139281A. doi:10.1117/12.3085170. (Preprint arXiv:2603.13393.)
>
> **Directly relevant prior art.** Combines Grounding DINO (a *text-prompted*
> detector) with SAM2, both fine-tuned to the microbiological domain, for
> zero-shot detection and instance segmentation of bacterial colonies on agar.
> Reports mAP 93.1 % and Dice@detection 0.85 on out-of-distribution data, and
> releases pipeline and weights openly. **Its existence rules out any claim that
> foundation models have not reached agar-plate colonies.**

> *Enhancing Colony Detection of Microorganisms in Agar Dishes Using SAM-Based
> Synthetic Data Augmentation in Low-Data Scenarios.* Applied Sciences (2025),
> 15(3), 1260. doi:10.3390/app15031260

### Foundation models on cells (context, not direct competitors)

> Archit, A., *et al.* (2025). *Segment Anything for Microscopy.* Nature
> Methods. doi:10.1038/s41592-024-02580-4
> **⚠ Use 2025**, the journal year. Version 1.0 dated `micro-sam` to 2023, which
> is the preprint stage.

> *CellSAM: a foundation model for cell segmentation.* Nature Methods (2025).
> doi:10.1038/s41592-025-02879-w

> *SAMCell: generalized label-free biological cell segmentation with segment
> anything.* PLOS ONE (2025). doi:10.1371/journal.pone.0319532

---

## 8. Licence acknowledgements — obligations, not courtesy

These are requirements. `docs/DEPENDENCIES_AND_LICENSING.md` gives the detail.

1. **SAM 3.** Include the SAM License text in the repository. Acknowledge SAM
   materials in any publication.
2. **Ultralytics (AGPL-3.0).** FungiCapture is released under AGPL-3.0 while it
   depends on this package. The full AGPL-3.0 text is in `LICENSE`.
3. **PySide6 (LGPL-3.0-only, as selected).** State that PySide6 is dynamically
   linked and that users may replace the Qt libraries. One-folder PyInstaller
   builds support this.
4. **LibRaw via rawpy.** Include the LibRaw licence notice. Note that `rawpy`
   itself is MIT.
5. **FFmpeg inside `opencv-python-headless` (LGPL-2.1).** Include the notice.
6. **imagecodecs.** Ship `imagecodecs/licenses/` and the Independent JPEG Group
   notice.
7. **All BSD / Apache / MIT dependencies.** Include copyright notices.

Generate the machine-readable list on every release:

```bash
pip install ".[gui,model,dev]"
pip install pip-licenses
pip-licenses --format=markdown --with-urls --with-license-file --with-authors \
             --output-file THIRD_PARTY_LICENSES.md
```

⚠ `pip-licenses` reads Python metadata only. It cannot see bundled native
libraries, model weights, fonts, codecs or hand-copied code. Those must be
audited manually, once, and the result written down.

---

## 9. What is genuinely original — rewritten in version 2.0

Version 1.0 listed six items as "genuinely yours" and said nothing in the
literature needed to be cited for them. **Two of the six did not survive a
prior-art check, and the blanket no-citation statement was wrong scholarly
practice.** An implementation can be original while still requiring citations to
related work.

The honest list:

| Claim | Status | What to cite alongside it |
|---|---|---|
| **The adaptive boundary ring** | ❌ **Not original.** ColTapp already defines halo bands as a multiple of colony radius | Bär *et al.* 2020. Position on *what is computed inside* the band — the complete intensity, GLCM, LBP and nine-statistic colour vector — rather than on the band itself |
| **The dual prompt-group union strategy** (`core_colony` ∪ `fine_filaments`) | ⚠ **Original as applied, but adjacent work exists.** Text-prompted foundation-model segmentation of agar colonies is published | Colony Grounded SAM2 (SPIE 2026). Claim the recall-favouring union of a core prompt group and a filament prompt group, for filamentous fungal margins specifically |
| **The three melanization indices** | ✅ **Original definitions**, but definitions rather than discoveries, and unvalidated | Lendenmann *et al.* 2015 for image-based melanization as a mapped trait. Give all three formulas and the `L* < 35` threshold in full. Call them image-derived pigmentation indices until validated |
| **Radial colour zonation via distance transform, reduced to slope and reversal count** | ✅ **Original.** No equivalent found in ColTapp, TAMMiCol, FFT or Goldschmidt *et al.* | Maurer *et al.* 2003 for the distance transform itself. **This is the strongest single novel feature** |
| **Angular sectoring as a numeric trait** | ✅ **Original as a packaged measurement.** Sectoring is a well-known phenomenon normally scored by eye | Nothing directly; cite sectoring literature for the biology |
| **Built-in colour-card calibration with ΔE residual rejection** | ✅ **Uncommon and useful** in a colony-phenotyping tool | CIE76 for the metric; standard colour-science texts for the linear model |
| **The perimeter-estimator bias characterisation** | ✅ **A real, small, reportable finding** — the bias is orientation-dependent and reverses sign between compact and spiked shapes | Legland *et al.* 2007; scikit-image |
| **The FungiCapture application itself** — gridding, foundation-model segmentation, paired core/ring features, colour calibration, QC, media-stratified phenotype export in one cross-platform tool | ✅ **The real contribution.** No single published tool spans this chain | All of §7 |
| **`sam3.1_multiplex.pt`** | ⚠ Derivative work of SAM 3; redistribution unresolved | See `docs/LICENSING.md` §2 |

**The honest summary:** FungiCapture contains no new algorithm. Every
mathematical operation is standard and from a published library. The
contribution is the integration, the domain focus on pigmented filamentous
fungi, and four measurement axes that existing tools do not package. That is a
software paper, and a legitimate one. Do not present it as a methods paper.

---

## 10. Before you submit — checklist

- [ ] Every DOI in this file opened and confirmed
- [ ] Zhang *et al.* 2018 author list copied from PubMed
- [x] TAMMiCol author list confirmed — corrected 14 September 2026
- [ ] MCount author list and 2025 journal details confirmed
- [ ] Colony Grounded SAM2 full author list and SPIE page numbers confirmed
- [ ] Mota *et al.* final journal volume, issue and pages confirmed
- [ ] Software versions recorded for every library (from the run's params file)
- [ ] `CITATION.cff` added to the repository
- [ ] `THIRD_PARTY_LICENSES.md` generated with all extras installed
- [ ] SBOM generated
- [ ] SAM acknowledgement sentence in the methods
- [ ] Perimeter estimator named in the methods
- [ ] Lab reference white (D65, 2°) named in the methods
- [ ] Melanization indices described as image-derived pigmentation indices,
      with all three formulas and the threshold
- [ ] Literature search repeated on Scopus, Web of Science, PubMed, bioRxiv and
      arXiv one month before submission
- [ ] Search strings and screening decisions archived as supplementary material
