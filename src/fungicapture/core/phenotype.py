"""
GWAS phenotype scoring: turning ~400 feature columns into a few scores.

The idea, from Zhang, Gao et al. (2018) *Animals* 8:239
-------------------------------------------------------
Running a separate GWAS on each of 400 features is a multiple-testing disaster,
and each single feature is a noisy view of the underlying biology. Instead,
correlated features are combined into a small number of **pseudo-traits** by
PCA, and the GWAS runs on those.

The statistical argument: the SNP effect on the pseudo-trait is a weighted sum
of its effects on every contributing feature. When those features share a
causal variant - which they do, since growth rate moves area, texture and
pigmentation together - the combined effect is larger than any single one, so
the test has more power.

Two caveats from that paper, worth knowing before you interpret results:

* The power advantage is greatest when the causal variant has minor allele
  frequency above about 0.2. Apply a MAF filter of at least 0.05 downstream.
* For two different causal variants in linkage, the advantage holds while
  r_LD is above roughly 0.7.

What this module does
---------------------
1. Drop features that are mostly missing, then impute the rest.
2. Drop near-constant features.
3. Standardise per medium, so MEX/PDA/SDA batch effects do not dominate.
4. Fit PCA twice: once on everything to find outliers, then again on the clean
   subset so extreme colonies do not bend the axes.
5. Flag multivariate outliers by Mahalanobis distance.
6. Write a GWAS-ready table.

Outliers are **flagged, not deleted**. Whether to exclude a strain is the
geneticist's decision, and silently dropping data is how results become
irreproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PhenotypeSettings:
    """Choices that change the phenotype numbers."""

    min_nonmissing: float = 0.90
    """Drop a feature present in fewer than this fraction of colonies."""

    min_cv: float = 0.01
    """Drop a feature whose coefficient of variation is below this. A
    near-constant column carries no information but can still create a
    spurious PCA direction out of numerical noise."""

    n_components: int = 20
    """Components to keep in the clean model. Only the first few are used as
    phenotypes; the rest inform the scree plot."""

    n_phenotype_pcs: int = 2
    """How many PC scores to export as phenotypes."""

    outlier_alpha: float = 0.001
    """Chi-squared p-value below which a colony is flagged. Deliberately
    strict, so only genuinely extreme colonies are marked."""

    outlier_n_pcs: int = 10
    """PCs used for the Mahalanobis distance."""

    stratify_by_media: bool = True
    """Standardise and fit separately for each growth medium. Colonies on MEX
    and SDA are not directly comparable, so pooling them would put a medium
    effect into PC1 and mask the genetics."""

    impute_strategy: str = "median"
    """Median, not mean: morphological features are skewed, and a few huge
    colonies would drag a mean imputation upward."""


@dataclass
class MediaResult:
    """Phenotype scores for one growth medium."""

    media: str
    table: pd.DataFrame
    features_used: list[str]
    explained_variance: np.ndarray
    loadings: pd.DataFrame
    n_outliers: int
    warnings: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.table)


@dataclass
class PhenotypeResult:
    """Everything the phenotype step produced."""

    per_media: dict[str, MediaResult]
    combined: pd.DataFrame
    settings: PhenotypeSettings
    dropped_features: dict[str, str]
    """Feature name to the reason it was dropped."""
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Feature QC
# --------------------------------------------------------------------------


def select_features(
    frame: pd.DataFrame,
    candidate_columns: list[str],
    settings: PhenotypeSettings,
) -> tuple[list[str], dict[str, str]]:
    """
    Keep the feature columns worth analysing, and say why the others went.

    Returning the reasons matters: a user who sees "277 features became 141"
    with no explanation cannot tell a healthy filter from a broken pipeline.
    """
    kept: list[str] = []
    dropped: dict[str, str] = {}

    for column in candidate_columns:
        if column not in frame.columns:
            dropped[column] = "not present in the data"
            continue
        series = pd.to_numeric(frame[column], errors="coerce")

        present = float(series.notna().mean())
        if present < settings.min_nonmissing:
            dropped[column] = f"only {present:.0%} of colonies have a value"
            continue

        values = series.dropna()
        if values.empty:
            dropped[column] = "no values at all"
            continue

        mean = float(values.mean())
        deviation = float(values.std(ddof=0))
        cv = deviation / abs(mean) if abs(mean) > 1e-8 else deviation
        if cv < settings.min_cv:
            dropped[column] = f"almost constant (CV {cv:.4f})"
            continue

        kept.append(column)
    return kept, dropped


# --------------------------------------------------------------------------
# Scoring one medium
# --------------------------------------------------------------------------


def score_media(
    frame: pd.DataFrame,
    features: list[str],
    media: str,
    settings: PhenotypeSettings,
) -> MediaResult:
    """Run the full QC, standardise, PCA and outlier chain for one medium."""
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    warnings: list[str] = []
    matrix = frame[features].apply(pd.to_numeric, errors="coerce")
    n_samples = len(matrix)

    if n_samples < 3:
        warnings.append(
            f"Only {n_samples} colonies on {media} - too few for PCA. "
            "Scores are not produced for this medium."
        )
        return MediaResult(
            media, frame.copy(), features, np.array([]), pd.DataFrame(), 0, warnings
        )

    imputed = SimpleImputer(strategy=settings.impute_strategy).fit_transform(matrix)

    # Pass A: fit on everything, only to find outliers.
    scaler_all = StandardScaler().fit(imputed)
    scaled_all = scaler_all.transform(imputed)
    n_full = min(scaled_all.shape[0], scaled_all.shape[1])
    scores_all = PCA(n_components=n_full, random_state=0).fit_transform(scaled_all)

    is_outlier, distances, p_values = flag_outliers(
        scores_all, settings.outlier_n_pcs, settings.outlier_alpha
    )
    n_outliers = int(is_outlier.sum())

    # Pass B: refit on the clean subset so extreme colonies do not bend the
    # axes, then project everything - including the outliers - onto them.
    clean = ~is_outlier
    if clean.sum() < 3:
        warnings.append(
            "Almost every colony was flagged as an outlier, which usually means "
            "the data are very heterogeneous. Using all colonies for the model."
        )
        clean = np.ones(n_samples, dtype=bool)

    scaler = StandardScaler().fit(imputed[clean])
    scaled = scaler.transform(imputed)
    n_components = min(settings.n_components, int(clean.sum()), len(features))
    model = PCA(n_components=n_components, random_state=0).fit(scaled[clean])
    scores = model.transform(scaled)

    keep_id = [
        c
        for c in ("sample_id", "accession", "media", "plate_index", "filename")
        if c in frame.columns
    ]
    table = frame[keep_id].copy() if keep_id else pd.DataFrame(index=frame.index)

    n_pcs = min(settings.n_phenotype_pcs, scores.shape[1])
    for i in range(n_pcs):
        table[f"PC{i + 1}_score"] = scores[:, i]
    for i in range(n_pcs):
        table[f"PC{i + 1}_variance_pct"] = float(
            model.explained_variance_ratio_[i] * 100
        )

    table["is_outlier"] = is_outlier.astype(int)
    table["mahalanobis_d2"] = distances
    table["outlier_p"] = p_values
    table["n_features_used"] = len(features)

    loadings = pd.DataFrame(
        model.components_.T,
        index=features,
        columns=[f"PC{i + 1}" for i in range(model.n_components_)],
    )

    if model.explained_variance_ratio_[0] < 0.15:
        warnings.append(
            f"PC1 explains only {model.explained_variance_ratio_[0]:.1%} of the "
            "variance on this medium. The features may not share a dominant "
            "axis, so PC1 is a weak phenotype here."
        )
    if n_outliers:
        warnings.append(
            f"{n_outliers} of {n_samples} colonies flagged as outliers. They are "
            "kept in the table with is_outlier=1 - decide yourself whether to "
            "exclude them."
        )

    return MediaResult(
        media, table, features, model.explained_variance_ratio_, loadings,
        n_outliers, warnings,
    )


def flag_outliers(
    scores: np.ndarray, n_pcs: int, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find colonies that are extreme across several dimensions at once.

    Mahalanobis distance measures how far a point is from the centre *in units
    of the data's own spread*, accounting for correlations between axes. A
    colony can be unremarkable on every single feature yet sit far from the
    cloud in combination - that is what this catches and a per-feature
    threshold misses.

    Under a multivariate normal the squared distance follows a chi-squared
    distribution with ``k`` degrees of freedom, which turns a distance into a
    p-value.

    Returns ``(is_outlier, squared_distances, p_values)``.
    """
    from scipy.stats import chi2

    n_samples = scores.shape[0]
    k = int(min(n_pcs, scores.shape[1], max(1, n_samples - 1)))
    if k < 1 or n_samples < 3:
        return (
            np.zeros(n_samples, bool),
            np.full(n_samples, np.nan),
            np.full(n_samples, np.nan),
        )

    subset = scores[:, :k]
    centre = subset.mean(axis=0)
    difference = subset - centre

    covariance = np.cov(subset, rowvar=False)
    covariance = np.atleast_2d(covariance)
    # Pseudo-inverse, not inverse: with more features than colonies the
    # covariance matrix is singular and a plain inverse would raise.
    inverse = np.linalg.pinv(covariance)

    squared = np.einsum("ij,jk,ik->i", difference, inverse, difference)
    squared = np.clip(squared, 0, None)
    p_values = 1.0 - chi2.cdf(squared, df=k)
    return p_values < alpha, squared, p_values


# --------------------------------------------------------------------------
# Front door
# --------------------------------------------------------------------------


def compute_phenotypes(
    frame: pd.DataFrame,
    feature_columns: list[str],
    settings: PhenotypeSettings | None = None,
) -> PhenotypeResult:
    """
    Score a whole feature table.

    ``frame`` is one row per colony, as written by the feature step.
    ``feature_columns`` is normally ``analysis_columns(...)``, which already
    excludes identifiers, LBP bins and colour histograms.
    """
    settings = settings or PhenotypeSettings()
    warnings: list[str] = []

    features, dropped = select_features(frame, feature_columns, settings)
    if not features:
        raise ValueError(
            "No usable features left after quality control. "
            f"{len(dropped)} were dropped - check the feature table is not empty."
        )
    if dropped:
        warnings.append(
            f"{len(dropped)} feature(s) dropped in QC, {len(features)} kept."
        )

    if settings.stratify_by_media and "media" in frame.columns:
        groups = {
            str(name): group
            for name, group in frame.groupby(frame["media"].fillna("UNKNOWN"))
        }
    else:
        groups = {"ALL": frame}
        if settings.stratify_by_media:
            warnings.append(
                "No 'media' column found, so all colonies were pooled. If they "
                "were grown on different media, a medium effect may dominate PC1."
            )

    per_media: dict[str, MediaResult] = {}
    tables: list[pd.DataFrame] = []
    for media, group in groups.items():
        result = score_media(group.reset_index(drop=True), features, media, settings)
        per_media[media] = result
        warnings.extend(f"[{media}] {w}" for w in result.warnings)
        if "PC1_score" in result.table.columns:
            table = result.table.copy()
            table["media_group"] = media
            tables.append(table)

    combined = (
        pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    )
    return PhenotypeResult(per_media, combined, settings, dropped, warnings)


def write_phenotypes(result: PhenotypeResult, out_dir: str | Path) -> list[Path]:
    """
    Write GWAS-ready tables.

    Tab-separated, because GEMMA, PLINK and GAPIT all read TSV happily and a
    comma inside a strain name cannot corrupt the file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for media, media_result in result.per_media.items():
        if "PC1_score" not in media_result.table.columns:
            continue
        path = out_dir / f"phenotypes_{media}.tsv"
        media_result.table.to_csv(path, sep="\t", index=False)
        written.append(path)

        loadings_path = out_dir / f"pca_loadings_{media}.tsv"
        media_result.loadings.to_csv(loadings_path, sep="\t")
        written.append(loadings_path)

    if not result.combined.empty:
        path = out_dir / "phenotypes_all_media.tsv"
        result.combined.to_csv(path, sep="\t", index=False)
        written.append(path)

    if result.dropped_features:
        path = out_dir / "features_dropped_in_qc.tsv"
        pd.DataFrame(
            sorted(result.dropped_features.items()), columns=["feature", "reason"]
        ).to_csv(path, sep="\t", index=False)
        written.append(path)

    return written


def write_diagnostics_pdf(result: PhenotypeResult, path: str | Path) -> Path:
    """
    A PDF of PCA diagnostics, one page group per medium.

    Four plots per medium:

    * **Scree** - how much variance each PC explains. Look for an elbow. If PC1
      is not clearly above the rest, it is a weak phenotype.
    * **PC1 vs PC2** - the colonies in phenotype space, outliers marked. A tight
      cloud with a few distant points is healthy; several clusters may mean a
      batch effect rather than biology.
    * **Top PC1 loadings** - which features drive the phenotype. This is how you
      say what PC1 *means* biologically.
    * **PC1 histogram** - GWAS assumes a roughly normal phenotype. A strongly
      bimodal distribution needs a second look before association testing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(path) as pdf:
        for media, media_result in result.per_media.items():
            if media_result.explained_variance.size == 0:
                continue

            figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
            figure.suptitle(
                f"PCA diagnostics - {media}  "
                f"({media_result.n_samples} colonies, "
                f"{len(media_result.features_used)} features)",
                fontsize=13,
            )

            variance = media_result.explained_variance
            axis = axes[0, 0]
            axis.bar(np.arange(1, len(variance) + 1), variance * 100, color="#4477aa")
            axis.plot(
                np.arange(1, len(variance) + 1),
                np.cumsum(variance) * 100,
                "o-", color="#cc6677", markersize=3, label="cumulative",
            )
            axis.set_xlabel("Principal component")
            axis.set_ylabel("Variance explained (%)")
            axis.set_title("Scree plot")
            axis.legend(fontsize=8)

            table = media_result.table
            axis = axes[0, 1]
            if "PC2_score" in table.columns:
                outlier = table["is_outlier"] == 1
                axis.scatter(
                    table.loc[~outlier, "PC1_score"],
                    table.loc[~outlier, "PC2_score"],
                    s=14, alpha=0.75, color="#4477aa", label="kept",
                )
                if outlier.any():
                    axis.scatter(
                        table.loc[outlier, "PC1_score"],
                        table.loc[outlier, "PC2_score"],
                        s=26, color="#cc3311", marker="x", label="outlier",
                    )
                axis.legend(fontsize=8)
                axis.set_xlabel("PC1 score")
                axis.set_ylabel("PC2 score")
            axis.set_title("Colonies in phenotype space")

            axis = axes[1, 0]
            if not media_result.loadings.empty:
                pc1 = media_result.loadings["PC1"].sort_values(key=abs, ascending=False)
                top = pc1.head(10)[::-1]
                axis.barh(
                    range(len(top)),
                    top.values,
                    color=["#cc6677" if v < 0 else "#228833" for v in top.values],
                )
                axis.set_yticks(range(len(top)))
                axis.set_yticklabels(
                    [name[:34] for name in top.index], fontsize=7
                )
                axis.axvline(0, color="black", lw=0.6)
            axis.set_title("Top 10 features driving PC1")
            axis.set_xlabel("Loading")

            axis = axes[1, 1]
            if "PC1_score" in table.columns:
                axis.hist(
                    table["PC1_score"].dropna(), bins=30,
                    color="#4477aa", edgecolor="white",
                )
            axis.set_title("PC1 score distribution")
            axis.set_xlabel("PC1 score")
            axis.set_ylabel("Colonies")

            figure.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(figure)
            plt.close(figure)

        info = pdf.infodict()
        info["Title"] = "FungiCapture PCA diagnostics"
        info["Subject"] = "GWAS phenotype preparation"

    return path
