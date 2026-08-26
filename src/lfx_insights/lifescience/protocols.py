"""Built-in protocol templates for common life-science assays and pipelines.

Each :func:`generate_protocol` returns a :class:`~consilium.models.Protocol` with an
ordered list of standard, methodologically correct ``steps`` and a ``qc_checklist`` of
the checks a competent analyst would run. Templates are deterministic and carry an
explicit disclaimer in ``notes``: they are starting points, not validated SOPs.

Correctness is paramount: the steps follow accepted community practice (e.g. RNA-seq
quality control before alignment, no-template controls for PCR, loading controls for
western blots, base-quality recalibration before variant calling) and reproducibility
hooks (replicates, fixed seeds, pinned versions) are baked into the QC checklists.
"""

from __future__ import annotations

from lfx_insights.models import Protocol

_NOTES = "Template â€” adapt to your platform/organism; not a substitute for a validated SOP."


# Each entry: kind -> (human-readable name, ordered steps, qc checklist).
_TEMPLATES: dict[str, tuple[str, list[str], list[str]]] = {
    "rna_seq": (
        "RNA-seq differential expression",
        [
            "Extract total RNA and assess integrity (RIN/RQN) before library prep.",
            "Select library type (poly-A enrichment vs rRNA depletion) appropriate to the sample.",
            "Prepare stranded libraries with at least three biological replicates per condition.",
            "Run raw-read QC (per-base quality, adapter content, duplication) with FastQC/MultiQC.",
            "Trim adapters and low-quality bases; re-run QC to confirm improvement.",
            "Align to the reference genome (or pseudo-align to transcriptome); record the rate.",
            "Quantify gene/transcript counts against a versioned annotation.",
            "Filter low-count features and normalize (e.g. DESeq2 median-of-ratios / edgeR TMM).",
            "Test for differential expression with a negative-binomial model; control the FDR.",
            "Inspect sample clustering (PCA/heatmap) for replicate concordance and batch effects.",
        ],
        [
            "RNA integrity (RIN/RQN) recorded and above the agreed threshold per sample.",
            "Adapter trimming QC: residual adapter content negligible after trimming.",
            "Alignment/assignment rate reported and within the expected range for the protocol.",
            "Library complexity / duplication assessed; flag low-complexity libraries.",
            "Replicate reproducibility checked (PCA/clustering, within-group correlation).",
            "Reproducibility: fixed random seeds and pinned tool/annotation versions recorded.",
        ],
    ),
    "variant_calling": (
        "Germline short-variant calling (DNA-seq)",
        [
            "Run raw-read QC (FastQC/MultiQC) and trim adapters/low-quality bases.",
            "Align reads to the reference genome with a versioned aligner (e.g. BWA-MEM).",
            "Sort, index, and mark (or remove) PCR/optical duplicates.",
            "Apply base-quality score recalibration against known variant sites.",
            "Call variants per sample (e.g. HaplotypeCaller GVCF), then joint-genotype the cohort.",
            "Apply variant filtration (VQSR or hard filters) separately for SNPs and indels.",
            "Annotate variants with functional consequence and population frequencies.",
            "Evaluate calls against a truth set / benchmark sample where available.",
        ],
        [
            "Mean coverage and uniformity assessed; flag under-covered samples/regions.",
            "Duplication rate recorded; excessive duplication investigated.",
            "Contamination / sample-swap check (e.g. cross-sample fingerprint concordance).",
            "Ti/Tv ratio and het/hom ratio within expected ranges for the assay.",
            "Concordance against a truth set (precision/recall) reported when a benchmark exists.",
            "Reproducibility: pinned reference build, tool versions, and fixed seeds recorded.",
        ],
    ),
    "pcr": (
        "Endpoint PCR amplification",
        [
            "Design and validate primers (specificity check, Tm matching, amplicon size).",
            "Prepare a clean master mix; aliquot to avoid repeated freeze-thaw.",
            "Set up reactions including a no-template control (NTC) and a positive control.",
            "Run a thermal-cycling program with an empirically optimized annealing temperature.",
            "Resolve products by gel/capillary electrophoresis against a size ladder.",
            "Confirm the expected amplicon size and the absence of non-specific products.",
        ],
        [
            "No-template control (NTC) shows no amplification (no contamination).",
            "Positive control amplifies the expected product.",
            "Single band of the expected size; no primer-dimers or off-target bands.",
            "Annealing temperature optimized (gradient) for specificity.",
            "Reproducibility: lot numbers, primer sequences, and cycling program recorded.",
        ],
    ),
    "western_blot": (
        "Western blot (immunoblot)",
        [
            "Lyse samples and quantify total protein (e.g. BCA) to load equal amounts per lane.",
            "Denature and separate proteins by SDS-PAGE alongside a molecular-weight ladder.",
            "Transfer to a membrane and verify transfer efficiency (total-protein stain).",
            "Block the membrane to minimize non-specific binding.",
            "Incubate with a validated primary antibody, then the matched secondary.",
            "Wash, develop, and image within the linear dynamic range of the detector.",
            "Quantify band intensity and normalize to a validated loading control.",
        ],
        [
            "Equal protein loading confirmed (assay + total-protein stain).",
            "Molecular-weight ladder confirms the band runs at the expected size.",
            "Loading/normalization control validated for the conditions tested.",
            "Antibody specificity validated (e.g. single band, knockout, or peptide block).",
            "Signal within the linear range (no saturated pixels) before quantification.",
            "Reproducibility: biological replicates included; antibody catalog/lot recorded.",
        ],
    ),
}

AVAILABLE_PROTOCOLS: list[str] = list(_TEMPLATES)


def generate_protocol(kind: str) -> Protocol:
    """Return a built-in :class:`~consilium.models.Protocol` template for ``kind``.

    Supported kinds are listed in :data:`AVAILABLE_PROTOCOLS`. The returned protocol
    carries ordered, standard steps and a QC checklist; it is deterministic.

    Raises:
        ValueError: if ``kind`` is not a known template (message lists the available
            kinds).
    """
    template = _TEMPLATES.get(kind)
    if template is None:
        available = ", ".join(AVAILABLE_PROTOCOLS)
        raise ValueError(f"Unknown protocol kind {kind!r}. Available: {available}.")

    name, steps, qc_checklist = template
    return Protocol(
        name=name,
        kind=kind,
        steps=list(steps),
        qc_checklist=list(qc_checklist),
        notes=_NOTES,
    )
