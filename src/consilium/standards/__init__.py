"""Standards exporters + grounding.

Internal Consilium models are serialized here to the Holobiomics standards
(indicium / ASTRA) and grounded via indicium's ``verify_quote`` kernel. Sibling
packages are optional: when absent, exporters still produce standards-shaped dicts
and the grounding gate falls back to a normalized substring match.
"""

from consilium.standards.asb_export import asb_available, run_to_capsule
from consilium.standards.astra_export import astra_available, insights_to_collection
from consilium.standards.grounding import (
    require_grounded,
    text_quote_selector,
    verify_quote_in,
)
from consilium.standards.indicium_export import (
    claims_to_document,
    hypothesis_evidence,
    hypothesis_to_claim,
    indicium_available,
    sources_to_indicium,
)

__all__ = [
    "asb_available",
    "astra_available",
    "claims_to_document",
    "hypothesis_evidence",
    "hypothesis_to_claim",
    "indicium_available",
    "insights_to_collection",
    "require_grounded",
    "run_to_capsule",
    "sources_to_indicium",
    "text_quote_selector",
    "verify_quote_in",
]
