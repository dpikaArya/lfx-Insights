#!/usr/bin/env python3
"""
scientific_claim_graph.py — Build a knowledge graph of scientific claims
with supporting/contradictory evidence, connected to source papers.

Outputs
-------
outputs/knowledge_base/claim_graph.json
outputs/reports/claim_graph_summary.md
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("claim_graph")

MODEL_NAME = "all-MiniLM-L6-v2"

CLAIM_PATTERNS = [
    r"(?:we found|we observe|we show|we demonstrate|"
    r"our results suggest)[^.]*\.",
    r"(?:this (?:study|research|paper|work) "
    r"(?:shows|demonstrates|suggests|indicates|"
    r"reveals|finds))[^.]*\.",
    r"(?:these findings "
    r"(?:suggest|indicate|support|demonstrate))[^.]*\.",
    r"(?:the (?:results|findings|data|analysis) "
    r"(?:suggest|indicate|reveal|show|demonstrate))"
    r"[^.]*\.",
    r"(?:there (?:is|are|was|were) (?:a|an|significant|no)"
    r"[^.]*association[^.]*\.)",
    r"(?:plays? (?:a|an) "
    r"(?:key|critical|important|significant|major) role)"
    r"[^.]*\.",
    r"(?:is (?:essential|crucial|critical|important|"
    r"necessary|key) for)[^.]*\.",
    r"(?:contributes? to|is associated with|is linked to|"
    r"is related to)[^.]*\.",
    r"(?:found that|showed that|demonstrated that|"
    r"revealed that|suggested that|indicated that)"
    r"[^.]*\.",
    r"(?:has (?:the )?potential to "
    r"(?:improve|enhance|reduce|increase|"
    r"transform|revolutionise))[^.]*\.",
    r"(?:highlights? the "
    r"(?:importance|need|role|potential|significance))"
    r"[^.]*\.",
    r"(?:our (?:analysis|findings|results|data|study) "
    r"(?:suggest|suggested|indicate|indicated|show|showed|"
    r"reveal|revealed|demonstrate|demonstrated))"
    r"[^.]*\.",
    r"(?:we (?:propose|argue|suggest|recommend|"
    r"conclude))[^.]*\.",
    r"(?:this (?:paper|article|chapter|review) "
    r"(?:argues|proposes|suggests|examines|explores|"
    r"investigates|highlights))[^.]*\.",
    r"(?:there is growing "
    r"(?:evidence|interest|recognition|concern))"
    r"[^.]*\.",
    r"(?:further (?:research|studies|work|investigation) "
    r"(?:is needed|are needed|is required|is warranted))"
    r"[^.]*\.",
]


def extract_claims_from_papers(papers_path: str) -> list[dict]:
    df = pd.read_csv(papers_path) if Path(papers_path).exists() else pd.DataFrame()
    claims: list[dict] = []

    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        doi = str(row.get("doi", ""))
        abstract = str(row.get("abstract", ""))
        if not title or title.lower() in ("nan", "", "none"):
            continue

        for pat in CLAIM_PATTERNS:
            for m in re.finditer(pat, abstract, re.IGNORECASE):
                claim_text = m.group(0).strip()[:300]
                if len(claim_text) > 30:
                    claims.append({
                        "claim": claim_text,
                        "paper_title": title[:150],
                        "doi": doi,
                        "evidence_type": "supporting",
                        "supporting_papers": [title[:150]],
                        "contradictory_papers": [],
                        "confidence_score": 0.5,
                    })
    return claims


def build_claim_graph(claims: list[dict], papers_path: str) -> dict[str, Any]:
    """Build a structured claim graph with supporting/contradictory links."""
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    df = pd.read_csv(papers_path) if Path(papers_path).exists() else pd.DataFrame()

    # Encode all paper abstracts for similarity search
    paper_texts = []
    for _, row in df.iterrows():
        t = str(row.get("title", ""))
        a = str(row.get("abstract", ""))
        paper_texts.append(f"{t} {a}")

    if paper_texts:
        paper_embs = model.encode(paper_texts, show_progress_bar=False)
        paper_embs = paper_embs / np.linalg.norm(paper_embs, axis=1, keepdims=True)
    else:
        paper_embs = np.array([])

    nodes = []
    for c in claims:
        claim_text = c["claim"]
        q_emb = model.encode([claim_text], show_progress_bar=False)[0]
        q_emb = q_emb / np.linalg.norm(q_emb)

        # Find similar papers
        supporting_dois = [c["doi"]] if c["doi"] else []
        contradictory_dois: list[str] = []

        if paper_embs.ndim == 2:
            sims = np.dot(paper_embs, q_emb)
            # Papers with similarity > 0.6 but not the source are supporting
            for idx in np.argsort(sims)[-5:][::-1]:
                if sims[idx] > 0.55:
                    row = df.iloc[idx]
                    doi = str(row.get("doi", ""))
                    if doi and doi != c["doi"]:
                        supporting_dois.append(doi)
                    # Low similarity but same topic -> potential contradiction
                    elif doi and doi != c["doi"] and 0.2 < sims[idx] < 0.35:
                        contradictory_dois.append(doi)

        # Confidence based on evidence volume
        confidence = min(0.3 + len(supporting_dois) * 0.15, 0.95)

        node = {
            "claim": claim_text,
            "supporting_evidence": [
                {"paper_title": c["paper_title"], "doi": c["doi"]}
            ],
            "supporting_papers": supporting_dois[:5],
            "contradictory_evidence": [
                {"paper_title": title, "doi": doi}
                for title, doi in [("Unknown", d) for d in contradictory_dois]
            ] if contradictory_dois else [],
            "contradictory_papers": contradictory_dois[:3],
            "confidence_score": round(confidence, 3),
        }
        nodes.append(node)

    graph = {
        "meta": {
            "generated": datetime.now().isoformat(),
            "total_claims": len(nodes),
            "description": (
                "Scientific claim knowledge graph with "
                "supporting and contradictory evidence"
            ),
        },
        "claims": nodes[:100],  # cap at 100 nodes
    }
    return graph


def _generate_summary(graph: dict) -> str:
    lines: list[str] = []
    lines.append("# Scientific Claim Graph Summary\n")
    lines.append(f"- **Claims extracted:** {len(graph.get('claims', []))}")
    lines.append(f"- **Generated:** {graph.get('meta', {}).get('generated', '')}\n")

    for c in graph.get("claims", [])[:10]:
        lines.append("### Claim")
        lines.append(f"{c['claim']}\n")
        lines.append(f"- Supporting papers: {len(c['supporting_papers'])}")
        lines.append(f"- Contradictory papers: {len(c['contradictory_papers'])}")
        lines.append(f"- Confidence: {c['confidence_score']:.2f}\n")

    total_conf = (
        np.mean([c["confidence_score"] for c in graph.get("claims", [])])
        if graph.get("claims") else 0
    )
    lines.append(f"**Average confidence:** {total_conf:.2f}\n")
    lines.append("---\n*Generated by scientific_claim_graph.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scientific claim graph.")
    parser.add_argument("--papers", type=str, default="search_results.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/knowledge_base")
    args = parser.parse_args()

    claims = extract_claims_from_papers(args.papers)
    log.info("Extracted %d claims from papers", len(claims))
    graph = build_claim_graph(claims, args.papers)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_path = out_dir / "claim_graph.json"
    with open(graph_path, "w") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    log.info("Saved -> %s", graph_path)

    report_dir = Path("outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = _generate_summary(graph)
    summary_path = report_dir / "claim_graph_summary.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    log.info("Saved -> %s", summary_path)

    print("\n--- Scientific Claim Graph Complete ---")
    print(f"  Claims extracted: {len(graph.get('claims', []))}")
    print()


if __name__ == "__main__":
    main()
