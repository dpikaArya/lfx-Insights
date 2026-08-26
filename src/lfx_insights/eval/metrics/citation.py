"""Citation precision / recall / F1 (AutoAIS / ALCE scheme).

Faithful re-implementation of ScholarQABench's
``scripts/citation_correctness_eval.py`` (Asai et al. 2024, arXiv:2411.14199),
itself derived from ALCE (Gao et al. 2023). The metric measures how well an
answer's inline ``[n]`` citation markers are *entailed* by the documents they
point at:

* **recall** â€” fraction of scored sentences whose joined cited passages
  entail the sentence (citation *recall*: is the claim supported?);
* **precision** â€” fraction of individual citations that are not redundant
  (citation *precision*: each marker is either sufficient on its own or
  necessary to the joint support â€” over-cited markers are penalised);
* **f1** â€” harmonic mean of the two.

The entailment judge is pluggable (see :class:`~lfx_insights.eval.entailment.Entailer`),
so the same arithmetic backs a deterministic lexical judge offline and an
LLM-as-judge online. Sentence segmentation uses a regex splitter (split on
``[.!?]`` followed by whitespace) as an approximation of the NLTK ``sent_tokenize``
used by the reference scripts; this is good enough for the short, well-punctuated
sentences in long-form scientific answers but is not a full sentence tokenizer.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from lfx_insights.eval.models import CitationScore

if TYPE_CHECKING:
    from lfx_insights.eval.entailment import Entailer
    from lfx_insights.eval.models import GeneratedAnswer

# Approximate NLTK sentence segmentation: break after . ! ? when followed by space.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Inline citation markers, e.g. "[3]" -> marker 3 (0-indexed into the answer's docs).
_MARKER = re.compile(r"\[(\d+)\]")
# Used to strip the markers (and any trailing space they leave) from the prose.
_MARKER_STRIP = re.compile(r"\[\d+\]")

# Sentences shorter than this (after stripping markers) are not scored â€” they are
# typically section headers or fragments, mirroring the reference scripts.
_MIN_SENTENCE_LEN = 50


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences (regex approximation of NLTK ``sent_tokenize``)."""
    return [s for s in (part.strip() for part in _SENT_SPLIT.split(text)) if s]


def _markers_in(sentence: str) -> list[int]:
    """Extract inline ``[n]`` citation markers as 0-indexed doc positions (no offset)."""
    return [int(m) for m in _MARKER.findall(sentence)]


def _strip_markers(sentence: str) -> str:
    """Return the sentence with all ``[n]`` markers removed (the entailment target)."""
    return _MARKER_STRIP.sub("", sentence).strip()


def compute_citation_prf(answer: GeneratedAnswer, entailer: Entailer) -> CitationScore:
    """Compute AutoAIS/ALCE citation precision, recall, and F1 for one answer.

    ``answer.docs`` is the ordered list of cited documents; an inline marker
    ``[n]`` refers to ``answer.docs[n]`` (0-indexed, no offset). The premise for
    marker ``n`` is ``answer.docs[n].text``.

    Scoring proceeds sentence by sentence over ``answer.text``:

    * Sentences whose marker-stripped length is below 50 characters are skipped
      entirely (not counted in ``n_sentences``).
    * A scored sentence with no markers inherits the previous *scored* sentence's
      markers (the running ``previous_markers``; initially empty).
    * **Recall** (``joint_entail``): 0.0 if the sentence has no markers or any
      marker is out of range ``[0, len(docs))``; otherwise 1.0 if the entailer
      reports that the joined cited passages entail the marker-stripped sentence,
      else 0.0. ``recall`` is the mean of ``joint_entail`` over scored sentences
      (0.0 if there are none).
    * **Precision**: each in-range marker contributes to ``total_citations``. If
      the sentence is jointly supported, a single citation is precise; with
      multiple citations each is precise iff it alone entails the sentence
      (sufficient) or the other cited passages together do *not* entail it
      (necessary). Citations of unsupported sentences never count as precise.
      ``precision`` is ``total_precise / total_citations`` over the whole answer
      (0.0 if there are no citations).
    * **F1** is the harmonic mean of precision and recall (0.0 if either is 0).

    Args:
        answer: the generated answer (text + ordered cited docs).
        entailer: an entailment judge exposing ``name`` and
            ``entails(premise, hypothesis) -> bool``.

    Returns:
        A :class:`~lfx_insights.eval.models.CitationScore`.
    """
    docs = answer.docs
    n_docs = len(docs)

    entails = []  # per-scored-sentence joint_entail (0.0 / 1.0)
    total_citations = 0
    total_precise = 0
    previous_markers: list[int] = []

    for sentence in _split_sentences(answer.text):
        target_sent = _strip_markers(sentence)
        if len(target_sent) < _MIN_SENTENCE_LEN:
            continue

        markers = _markers_in(sentence)
        if markers:
            previous_markers = markers
        else:
            # Inherit the previous scored sentence's markers.
            markers = previous_markers

        in_range = [m for m in markers if 0 <= m < n_docs]

        # --- recall (joint entailment) ---
        if not markers or any(not (0 <= m < n_docs) for m in markers):
            joint_entail = 0.0
        else:
            joined = " ".join(docs[m].text for m in markers)
            joint_entail = 1.0 if entailer.entails(joined, target_sent) else 0.0
        entails.append(joint_entail)

        # --- precision ---
        total_citations += len(in_range)
        if joint_entail != 1.0:
            continue
        if len(in_range) == 1:
            total_precise += 1
            continue
        for i, marker in enumerate(in_range):
            alone = docs[marker].text
            others = " ".join(docs[m].text for j, m in enumerate(in_range) if j != i)
            sufficient = entailer.entails(alone, target_sent)
            necessary = not entailer.entails(others, target_sent)
            if sufficient or necessary:
                total_precise += 1

    recall = sum(entails) / len(entails) if entails else 0.0
    precision = total_precise / total_citations if total_citations else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0

    return CitationScore(
        precision=precision,
        recall=recall,
        f1=f1,
        n_sentences=len(entails),
        n_citations=total_citations,
        judge=entailer.name,
    )
