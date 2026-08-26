"""LLM-backed, corpus-grounded generation (hypotheses, questions, drafts, review).

Every generated citation is verified against the corpus; ungrounded references are
dropped, never asserted. Shared helpers live in :mod:`lfx_insights.generation.common`.
"""
