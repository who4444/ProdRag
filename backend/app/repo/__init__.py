"""Codebase indexing for repo-grounded R&D (repo RAG).

walk -> AST-aware chunk -> embed -> Qdrant `code_chunks` (dense + BM25).
"""
