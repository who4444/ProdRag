"""Run control layer — LangGraph state machine over the R&D phases.

understand -> clarify(interrupt) -> codebase_map -> assess x3 -> evidence gate ->
judge -> reframe(interrupt) -> PoC -> debug(<=1) -> evaluate -> finalize.
Postgres checkpoints make runs durable/resumable; every node appends to the
evidence ledger.
"""
