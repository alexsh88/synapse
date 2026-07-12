"""Shared constants for the native relationship vector index over RELATES_TO.fact_embedding.

Both the write pipeline (creates + queries it for dedup) and the curation engine
(queries it for candidate-pair generation) must reference the SAME index name.
Defining it here avoids the silent drift of two independent string literals.
"""

from __future__ import annotations

FACT_VECTOR_INDEX = "synapse_relates_fact_vec"
