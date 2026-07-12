"""Vector store seam — intentionally empty.

Synapse stores embeddings in Neo4j's native vector index via Graphiti (BGE-M3,
1024-dim); there is no separate vector database. Qdrant was evaluated and dropped
as redundant at Synapse's scale — see docs/research/qdrant-vs-neo4j-vectors.md.

Re-introduce a dedicated vector store here only if a documented trigger fires
(~500K+ facts, a bespoke semantic layer outside Graphiti, or Graphiti shipping
first-class external-vector-store support). The standard pattern: store vector ids
as Neo4j node properties and join after retrieval.
"""
