-- KN-VS-04: HNSW index, cosine distance. Build parameters are fixed here; search-time
-- settings (ef_search, iterative scan) are set per session by the caller.

CREATE INDEX kn_chunks_embedding_hnsw
    ON kn_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
