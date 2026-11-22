"""Scratch helper: raw dense retrieval checks against the Chroma index."""
import sys

sys.path.insert(0, ".")

import src.config as config
from src.indexing.build_index import load_vector_store

QUERIES = [
    "What is AWS Lambda?",
    "How is Kinesis Data Streams different from Amazon Data Firehose?",
    "Which services are in the Analytics category?",
    "How much does Lambda cost per request?",
]

store = load_vector_store()
for query in QUERIES:
    print("=" * 78)
    print(f"QUERY: {query}")
    results = store.similarity_search_with_score(
        f"{config.QUERY_PREFIX}{query}", k=config.TOP_K
    )
    for document, distance in results:
        metadata = document.metadata
        print(
            f"  {1 - distance:.3f}  p{metadata['page_start']}-{metadata['page_end']}"
            f"  [{metadata['doc_type']}]  {metadata['section_path']}"
        )
