#!/usr/bin/env python3
"""
Benchmark script for Lab 7 - Retrieval evaluation.

Usage:
    python bench.py

Each team member should change ONLY the CHUNKER line below to their strategy.
"""

import sys
import re
from pathlib import Path

# Fix encoding on Windows
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.chunking import (
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    HeadingChunker,
)
from src.store import EmbeddingStore
from src.embeddings import _mock_embed
from src.models import Document


# ============================================================
# CHANGE ONLY THIS LINE TO YOUR CHOSEN STRATEGY
# ============================================================
# CHUNKER = RecursiveChunker(chunk_size=500)
CHUNKER = FixedSizeChunker(chunk_size=500, overlap=50)
# CHUNKER = SentenceChunker(max_sentences_per_chunk=3)
# CHUNKER = HeadingChunker(max_chars=800)


# ============================================================
# BENCHMARK QUERIES (5 queries - MUST match team's agreed queries)
# ============================================================
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Thời hạn yêu cầu đổi trả hoặc hoàn tiền dành cho người mua là bao nhiêu ngày?",
        "filter": {"audience": "buyer"},
        "gold_answer": "Được đổi mới miễn phí trong 30 ngày đầu tại TGDĐ/CellphoneS nếu lỗi kỹ thuật NSX.",
        "target_keywords": ["30 ngày", "đổi", "hoàn tiền"],
        "expected_docs": ["doi-tra-bao-hanh-cellphones-buyer", "doi-tra-bao-hanh-tgdd-buyer", "doi-tra-bao-hanh-lazada-buyer"],
    },
    {
        "id": 2,
        "query": "Thời hạn Người bán phải phản hồi và gửi khiếu nại Trả hàng/Hoàn tiền là bao nhiêu lâu?",
        "filter": {"audience": "seller"},
        "gold_answer": "Người bán có thời hạn 2 ngày (Shopee) hoặc 48 giờ (Lazada) để xử lý và gửi khiếu nại.",
        "target_keywords": ["2 ngày", "48 giờ", "khiếu nại"],
        "expected_docs": ["doi-tra-bao-hanh-shopee-seller", "doi-tra-bao-hanh-lazada-seller", "doi-tra-bao-hanh-tiki-seller"],
    },
    {
        "id": 3,
        "query": "Thời hạn bảo hành xe máy điện và Pin LFP VinFast là bao nhiêu năm?",
        "filter": None,
        "gold_answer": "Bảo hành xe là 6 năm/không giới hạn km; thời hạn bảo hành pin LFP lên tới 8 năm.",
        "target_keywords": ["6 năm", "8 năm", "Pin LFP"],
        "expected_docs": ["doi-tra-bao-hanh-vinfast-buyer"],
    },
    {
        "id": 4,
        "query": "Các trường hợp nào thiết bị di động bị từ chối bảo hành hoặc bị trừ phí khi đổi trả?",
        "filter": None,
        "gold_answer": "Từ chối khi tự ý tháo mở sửa chữa, rơi vỡ ngập nước; bị trừ 10-20% phí nếu trả máy không lỗi hoặc mất hộp/phụ kiện.",
        "target_keywords": ["rơi vỡ", "tháo", "trừ phí"],
        "expected_docs": ["doi-tra-bao-hanh-tgdd-buyer", "doi-tra-bao-hanh-cellphones-buyer"],
    },
    {
        "id": 5,
        "query": "Người bán cần chuẩn bị những bằng chứng gì khi khiếu nại đơn hàng bị trả về không nguyên vẹn?",
        "filter": {"audience": "seller"},
        "gold_answer": "Video mở kiện hàng có sự hiện diện của shipper, quay rõ 6 mặt kiện hàng nguyên vẹn và tình trạng sản phẩm bên trong.",
        "target_keywords": ["video", "6 mặt", "shipper"],
        "expected_docs": ["doi-tra-bao-hanh-shopee-seller", "doi-tra-bao-hanh-lazada-seller", "doi-tra-bao-hanh-tiki-seller"],
    },
]


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    fm_text = parts[1].strip()
    body = parts[2].strip()
    metadata = {}
    for line in fm_text.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"')
    return metadata, body


def chunk_and_create_docs(file_path: Path, chunker) -> list[Document]:
    """Read file, parse frontmatter, chunk body, create Documents."""
    content = file_path.read_text(encoding="utf-8")
    metadata, body = parse_frontmatter(content)
    if not body:
        return []
    chunks = chunker.chunk(body)
    docs = []
    for i, chunk in enumerate(chunks):
        doc_id = f"{file_path.stem}#{i}"
        doc_metadata = {**metadata, "doc_id": file_path.stem, "source": str(file_path)}
        docs.append(Document(id=doc_id, content=chunk, metadata=doc_metadata))
    return docs


def main():
    corpus_dir = Path("data/warranty-policies")
    md_files = sorted(corpus_dir.glob("*.md"))

    print(f"Loading {len(md_files)} documents from {corpus_dir}...")
    all_docs = []
    for f in md_files:
        docs = chunk_and_create_docs(f, CHUNKER)
        all_docs.extend(docs)
        print(f"  {f.name}: {len(docs)} chunks")

    print(f"\nTotal chunks: {len(all_docs)}")

    store = EmbeddingStore(collection_name="benchmark", embedding_fn=_mock_embed)
    store.add_documents(all_docs)
    print(f"Stored {store.get_collection_size()} documents in EmbeddingStore")

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for bq in BENCHMARK_QUERIES:
        qid = bq["id"]
        query = bq["query"]
        metadata_filter = bq["filter"]
        gold_answer = bq["gold_answer"]
        target_keywords = bq["target_keywords"]
        expected_docs = bq["expected_docs"]

        print(f"\n--- Query {qid}: {query} ---")
        if metadata_filter:
            print(f"Filter: {metadata_filter}")
            results = store.search_with_filter(query, top_k=3, metadata_filter=metadata_filter)
        else:
            results = store.search(query, top_k=3)

        for rank, result in enumerate(results, 1):
            score = result["score"]
            doc_id = result["metadata"].get("doc_id", "unknown")
            content = result["content"]
            content_preview = content[:150].replace("\n", " ")
            
            # Check if any expected doc matches
            doc_match = " ✓ EXPECTED" if doc_id in expected_docs else ""
            
            # Check if target keywords appear in content
            keyword_hits = [kw for kw in target_keywords if kw.lower() in content.lower()]
            keyword_str = f" [kw: {', '.join(keyword_hits)}]" if keyword_hits else ""
            
            print(f"  {rank}. score={score:.4f} doc_id={doc_id}{doc_match}{keyword_str}")
            print(f"     {content_preview}...")

        # Check if any expected doc is in top-3
        found = any(r["metadata"].get("doc_id") in expected_docs for r in results)
        print(f"  Expected doc in top-3: {'YES' if found else 'NO'}")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()