from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from llm.llm import read_llm

load_dotenv(dotenv_path=PROJECT_ROOT / "llm" / ".env")
read_llm()

RUN_FIELDS = ["query_id", "doc_id", "rank", "score"]
DEFAULT_METRIC_KS = "1,3,5,10,20"


def load_qrels(path: str | Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row["relevance"]) > 0:
                qrels[row["query_id"]].add(row["doc_id"])
    return qrels


def load_runs(path: str | Path) -> dict[str, list[str]]:
    runs: dict[str, list[str]] = defaultdict(list)
    delimiter = "\t" if str(path).endswith(".tsv") else ","
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            runs[row["query_id"]].append(row["doc_id"])
    return runs


def load_queries(path: str | Path) -> list[dict]:
    queries: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def extract_doc_id(metadata: dict, page_content: str) -> str | None:
    if isinstance(metadata, dict):
        for key in ("doc_id", "id", "source_id", "_doc_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip().startswith("DOC_"):
                return value.strip()

        for key in ("source", "filename"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip().startswith("DOC_"):
                return value.strip()

        chroma_document = metadata.get("chroma:document")
        if isinstance(chroma_document, str):
            doc_id = extract_doc_id_from_json_text(chroma_document)
            if doc_id:
                return doc_id

    return extract_doc_id_from_json_text(page_content)


def extract_doc_id_from_json_text(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    value = payload.get("doc_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def build_runs_from_project_retriever(
    queries_path: str | Path,
    collection_name: str,
    top_k: int,
    max_queries: int,
) -> dict[str, list[str]]:
    from llm.creat_retriver import creat_retriver

    retriever = creat_retriver(
        {
            "top_k": top_k,
            "retrieval_profile": "online",
            "final_rank_enabled": False,
        },
        collection_name=collection_name,
    )
    queries = load_queries(queries_path)[:max_queries]
    runs: dict[str, list[str]] = defaultdict(list)

    total = len(queries)
    progress_step = 50 if total <= 500 else 500

    for index, item in enumerate(queries, start=1):
        query_id = item["query_id"]
        docs = retriever.invoke(item["query"])

        seen_doc_ids: set[str] = set()
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            page_content = getattr(doc, "page_content", "") or ""
            doc_id = extract_doc_id(metadata, page_content)
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            runs[query_id].append(doc_id)

        if index % progress_step == 0 or index == total:
            print(f"[progress] processed {index}/{total} queries")

    return runs


def save_runs(path: str | Path, runs: dict[str, list[str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for query_id, doc_ids in runs.items():
            for rank, doc_id in enumerate(doc_ids, start=1):
                writer.writerow(
                    {
                        "query_id": query_id,
                        "doc_id": doc_id,
                        "rank": rank,
                        "score": "",
                    }
                )


def recall_at_k(qrels: dict[str, set[str]], runs: dict[str, list[str]], k: int) -> float:
    hit = 0
    total = 0
    for query_id, rels in qrels.items():
        total += 1
        docs = runs.get(query_id, [])[:k]
        if any(doc_id in rels for doc_id in docs):
            hit += 1
    return hit / total if total else 0.0


def mrr_at_k(qrels: dict[str, set[str]], runs: dict[str, list[str]], k: int) -> float:
    score = 0.0
    total = 0
    for query_id, rels in qrels.items():
        total += 1
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(runs.get(query_id, [])[:k], start=1):
            if doc_id in rels:
                reciprocal_rank = 1.0 / rank
                break
        score += reciprocal_rank
    return score / total if total else 0.0


def build_default_output_runs_path(collection_name: str, top_k: int, max_queries: int) -> Path:
    safe_collection_name = collection_name.replace("\\", "_").replace("/", "_")
    return Path(f"./runs_{safe_collection_name}_top{top_k}_q{max_queries}.csv")


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Evaluate recall and MRR with the project retriever.")
    parser.add_argument("--qrels", default=str(base_dir / "qrels.tsv"), help="Path to qrels.tsv.")
    parser.add_argument("--queries", default=str(base_dir / "queries.jsonl"), help="Path to queries.jsonl.")
    parser.add_argument("--runs-file", default="", help="Use a precomputed runs file instead of live retrieval.")
    parser.add_argument(
        "--collection-name",
        default="user_1_kb",
        help="Chroma collection name to evaluate.",
    )
    parser.add_argument("--user-id", type=int, default=0, help="Fallback user id used to derive a collection name.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of retrieved documents kept per query for evaluation.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=100,
        help="Maximum number of queries to evaluate.",
    )
    parser.add_argument(
        "--ks",
        default=DEFAULT_METRIC_KS,
        help="Comma-separated K values used for Recall@K and MRR@K.",
    )
    parser.add_argument(
        "--output-runs",
        default="",
        help="Optional output path for the generated runs CSV.",
    )
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)

    if args.runs_file:
        runs = load_runs(args.runs_file)
    else:
        collection_name = args.collection_name
        if not collection_name:
            collection_name = f"user_{args.user_id}_kb" if args.user_id > 0 else "knowledge_base"

        print(f"[info] use collection: {collection_name}")
        print(f"[info] final top_k: {args.top_k}")
        print(f"[info] max_queries: {args.max_queries}")

        runs = build_runs_from_project_retriever(
            queries_path=args.queries,
            collection_name=collection_name,
            top_k=args.top_k,
            max_queries=args.max_queries,
        )

        output_runs = (
            Path(args.output_runs)
            if args.output_runs
            else build_default_output_runs_path(
                collection_name=collection_name,
                top_k=args.top_k,
                max_queries=args.max_queries,
            )
        )
        save_runs(output_runs, runs)
        print(f"[info] runs saved: {output_runs}")

    # Only evaluate query ids that are present in the current run.
    run_query_ids = set(runs.keys())
    qrels_eval = {query_id: rels for query_id, rels in qrels.items() if query_id in run_query_ids}

    ks = [int(item.strip()) for item in args.ks.split(",") if item.strip()]
    for k in ks:
        print(f"Recall@{k}: {recall_at_k(qrels_eval, runs, k):.4f}")
        print(f"MRR@{k}: {mrr_at_k(qrels_eval, runs, k):.4f}")


if __name__ == "__main__":
    main()
