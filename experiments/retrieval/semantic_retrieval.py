import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
TOOLS_PATH = ROOT / "data" / "tools_registry.json"
EVAL_PATH = ROOT / "data" / "eval_dataset.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def reciprocal_rank(ranking, expected):
    for position, name in enumerate(ranking, start=1):
        if name == expected:
            return 1 / position
    return 0.0


def main():
    tools = load_json(TOOLS_PATH)
    eval_data = load_json(EVAL_PATH)

    agent_cases = [
        item
        for item in eval_data
        if item["expected_route"] == "AGENT"
        and item.get("expected_tool")
    ]

    tool_names = [
        tool["name"]
        for tool in tools
    ]

    # Mantemos name-only para comparar com os
    # experimentos lexicais anteriores.
    tool_texts = [
        tool["name"].replace("_", " ")
        for tool in tools
    ]

    queries = [
        item["query"]
        for item in agent_cases
    ]

    expected_tools = [
        item["expected_tool"]
        for item in agent_cases
    ]

    print(f"Modelo: {MODEL_NAME}")
    print(f"Tools: {len(tools)}")
    print(f"Queries AGENT: {len(agent_cases)}")
    print("\nGerando embeddings...")

    model = SentenceTransformer(MODEL_NAME)

    tool_embeddings = model.encode(
        tool_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embeddings = model.encode(
        queries,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Como os vetores estão normalizados,
    # produto escalar equivale ao cosine similarity.
    similarities = (
        query_embeddings @ tool_embeddings.T
    )

    top1_hits = []
    hit2 = []
    recall5 = []
    recall10 = []
    reciprocal_ranks = []

    failures_top10 = []

    for query, expected, scores in zip(
        queries,
        expected_tools,
        similarities,
    ):
        indexes = np.argsort(scores)[::-1]

        ranking = [
            tool_names[index]
            for index in indexes
        ]

        expected_rank = ranking.index(expected) + 1

        top1_hits.append(
            int(expected_rank == 1)
        )

        hit2.append(
            int(expected_rank <= 2)
        )

        recall5.append(
            int(expected_rank <= 5)
        )

        recall10.append(
            int(expected_rank <= 10)
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                ranking,
                expected,
            )
        )

        if expected_rank > 10:
            failures_top10.append(
                {
                    "query": query,
                    "expected": expected,
                    "rank": expected_rank,
                    "top1": ranking[0],
                }
            )

    print("\nSemantic Retrieval")
    print("-" * 40)
    print(
        f"Top-1:     "
        f"{np.mean(top1_hits):.1%}"
    )
    print(
        f"Hit@2:     "
        f"{np.mean(hit2):.1%}"
    )
    print(
        f"Recall@5:  "
        f"{np.mean(recall5):.1%}"
    )
    print(
        f"Recall@10: "
        f"{np.mean(recall10):.1%}"
    )
    print(
        f"MRR:       "
        f"{np.mean(reciprocal_ranks):.3f}"
    )

    print("\nGold fora do Top-10")
    print("-" * 40)

    if not failures_top10:
        print("Nenhum.")
    else:
        for failure in failures_top10:
            print()
            print(
                f"Query:    {failure['query']}"
            )
            print(
                f"Expected: {failure['expected']}"
            )
            print(
                f"Rank:     {failure['rank']}"
            )
            print(
                f"Top 1:    {failure['top1']}"
            )


if __name__ == "__main__":
    main()