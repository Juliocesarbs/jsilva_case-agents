import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
TOOLS_PATH = ROOT / "data" / "tools_registry.json"
EVAL_PATH = ROOT / "data" / "eval_dataset.json"

CANDIDATE_SIZES = [5, 10, 20]
ALPHAS = [0.0, 0.5, 1.0, 1.5]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def reciprocal_rank(ranking, expected):
    for position, name in enumerate(ranking, start=1):
        if name == expected:
            return 1 / position
    return 0.0


def evaluate(rankings, expected_tools):
    top1 = []
    hit2 = []
    mrr = []

    for ranking, expected in zip(
        rankings,
        expected_tools,
    ):
        top1.append(
            int(ranking[0] == expected)
        )

        hit2.append(
            int(expected in ranking[:2])
        )

        mrr.append(
            reciprocal_rank(ranking, expected)
        )

    return {
        "top1": np.mean(top1),
        "hit2": np.mean(hit2),
        "mrr": np.mean(mrr),
    }


def build_ranking(
    similarities,
    tool_names,
    token_counts,
    candidate_size,
    alpha,
):
    # Primeira etapa: candidate retrieval puramente lexical.
    lexical_indexes = np.argsort(
        similarities
    )[::-1]

    candidate_indexes = lexical_indexes[
        :candidate_size
    ]

    # Segunda etapa: reranking apenas dos candidatos.
    candidate_scores = (
        similarities[candidate_indexes]
        / np.power(
            token_counts[candidate_indexes],
            alpha,
        )
    )

    reranked_positions = np.argsort(
        candidate_scores
    )[::-1]

    reranked_indexes = candidate_indexes[
        reranked_positions
    ]

    return [
        tool_names[index]
        for index in reranked_indexes
    ]


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

    tool_texts = [
        tool["name"].replace("_", " ")
        for tool in tools
    ]

    token_counts = np.array(
        [
            len(tool["name"].split("_"))
            for tool in tools
        ],
        dtype=float,
    )

    queries = [
        item["query"]
        for item in agent_cases
    ]

    expected_tools = [
        item["expected_tool"]
        for item in agent_cases
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    tool_matrix = vectorizer.fit_transform(
        tool_texts
    )

    query_matrix = vectorizer.transform(
        queries
    )

    similarities = cosine_similarity(
        query_matrix,
        tool_matrix,
    )

    print(f"Queries AGENT: {len(agent_cases)}")

    print(
        f"\n{'TOP-N':<8}"
        f"{'ALPHA':<8}"
        f"{'TOP1':>10}"
        f"{'HIT@2':>10}"
        f"{'MRR':>10}"
    )

    print("-" * 46)

    results = []

    for candidate_size in CANDIDATE_SIZES:
        for alpha in ALPHAS:
            rankings = []

            for scores in similarities:
                ranking = build_ranking(
                    scores,
                    tool_names,
                    token_counts,
                    candidate_size,
                    alpha,
                )

                rankings.append(ranking)

            metrics = evaluate(
                rankings,
                expected_tools,
            )

            result = {
                "candidate_size": candidate_size,
                "alpha": alpha,
                **metrics,
            }

            results.append(result)

            print(
                f"{candidate_size:<8}"
                f"{alpha:<8.1f}"
                f"{metrics['top1']:>9.1%}"
                f"{metrics['hit2']:>9.1%}"
                f"{metrics['mrr']:>10.3f}"
            )

    best = max(
        results,
        key=lambda result: (
            result["hit2"],
            result["mrr"],
            result["top1"],
        ),
    )

    print("\nMelhor configuração observada")
    print("-" * 40)

    print(
        f"Candidate Top-N: "
        f"{best['candidate_size']}"
    )

    print(
        f"Alpha:           "
        f"{best['alpha']:.1f}"
    )

    print(
        f"Top-1:           "
        f"{best['top1']:.1%}"
    )

    print(
        f"Hit@2:           "
        f"{best['hit2']:.1%}"
    )

    print(
        f"MRR:             "
        f"{best['mrr']:.3f}"
    )


if __name__ == "__main__":
    main()