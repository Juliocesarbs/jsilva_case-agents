import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]

TOOLS_PATH = ROOT / "data" / "tools_registry.json"
EVAL_PATH = ROOT / "data" / "eval_dataset.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def tool_name(tool):
    return tool["name"].replace("_", " ")


def build_representation(tool, strategy):
    name = tool_name(tool)
    description = tool["description"]
    category = tool["category"]

    if strategy == "name":
        return name

    if strategy == "description":
        return description

    if strategy == "name_description":
        return f"{name} {description}"

    if strategy == "name_description_category":
        return f"{name} {description} {category}"

    if strategy == "name_boosted":
        return f"{name} {name} {description} {category}"

    raise ValueError(
        f"Estratégia desconhecida: {strategy}"
    )


def reciprocal_rank(ranking, expected_tool):
    for position, name in enumerate(ranking, start=1):
        if name == expected_tool:
            return 1 / position

    return 0.0


def evaluate_strategy(
    tools,
    agent_cases,
    strategy,
):
    tool_texts = [
        build_representation(
            tool,
            strategy,
        )
        for tool in tools
    ]

    tool_names = [
        tool["name"]
        for tool in tools
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

    top1_hits = []
    top2_hits = []
    reciprocal_ranks = []

    for item in agent_cases:
        query = item["query"]
        expected_tool = item["expected_tool"]

        query_vector = vectorizer.transform(
            [query]
        )

        scores = cosine_similarity(
            query_vector,
            tool_matrix,
        )[0]

        ranking_indexes = np.argsort(
            scores
        )[::-1]

        ranked_names = [
            tool_names[index]
            for index in ranking_indexes
        ]

        top1_hits.append(
            int(
                ranked_names[0]
                == expected_tool
            )
        )

        top2_hits.append(
            int(
                expected_tool
                in ranked_names[:2]
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                ranked_names,
                expected_tool,
            )
        )

    return {
        "strategy": strategy,
        "top1": np.mean(top1_hits),
        "hit_at_2": np.mean(top2_hits),
        "mrr": np.mean(reciprocal_ranks),
    }


def main():
    tools = load_json(
        TOOLS_PATH
    )

    eval_data = load_json(
        EVAL_PATH
    )

    agent_cases = [
        item
        for item in eval_data
        if item["expected_route"] == "AGENT"
        and item.get("expected_tool")
    ]

    strategies = [
        "name",
        "description",
        "name_description",
        "name_description_category",
        "name_boosted",
    ]

    print(
        f"Tools: {len(tools)}"
    )

    print(
        f"Queries AGENT: "
        f"{len(agent_cases)}"
    )

    print(
        "\n"
        f"{'REPRESENTATION':<28}"
        f"{'TOP1':>10}"
        f"{'HIT@2':>10}"
        f"{'MRR':>10}"
    )

    print(
        "-" * 58
    )

    results = []

    for strategy in strategies:
        result = evaluate_strategy(
            tools,
            agent_cases,
            strategy,
        )

        results.append(
            result
        )

        print(
            f"{strategy:<28}"
            f"{result['top1']:>9.1%}"
            f"{result['hit_at_2']:>9.1%}"
            f"{result['mrr']:>10.3f}"
        )

    best = max(
        results,
        key=lambda item: (
            item["hit_at_2"],
            item["mrr"],
            item["top1"],
        ),
    )

    print(
        "\nMelhor estratégia "
        "pelo critério Hit@2 → MRR → Top1:"
    )

    print(
        best["strategy"]
    )


if __name__ == "__main__":
    main()