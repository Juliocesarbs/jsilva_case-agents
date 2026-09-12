import json
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
TOOLS_PATH = ROOT / "data" / "tools_registry.json"
EVAL_PATH = ROOT / "data" / "eval_dataset.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_name(name):
    return name.replace("_", " ")


def reciprocal_rank(ranked_names, expected_tool):
    for position, name in enumerate(ranked_names, start=1):
        if name == expected_tool:
            return 1 / position

    return 0.0


def evaluate_rankings(rankings, expected_tools):
    top1 = []
    hit2 = []
    reciprocal_ranks = []

    for ranked_names, expected_tool in zip(
        rankings,
        expected_tools,
    ):
        top1.append(
            int(ranked_names[0] == expected_tool)
        )

        hit2.append(
            int(expected_tool in ranked_names[:2])
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                ranked_names,
                expected_tool,
            )
        )

    return {
        "top1": np.mean(top1),
        "hit_at_2": np.mean(hit2),
        "mrr": np.mean(reciprocal_ranks),
    }


def word_rankings(tool_texts, queries, tool_names):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    tool_matrix = vectorizer.fit_transform(
        tool_texts
    )

    rankings = []

    for query in queries:
        query_vector = vectorizer.transform(
            [query]
        )

        scores = cosine_similarity(
            query_vector,
            tool_matrix,
        )[0]

        indexes = np.argsort(scores)[::-1]

        rankings.append(
            [tool_names[index] for index in indexes]
        )

    return rankings


def char_rankings(tool_texts, queries, tool_names):
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
    )

    tool_matrix = vectorizer.fit_transform(
        tool_texts
    )

    rankings = []

    for query in queries:
        query_vector = vectorizer.transform(
            [query]
        )

        scores = cosine_similarity(
            query_vector,
            tool_matrix,
        )[0]

        indexes = np.argsort(scores)[::-1]

        rankings.append(
            [tool_names[index] for index in indexes]
        )

    return rankings


def word_char_rankings(
    tool_texts,
    queries,
    tool_names,
):
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
    )

    word_tools = word_vectorizer.fit_transform(
        tool_texts
    )

    char_tools = char_vectorizer.fit_transform(
        tool_texts
    )

    tool_matrix = hstack(
        [word_tools, char_tools]
    )

    rankings = []

    for query in queries:
        word_query = word_vectorizer.transform(
            [query]
        )

        char_query = char_vectorizer.transform(
            [query]
        )

        query_matrix = hstack(
            [word_query, char_query]
        )

        scores = cosine_similarity(
            query_matrix,
            tool_matrix,
        )[0]

        indexes = np.argsort(scores)[::-1]

        rankings.append(
            [tool_names[index] for index in indexes]
        )

    return rankings


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
        normalize_name(tool["name"])
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

    experiments = {
        "word": word_rankings(
            tool_texts,
            queries,
            tool_names,
        ),
        "char": char_rankings(
            tool_texts,
            queries,
            tool_names,
        ),
        "word_char": word_char_rankings(
            tool_texts,
            queries,
            tool_names,
        ),
    }

    print(f"Tools: {len(tools)}")
    print(f"Queries AGENT: {len(agent_cases)}")

    print(
        f"\n{'FEATURES':<18}"
        f"{'TOP1':>10}"
        f"{'HIT@2':>10}"
        f"{'MRR':>10}"
    )

    print("-" * 48)

    results = {}

    for name, rankings in experiments.items():
        metrics = evaluate_rankings(
            rankings,
            expected_tools,
        )

        results[name] = metrics

        print(
            f"{name:<18}"
            f"{metrics['top1']:>9.1%}"
            f"{metrics['hit_at_2']:>9.1%}"
            f"{metrics['mrr']:>10.3f}"
        )

    best = max(
        results,
        key=lambda name: (
            results[name]["hit_at_2"],
            results[name]["mrr"],
            results[name]["top1"],
        ),
    )

    print(
        "\nMelhor estratégia "
        "por Hit@2 → MRR → Top1:"
    )
    print(best)


if __name__ == "__main__":
    main()