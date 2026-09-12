import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]

TOOLS_PATH = ROOT / "data" / "tools_registry.json"
EVAL_PATH = ROOT / "data" / "eval_dataset.json"

RRF_K = 60


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_name(name):
    return name.replace("_", " ")


def reciprocal_rank(ranking, expected_tool):
    for position, tool_name in enumerate(ranking, start=1):
        if tool_name == expected_tool:
            return 1 / position

    return 0.0


def build_rankings(tool_texts, queries, tool_names, analyzer):
    if analyzer == "word":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

    elif analyzer == "char":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
        )

    else:
        raise ValueError(analyzer)

    tool_matrix = vectorizer.fit_transform(tool_texts)

    rankings = []

    for query in queries:
        query_vector = vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            tool_matrix,
        )[0]

        indexes = np.argsort(scores)[::-1]

        rankings.append(
            [tool_names[index] for index in indexes]
        )

    return rankings


def fuse_rankings(word_ranking, char_ranking):
    scores = {}

    for position, tool in enumerate(
        word_ranking,
        start=1,
    ):
        scores[tool] = scores.get(tool, 0.0) + (
            1 / (RRF_K + position)
        )

    for position, tool in enumerate(
        char_ranking,
        start=1,
    ):
        scores[tool] = scores.get(tool, 0.0) + (
            1 / (RRF_K + position)
        )

    return sorted(
        scores,
        key=scores.get,
        reverse=True,
    )


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
            reciprocal_rank(
                ranking,
                expected,
            )
        )

    return {
        "top1": np.mean(top1),
        "hit2": np.mean(hit2),
        "mrr": np.mean(mrr),
    }


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

    word_rankings = build_rankings(
        tool_texts,
        queries,
        tool_names,
        "word",
    )

    char_rankings = build_rankings(
        tool_texts,
        queries,
        tool_names,
        "char",
    )

    fused_rankings = [
        fuse_rankings(word, char)
        for word, char in zip(
            word_rankings,
            char_rankings,
        )
    ]

    experiments = {
        "word": word_rankings,
        "char": char_rankings,
        "rrf": fused_rankings,
    }

    print(f"Tools: {len(tools)}")
    print(f"Queries AGENT: {len(agent_cases)}")

    print(
        f"\n{'METHOD':<12}"
        f"{'TOP1':>10}"
        f"{'HIT@2':>10}"
        f"{'MRR':>10}"
    )

    print("-" * 42)

    for name, rankings in experiments.items():
        metrics = evaluate(
            rankings,
            expected_tools,
        )

        print(
            f"{name:<12}"
            f"{metrics['top1']:>9.1%}"
            f"{metrics['hit2']:>9.1%}"
            f"{metrics['mrr']:>10.3f}"
        )

    print("\nComplementaridade Word × Char")

    word_only = 0
    char_only = 0
    both = 0
    neither = 0

    for expected, word, char in zip(
        expected_tools,
        word_rankings,
        char_rankings,
    ):
        word_hit = expected in word[:2]
        char_hit = expected in char[:2]

        if word_hit and char_hit:
            both += 1
        elif word_hit:
            word_only += 1
        elif char_hit:
            char_only += 1
        else:
            neither += 1

    print(f"Ambos acertam Top-2: {both}")
    print(f"Somente Word:        {word_only}")
    print(f"Somente Char:        {char_only}")
    print(f"Nenhum:              {neither}")


if __name__ == "__main__":
    main()