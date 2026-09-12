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


def normalize_name(name):
    return name.replace("_", " ")


def token_count(name):
    return len(name.split("_"))


def main():
    tools = load_json(TOOLS_PATH)
    eval_data = load_json(EVAL_PATH)

    tool_names = [tool["name"] for tool in tools]
    tool_texts = [normalize_name(name) for name in tool_names]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    tool_matrix = vectorizer.fit_transform(tool_texts)

    agent_cases = [
        item
        for item in eval_data
        if item["expected_route"] == "AGENT"
        and item.get("expected_tool")
    ]

    errors = []

    for item in agent_cases:
        query = item["query"]
        expected = item["expected_tool"]

        query_vector = vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            tool_matrix,
        )[0]

        ranking_indexes = np.argsort(scores)[::-1]

        ranked_names = [
            tool_names[index]
            for index in ranking_indexes
        ]

        top1 = ranked_names[0]
        expected_rank = ranked_names.index(expected) + 1

        if top1 == expected:
            continue

        expected_tokens = token_count(expected)
        top1_tokens = token_count(top1)

        errors.append(
            {
                "query": query,
                "expected": expected,
                "top1": top1,
                "expected_rank": expected_rank,
                "expected_tokens": expected_tokens,
                "top1_tokens": top1_tokens,
                "expected_chars": len(expected),
                "top1_chars": len(top1),
            }
        )

    shorter_tokens = 0
    same_tokens = 0
    longer_tokens = 0

    shorter_chars = 0
    same_chars = 0
    longer_chars = 0

    print(f"Erros Top-1 analisados: {len(errors)}\n")

    for error in errors:
        if error["expected_tokens"] < error["top1_tokens"]:
            shorter_tokens += 1
            token_relation = "EXPECTED MENOR"
        elif error["expected_tokens"] == error["top1_tokens"]:
            same_tokens += 1
            token_relation = "MESMO TAMANHO"
        else:
            longer_tokens += 1
            token_relation = "EXPECTED MAIOR"

        if error["expected_chars"] < error["top1_chars"]:
            shorter_chars += 1
        elif error["expected_chars"] == error["top1_chars"]:
            same_chars += 1
        else:
            longer_chars += 1

        print(f"Query: {error['query']}")
        print(
            f"Expected: {error['expected']} "
            f"({error['expected_tokens']} tokens)"
        )
        print(
            f"Top 1:    {error['top1']} "
            f"({error['top1_tokens']} tokens)"
        )
        print(f"Rank expected: {error['expected_rank']}")
        print(f"Relação: {token_relation}")
        print()

    total = len(errors)

    print("Resumo por número de tokens")
    print("-" * 35)
    print(
        f"Expected menor: {shorter_tokens:>2}/{total} "
        f"({shorter_tokens / total:.1%})"
    )
    print(
        f"Mesmo tamanho:  {same_tokens:>2}/{total} "
        f"({same_tokens / total:.1%})"
    )
    print(
        f"Expected maior: {longer_tokens:>2}/{total} "
        f"({longer_tokens / total:.1%})"
    )

    print("\nResumo por número de caracteres")
    print("-" * 35)
    print(
        f"Expected menor: {shorter_chars:>2}/{total} "
        f"({shorter_chars / total:.1%})"
    )
    print(
        f"Mesmo tamanho:  {same_chars:>2}/{total} "
        f"({same_chars / total:.1%})"
    )
    print(
        f"Expected maior: {longer_chars:>2}/{total} "
        f"({longer_chars / total:.1%})"
    )

    ranks = [
        error["expected_rank"]
        for error in errors
    ]

    print("\nPosição da tool esperada nos erros")
    print("-" * 35)
    print(f"Rank médio:   {np.mean(ranks):.2f}")
    print(f"Rank mediano: {np.median(ranks):.1f}")
    print(f"Melhor rank:  {min(ranks)}")
    print(f"Pior rank:    {max(ranks)}")


if __name__ == "__main__":
    main()