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


def build_tool_text(tool):
    # Baseline simples: nenhuma regra específica por tool.
    name = tool["name"].replace("_", " ")
    return f"{name} {tool['description']} {tool['category']}"


def reciprocal_rank(ranked_names, expected_tool):
    if not expected_tool:
        return None

    for position, name in enumerate(ranked_names, start=1):
        if name == expected_tool:
            return 1 / position

    return 0.0


def main():
    tools = load_json(TOOLS_PATH)
    eval_data = load_json(EVAL_PATH)

    tool_texts = [build_tool_text(tool) for tool in tools]
    tool_names = [tool["name"] for tool in tools]

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

    top1_hits = []
    top2_hits = []
    reciprocal_ranks = []

    errors = []

    for item in agent_cases:
        query = item["query"]
        expected_tool = item["expected_tool"]

        query_vector = vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            tool_matrix,
        )[0]

        ranking = np.argsort(scores)[::-1]
        ranked_names = [tool_names[index] for index in ranking]

        top1 = ranked_names[0]
        top2 = ranked_names[:2]

        top1_hit = int(top1 == expected_tool)
        top2_hit = int(expected_tool in top2)

        top1_hits.append(top1_hit)
        top2_hits.append(top2_hit)

        reciprocal_ranks.append(
            reciprocal_rank(ranked_names, expected_tool)
        )

        if not top2_hit:
            errors.append(
                {
                    "query": query,
                    "expected": expected_tool,
                    "top1": top1,
                    "top2": top2[1],
                    "top1_score": scores[ranking[0]],
                    "top2_score": scores[ranking[1]],
                }
            )

    print(f"Tools: {len(tools)}")
    print(f"Queries AGENT avaliadas: {len(agent_cases)}")

    print("\nBaseline: Word TF-IDF + cosine similarity")
    print(f"Top-1 accuracy: {np.mean(top1_hits):.1%}")
    print(f"Hit@2:          {np.mean(top2_hits):.1%}")
    print(f"MRR:            {np.mean(reciprocal_ranks):.3f}")

    print("\nErros fora do Top-2:")

    if not errors:
        print("Nenhum.")
        return

    for error in errors:
        print(f"\nQuery:    {error['query']}")
        print(f"Esperado: {error['expected']}")
        print(
            f"Top 1:    {error['top1']} "
            f"({error['top1_score']:.3f})"
        )
        print(
            f"Top 2:    {error['top2']} "
            f"({error['top2_score']:.3f})"
        )


if __name__ == "__main__":
    main()