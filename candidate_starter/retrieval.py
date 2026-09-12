"""Pilar 2 — Seleção de Tools Relevantes.

O catálogo de tools está em `data/tools_registry.json`. Passar todas as tools no prompt
de um LLM não escala (estoura contexto, confunde o modelo, aumenta custo e latência).

`search(query, k=2)` deve retornar as `k` tools mais relevantes do catálogo para a query,
antes de qualquer chamada ao LLM. A estratégia de seleção/ranking é livre — escolha o que
fizer sentido e esteja preparado para justificar os trade-offs.
"""

import time
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common.interfaces import BaseToolRetriever
from common.schemas import RetrievalResult, Tool, ToolMatch


class ToolRetriever(BaseToolRetriever):
    def __init__(self) -> None:
        self._tools: List[Tool] = []
        self._fitted = False

        self._candidate_size = 10
        self._generality_alpha = 1.5

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        self._tool_matrix = None
        self._token_counts = None

    def fit(self, tools: List[Tool]) -> "ToolRetriever":
        """Indexa o catálogo de tools para busca."""
        self._tools = tools

        tool_texts = [
            tool.name.replace("_", " ")
            for tool in tools
        ]

        self._token_counts = np.array(
            [
                len(tool.name.split("_"))
                for tool in tools
            ],
            dtype=float,
        )

        self._tool_matrix = self._vectorizer.fit_transform(
            tool_texts
        )

        self._fitted = True
        return self

    def search(self, query: str, k: int = 2) -> RetrievalResult:
        """Retorna as top-k tools mais relevantes para `query`."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de search().")

        start = time.perf_counter()

        query_vector = self._vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self._tool_matrix,
        )[0]

        candidate_size = min(
            self._candidate_size,
            len(self._tools),
        )

        lexical_ranking = np.argsort(similarities)[::-1]

        candidate_indexes = lexical_ranking[
            :candidate_size
        ]

        candidate_scores = (
            similarities[candidate_indexes]
            / np.power(
                self._token_counts[candidate_indexes],
                self._generality_alpha,
            )
        )

        reranked_positions = np.argsort(
            candidate_scores
        )[::-1]

        top_positions = reranked_positions[:k]

        matches = [
            ToolMatch(
                name=self._tools[
                    candidate_indexes[position]
                ].name,
                score=float(
                    candidate_scores[position]
                ),
            )
            for position in top_positions
        ]

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        return RetrievalResult(
            matches=matches,
            latency_ms=latency_ms,
        )