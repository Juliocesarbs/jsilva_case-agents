"""Pilar 2 — Seleção de Tools Relevantes.

Seleciona as tools mais relevantes para a query antes de qualquer chamada ao LLM.
A estratégia usa TF-IDF sobre o nome das tools, seguido de um reranking simples
para reduzir a preferência por tools excessivamente específicas.
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

        self._tool_matrix = self._vectorizer.fit_transform(tool_texts)
        self._fitted = True

        return self

    def search(self, query: str, k: int = 2) -> RetrievalResult:
        """Retorna as top-k tools mais relevantes para a query."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de search().")

        start = time.perf_counter()

        query_vector = self._vectorizer.transform([query])
        similarities = cosine_similarity(
            query_vector,
            self._tool_matrix,
        )[0]

        candidate_size = min(
            max(self._candidate_size, k),
            len(self._tools),
        )

        lexical_ranking = np.argsort(similarities)[::-1]
        candidate_indexes = lexical_ranking[:candidate_size]

        candidate_scores = (
            similarities[candidate_indexes]
            / np.power(
                self._token_counts[candidate_indexes],
                self._generality_alpha,
            )
        )

        reranked_positions = np.argsort(candidate_scores)[::-1]
        top_positions = reranked_positions[:k]

        matches = [
            ToolMatch(
                name=self._tools[candidate_indexes[position]].name,
                score=float(candidate_scores[position]),
            )
            for position in top_positions
        ]

        latency_ms = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            matches=matches,
            latency_ms=latency_ms,
        )