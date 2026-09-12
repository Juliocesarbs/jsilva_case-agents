"""Pilar 1 — Router de Queries.

Classifica queries entre FAST_PATH e AGENT usando um modelo local e leve,
evitando uma chamada de LLM apenas para decidir o roteamento.
"""

import time
from collections import Counter
from typing import List

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common.interfaces import BaseRouter
from common.schemas import RouteResult


class QueryRouter(BaseRouter):
    def __init__(self) -> None:
        self._fitted = False
        self._model = None

    def fit(self, texts: List[str], labels: List[str]) -> "QueryRouter":
        """Treina o router com os exemplos de `data/router_training_data.json`."""
        class_counts = Counter(labels)

        if len(class_counts) < 2:
            raise ValueError("O router precisa de pelo menos duas classes.")

        min_class_size = min(class_counts.values())

        if min_class_size < 2:
            raise ValueError(
                "Cada classe precisa de pelo menos dois exemplos para calibração."
            )

        calibration_cv = min(3, min_class_size)

        self._model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(),
                        method="sigmoid",
                        cv=calibration_cv,
                    ),
                ),
            ]
        )

        self._model.fit(texts, labels)
        self._fitted = True
        return self

    def predict(self, query: str) -> RouteResult:
        """Classifica a query e retorna rota, latência e confiança."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de predict().")

        start = time.perf_counter()

        probabilities = self._model.predict_proba([query])[0]
        best_index = probabilities.argmax()

        route = self._model.classes_[best_index]
        confidence = float(probabilities[best_index])
        latency_ms = (time.perf_counter() - start) * 1000

        return RouteResult(
            route=route,
            latency_ms=latency_ms,
            confidence=confidence,
        )