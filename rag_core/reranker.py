from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[str]) -> list[str]:
        pairs = [[query, c] for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked]
