from sentence_transformers import SentenceTransformer
from raptor.EmbeddingModels import BaseEmbeddingModel


class MiniLMEmbedding(BaseEmbeddingModel):

    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def create_embedding(self, text):
        return self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()