from __future__ import annotations
from typing import Optional
import numpy as np
from prototype import config
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: Optional[str] = None, batch_size: Optional[int] = None):
        self.model_name = model_name or config.KB_EMBED_MODEL
        self.device = "cuda"
        self.batch_size = batch_size or config.KB_EMBED_BATCH
        self.query_prefix = f"Instruct: {config.KB_QUERY_INSTRUCTION}\nQuery: "

        print(f"[embeddings] loading {self.model_name} on {self.device}")
        st_kwargs = {"device": self.device, "trust_remote_code": True}
        st_kwargs["model_kwargs"] = {"torch_dtype": "auto"}
        self.model = SentenceTransformer(self.model_name, **st_kwargs)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 64,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    def encode_query(self, text: str) -> np.ndarray:
        out = self.model.encode(
            [self.query_prefix + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(out[0], dtype=np.float32)
