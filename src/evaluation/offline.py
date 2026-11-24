"""Deterministic test embedding. Lexical hashing, NOT a semantic model benchmark."""
import hashlib
import math
from langchain_core.embeddings import Embeddings
from src.retrieval.retriever import tokenize


class HashEmbeddings(Embeddings):
    dimensions = 512

    def embed_query(self, text):
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % self.dimensions
            vector[index] += 1
        norm = math.sqrt(sum(x*x for x in vector)) or 1
        return [x/norm for x in vector]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]
