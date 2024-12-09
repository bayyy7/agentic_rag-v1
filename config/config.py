from dataclasses import dataclass

@dataclass
class QNAModel:
   model: str = "gemini-1.5-pro"
   temperature: float = 0.8
   top_p: float = 0.6
   max_tokens: int = 2000

@dataclass
class Embedding:
   model: str = "intfloat/multilingual-e5-small"
   dimension: int = 384

@dataclass
class ChunkStrategy:
   chunk_size: int = 500
   chunk_overlap: int = 10

@dataclass
class VectorSearchStrateg:
   method: str = "similarity"
   k: int = 3

@dataclass
class Configuration:
   qna_model: QNAModel = QNAModel()
   embedding: Embedding = Embedding()
   chunk_strategy: ChunkStrategy = ChunkStrategy()
   vector_search_strategy: VectorSearchStrateg = VectorSearchStrateg()
