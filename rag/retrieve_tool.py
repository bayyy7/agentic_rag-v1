from langchain_community.vectorstores import FAISS
from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class RetrieverInput(BaseModel):
   query: str = Field(description="Query User untuk mencari informasi pada dokumen")

class Retrieve(BaseTool):
   name: str = "document_retrieve_tool"
   description: str = "Gunakan tools ini untuk mencari informasi berkaitan dengan Akuntansi, Finansial, dan Perbankan"
   response_format: str = "content_and_artifact"
   args_schema: Type[BaseModel] = RetrieverInput
   vector_store: FAISS
   search_type: str
   k: int

   def _run(self, query: str) -> dict[str, any]:
      retrieve_docs = self.vector_store.search(query=query, search_type=self.search_type, k=self.k)
      serialized = "\n\n".join(
         (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") 
         for doc in retrieve_docs
      )
      return serialized, retrieve_docs