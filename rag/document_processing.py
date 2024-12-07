from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import yaml

with open("config/config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

def processing_documents(document, chunk_size: int, chunk_overlap: int, embedding_model: str, dimension: int) -> FAISS:
   loader = PyPDFLoader(file_path=document)
   docs = loader.load()
   text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size, chunk_overlap=chunk_overlap
   )
   chunks = text_splitter.split_documents(docs)
   embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
   vector_store = FAISS(
      embedding_function=embeddings,
      index=faiss.IndexFlatL2(dimension),
      docstore=InMemoryDocstore(),
      index_to_docstore_id={}
   )
   vector_store.add_documents(chunks)
   return vector_store