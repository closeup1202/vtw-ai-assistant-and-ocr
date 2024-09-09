from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

cache_store = LocalFileStore("./cache/")
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# upstage = none, "solar-embedding-1-large" (4096)
# openai = "gpt-4o', "text-embedding-3-large" (3072)

class Llms:
  
  def __new__(cls, *arg, **kargs):
    if not hasattr(cls, "_instance"):         
      cls._instance = super().__new__(cls)  
    return cls._instance                      

  def __init__(self, llm_model, embedings_model = None):
    cls = type(self)
    if not hasattr(cls, "_init"):             
      self.llm_model = llm_model
      self.embedings_model = embedings_model
      cls._init = True

  def upstage(self):
    return ChatUpstage(model=self.llm_model)

  def upstage_embedding(self):
    return UpstageEmbeddings(model=self.embedings_model)

  def openai(self):
    return ChatOpenAI(
      model=self.llm_model,
      temperature= 0.5
    )

  def openai_embeddings_cached(self):
    embedding = OpenAIEmbeddings(model=self.embedings_model)
    return CacheBackedEmbeddings.from_bytes_store(
      embedding, cache_store, namespace=embedding.model
  )

  def openai_embeddings(self):
    return OpenAIEmbeddings(model=self.embedings_model)

  def ollama(self):
    return ChatOllama(
      model = self.llm_model,
      temperature = 0.8,
      num_predict = 256,
  )

  def ollama_embeddings(self):
    return OllamaEmbeddings(model=self.embedings_model, base_url=None)