from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

# upstage = none, "solar-embedding-1-large" (4096)
# openai = "gpt-4o', "text-embedding-3-large" (3072)

class Llms:
  
  def __new__(cls, *args, **kwargs):
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
    return ChatOpenAI(model=self.llm_model)

  def openai_embeddings(self):
    return OpenAIEmbeddings(model=self.embedings_model)

  def ollama(model):
    return ChatOllama(
      model = model,
      temperature = 0.8,
      num_predict = 256,
  )

  def ollama_embeddings(model):
    return OllamaEmbeddings(model=model, base_url=None)