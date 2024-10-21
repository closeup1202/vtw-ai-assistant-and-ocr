from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# upstage = none, "solar-embedding-1-large" (4096)
# openai = "gpt-4o-mini', "text-embedding-3-large" (3072)

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

  def upstage_embeddings(self):
    return UpstageEmbeddings(model=self.embedings_model)

  def openai(self):
    return ChatOpenAI(
      model=self.llm_model,
      temperature= 0.5,
    )

  def openai_embeddings(self):
    return OpenAIEmbeddings(model=self.embedings_model)
