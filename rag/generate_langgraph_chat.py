from dotenv import load_dotenv
from rag.llm import Llms

from typing import Literal
from pydantic import BaseModel, Field

from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough



load_dotenv()
model = Llms(llm_model="gpt-4o-mini", embedings_model="text-embedding-3-large")
llm = model.openai()
embedding = model.openai_embeddings()
database = PineconeVectorStore.from_existing_index(
  index_name="vtw-pdf-openai", 
  embedding=embedding,
)
retriever = database.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_PineconeVectorStoreDB_content",
    "Use to return information about PineconeVectorStoreDB.",
)

tools = [retriever_tool]

def question_router():
  class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "own"] = Field(
      ...,
      description="Given a user question choose to route it to own or a vectorstore.",
    )

  structured_llm_router = llm.with_structured_output(RouteQuery)

  system = """You are an expert at routing a user question to a vectorstore or not
  The vectorstore contains documents related to IT company VTW.
  If there are questions about 'VTW', '브띠따', '브이티더블유', Use the vectorstore for questions."""
  route_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system),
      ("human", "{messages}"),
    ]
  )
  return route_prompt | structured_llm_router

def vtw_expert():
  system = """You are an assistant for question-answering tasks. \n 
      Use the following pieces of retrieved context to answer the question \n
      If you don't know the answer, say that you don't know
      You don't have to say friendly guidance on previously identified misinformation
      Use three sentences maximum and keep the answer concise
      If the Korean word '문체비' is included, please change the word '문체비' to the Korean word '문화체련비'
      \n\n
      {context}
      """
  vtw_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system),
      ("human", "{messages}"),
    ]
  )

  def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)

  return (
    {"context": retriever | format_docs, "messages": RunnablePassthrough()}
    | vtw_prompt
    | llm
    | StrOutputParser()
  )

## Nodes
def retrieve(state):
    print("---RETRIEVER GENERATING---")
    question = state["messages"]
    response = vtw_expert().invoke(question[0].content)
    return {"messages": [response]}

def own(state):
    print("---OWN GENERATING---")
    question = state["messages"]
    response = llm.invoke(question)
    return {"messages": [response]}

## Edges
def route_question(state):
  print("---ROUTE QUESTION---")
  question = state["messages"]
  source = question_router().invoke({"messages": question})
  if source.datasource == "own":
    print("---ROUTE QUESTION TO OWN---")
    return "own"
  elif source.datasource == "vectorstore":
    print("---ROUTE QUESTION TO RAG---")
    return "vectorstore"


###########################################

# Set Graph
workflow = StateGraph(MessagesState)

# Define the nodes
workflow.add_node("retrieve", retrieve) #retrieve
workflow.add_node("own", own) #own


# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "own": "own",
        "vectorstore": "retrieve",
    },
)

memory = MemorySaver()
app = workflow.compile()
config = {"configurable": {"thread_id": "abc123"}}

def get_graph_response(user_question):
  inputs = {
    "messages": [
      ("user", user_question),
    ]
  }
  output = app.invoke(inputs, config)
  return output["messages"][-1].content