from dotenv import load_dotenv
from rag.llm import Llms
from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, StateGraph, END
from langchain_core.runnables import RunnableConfig, RunnableWithMessageHistory
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory

set_llm_cache(InMemoryCache())

load_dotenv()
model = Llms(llm_model="gpt-4o-mini", embedings_model="text-embedding-3-large")
llm = model.openai()
embedding = model.openai_embeddings()
database = PineconeVectorStore.from_existing_index(
  index_name="vtw-pdf-openai", 
  embedding=embedding,
)
retriever = database.as_retriever()
store = {} 

def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in store:
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]


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
  retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system),
      ("human", "{input}"),
    ]
  )
  combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
  return create_retrieval_chain(retriever, combine_docs_chain)

# def answer_with_session():
#   contextualize_q_system_prompt = (
#         "Given a chat history and the latest user question "
#         "which might reference context in the chat history, "
#         "formulate a standalone question which can be understood "
#         "without the chat history. Do NOT answer the question, "
#         "just reformulate it if needed and otherwise return it as is."
#     )

#   contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#       ("system", contextualize_q_system_prompt),
#       MessagesPlaceholder("chat_history"),
#       ("human", "{input}"),
#     ]
#   )

#   runnable = contextualize_q_prompt | llm

#   return RunnableWithMessageHistory(
#     runnable,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
#   )

## Nodes
def retrieve(state, config: RunnableConfig):
  print("---RETRIEVER GENERATING---")
  question = state["messages"]
  response = vtw_expert().invoke({"input": question})
  source = response['context'][0].metadata['source']
  cleaned_source = source.replace('pdf\\', '출처: ')
  return {"answer": response['answer'], "source": cleaned_source}

def own(state, config: RunnableConfig):
  print("---OWN GENERATING---")
  question = state["messages"]
  response = llm.invoke(question)
  return {"answer": response.content}

## Edges
def route_question(state, config: RunnableConfig):
  print("---ROUTE QUESTION---")
  question = state["messages"]
  source = question_router().invoke({"messages": question})
  if source.datasource == "own":
    print("---ROUTE QUESTION TO OWN---")
    return "own"
  elif source.datasource == "vectorstore":
    print("---ROUTE QUESTION TO RAG---")
    return "vectorstore"

class GraphState(TypedDict):
  messages: str
  answer: str
  source: str

# Set Graph
workflow = StateGraph(GraphState)

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
app = workflow.compile()

def get_graph_response(user_question):
  inputs = { "messages": user_question }
  output = app.invoke(inputs)
  print(output)
  return output