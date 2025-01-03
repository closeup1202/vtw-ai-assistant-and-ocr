from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from rag.history.prompt import get_retriever_chain
from rag.llm import Llms

load_dotenv()
store = {}
llm = Llms(llm_model="gpt-4o-mini", embedings_model="text-embedding-3-large")
embeding = llm.openai_embeddings()
openai = llm.openai()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
  if session_id not in store:
    store[session_id] = ChatMessageHistory()
  return store[session_id]

def get_retriever(index, embedding):
  database = PineconeVectorStore.from_existing_index(
    index_name=index, 
    embedding=embedding,
  )
  return database.as_retriever()

def get_rag_chain(llm, retriever):
  retriever_chain = get_retriever_chain(llm, retriever)
  return RunnableWithMessageHistory(
    retriever_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
  )

def get_ai_response(user_question):
  retriever = get_retriever("vtw-pdf-openai", embeding)
  rag_chain = get_rag_chain(llm=openai, retriever=retriever)
  result = rag_chain.invoke(
    {"input": user_question}, 
    config={"configurable": {"session_id": "abc123"}},
  )['answer']
  return result