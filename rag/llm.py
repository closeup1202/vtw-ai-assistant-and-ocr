from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from rag.rag_prompt import get_retriever_chain

load_dotenv()
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
  if session_id not in store:
    store[session_id] = ChatMessageHistory()
  return store[session_id]

def get_llm(model):
  return ChatOpenAI(model = model)

def get_embedding(model):
  return OpenAIEmbeddings(model=model)

def get_retriever(index):
  embedding = get_embedding("text-embedding-3-large")
  database = PineconeVectorStore.from_existing_index(index_name=index, embedding=embedding)
  return database.as_retriever()

def get_rag_chain(llm, retriever):
  retriever_chain = get_retriever_chain(llm, retriever)
  return RunnableWithMessageHistory(
    retriever_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
  ).pick("answer")

def get_ai_response(user_question):
  retriever = get_retriever("vtw")
  llm = get_llm("gpt-4o-mini")
  rag_chain = get_rag_chain(llm, retriever=retriever)
  return rag_chain.stream({"input": user_question}, config={"configurable": {"session_id": "abc123"}},)