from dotenv import load_dotenv 
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from rag.rag_prompt import get_contextualize_q_prompt, get_qa_prompt

load_dotenv()
store = {}

def get_llm():
  llm = ChatUpstage()
  return llm

def get_embedding(model):
  return UpstageEmbeddings(model=model)

def get_retriever(index):
  embedding = get_embedding("solar-embedding-1-large");
  database = PineconeVectorStore.from_existing_index(index_name=index, embedding=embedding)
  return database.as_retriever()

def get_history_retriever(llm, retriever):
  return create_history_aware_retriever(llm, retriever, get_contextualize_q_prompt)

def get_rag_chain(llm, retriever):
  history_aware_retriever = get_history_retriever(llm, retriever)
  question_answer_chain = create_stuff_documents_chain(llm, get_qa_prompt)
  rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

  conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
  ).pick("answer")

  return conversational_rag_chain

def get_session_history(session_id: str) -> BaseChatMessageHistory:
  if session_id not in store:
    store[session_id] = ChatMessageHistory()
  return store[session_id]

def get_ai_response(user_question):
  retriever = get_retriever("vtw")
  llm = get_llm()
  rag_chain = get_rag_chain(llm, retriever=retriever)
  ai_response = rag_chain.stream({"input": user_question}, config={"configurable": {"session_id": "abc123"}},)
  return ai_response