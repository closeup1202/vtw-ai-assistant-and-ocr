from dotenv import load_dotenv 
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

load_dotenv()
store = {}

def get_llm():
  llm = ChatUpstage()
  return llm

def get_embedding(model):
  return UpstageEmbeddings(model=model)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_ai_response(user_question):
  return "안녕하세요"