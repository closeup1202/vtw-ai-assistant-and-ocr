from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from rag.config import answer_examples

def get_contextualize_q_system_prompt():
  return (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
  )

def get_contextualize_q_prompt():
  return ChatPromptTemplate.from_messages(
    [
      ("system", get_contextualize_q_system_prompt()),
      MessagesPlaceholder("chat_history"),
      ("human", "{input}"),
    ]
  )

def get_few_shot_prompt(examples):
  example_prompt = ChatPromptTemplate.from_messages(
    [
      ("human", "{input}"),
      ("ai", "{answer}"),
    ]
  )

  return FewShotChatMessagePromptTemplate(
      example_prompt=example_prompt,
      examples=examples,
  )

def get_system_prompt():
   return (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
  )

def get_qa_prompt():
  return ChatPromptTemplate.from_messages(
    [
      ("system", get_system_prompt()), 
      get_few_shot_prompt(answer_examples),
      MessagesPlaceholder("chat_history"),
      ("human", "{input}"),
    ]
  )