from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from rag.config import answer_examples

def get_history_retriever(llm, retriever):
  contextualize_q_system_prompt = (
      "Given a chat history and the latest user question "
      "which might reference context in the chat history, "
      "formulate a standalone question which can be understood "
      "without the chat history. Do NOT answer the question, "
      "just reformulate it if needed and otherwise return it as is."
  )

  contextualize_q_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", contextualize_q_system_prompt),
          MessagesPlaceholder("chat_history"),
          ("human", "{input}"),
      ]
  )

  return create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
  )

def get_dictionary_chain(llm):
  dictionary = ["브띠따 -> VTW"]
  prompt = ChatPromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해 주세요.
    만약 변경할 필요가 없다고 판단되면, 사용자의 질문을 변경하지 않아도 됩니다.
    사전 : {dictionary}

    질문 : {{question}}
  """)

  return prompt | llm | StrOutputParser()

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

def get_question_answer_chain(llm):
  few_shot_prompt = get_few_shot_prompt(answer_examples)

  system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
  )

  qa_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", system_prompt), 
          few_shot_prompt,
          MessagesPlaceholder("chat_history"),
          ("human", "{input}"),
      ]
  )

  return create_stuff_documents_chain(llm, qa_prompt)

def get_retriever_chain(llm, retriever):
  history_aware_retriever = get_history_retriever(llm, retriever)
  question_answer_chain = get_question_answer_chain(llm)
  return create_retrieval_chain(history_aware_retriever, question_answer_chain)