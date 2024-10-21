from typing import Literal, Annotated, Sequence, TypedDict
from typing_extensions import Annotated, TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from dotenv import load_dotenv
from rag.llm import Llms
from langchain.tools.retriever import create_retriever_tool
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = Llms(llm_model="gpt-4o-mini", embedings_model="text-embedding-3-large")
llm = model.openai()
embedding = model.openai_embeddings()
database = PineconeVectorStore.from_existing_index(
  index_name="vtw-pdf-openai", 
  embedding=embedding,
)
retriever = database.as_retriever()

# retriever_tool = create_retriever_tool(
#     retriever,
#     "retrieve_PineconeVectorStoreDB_content",
#     "Use to return information about PineconeVectorStoreDB.",
# )

# tools = [retriever_tool]

def question_router():
  class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "own-generating"] = Field(
      ...,
      description="Given a user question choose to route it to own-generating or a vectorstore.",
    )

  structured_llm_router = llm.with_structured_output(RouteQuery)

  system = """You are an expert at routing a user question to a vectorstore or not
  The vectorstore contains documents related to IT company VTW.
  If there are questions about 'VTW', '브띠따', '브이티더블유', Use the vectorstore for questions."""
  route_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system),
      ("human", "{question}"),
    ]
  )
  return route_prompt | structured_llm_router

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def agent(state):
  messages = state["messages"]
  model = llm.bind_tools(tools)
  response = model.invoke(messages)
  return {"messages": [response]}

def grade_documents(state) -> Literal["generate", "agent"]:
    # Data model for grading
    class Grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    llm_with_tool = openai.with_structured_output(Grade)
    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. 
        Here is the retrieved document:
        {context}
        Here is the user question: {question}
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    # Chain
    chain = prompt | llm_with_tool
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score
    if score == "yes":
        return "generate"
    else:
        return "agent"

def rewrite(state):
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" 
        Look at the input and try to reason about the underlying semantic intent / meaning.
        Here is the initial question:
        -------
        {question} 
        -------
        Formulate an improved question:""",
        )
    ]
    # LLM
    response = openai.invoke(msg)
    return {"messages": [response]}

def generate(state):
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    rag_chain = prompt | openai
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

def tools_condition(state) -> Literal["retrieve", "rewrite"]:
    messages = state["messages"]
    question = messages[0].content.lower()

    if "VTW" in question or "vtw" in question:
        return "retrieve"
    else:
        return "generate"

workflow = StateGraph(state_schema=State)

workflow.add_node("agent", agent)  # Agent node
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # Retrieval node
workflow.add_node("generate", generate)  # Generate node

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "generate": "generate", 
        "retrieve": "retrieve", 
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")
graph = workflow.compile()

def get_ai_response_with_graph(user_question):
  inputs = {
      "messages": [
          ("user", user_question),
      ]
  }
  config = {"configurable": {"thread_id": "abc345"}}
  output = graph.invoke(inputs, config)
  response = output["messages"][-1].content
  print(response)