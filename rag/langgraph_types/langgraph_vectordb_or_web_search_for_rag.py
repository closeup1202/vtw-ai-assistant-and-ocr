### Model
from dotenv import load_dotenv
from rag.llm import Llms

from typing import Literal, List, TypedDict
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain import hub
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import START, END, StateGraph
from langchain.schema import Document


load_dotenv()
model = Llms(llm_model="gpt-4o-mini", embedings_model="text-embedding-3-large")
llm = model.openai()
embedding = model.openai_embeddings()
web_search_tool = TavilySearchResults(k=3)
database = PineconeVectorStore.from_existing_index(
  index_name="vtw-pdf-openai", 
  embedding=embedding,
)
retriever = database.as_retriever()

# 1. Router
def question_router():
  class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
      ...,
      description="Given a user question choose to route it to web search or a vectorstore.",
    )

  structured_llm_router = llm.with_structured_output(RouteQuery)

  system = """You are an expert at routing a user question to a vectorstore or web search.
  The vectorstore contains documents related to agents, prompt engineering, adversarial attacks and IT company VTW.
  Use the vectorstore for questions on these topics. Otherwise, use web-search."""
  route_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system),
      ("human", "{question}"),
    ]
  )
  return route_prompt | structured_llm_router

# 2. Retrieval Grader
def retrieval_grader():
  class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
      description="Documents are relevant to the question, 'yes' or 'no'"
    )

  structured_llm_grader = llm.with_structured_output(GradeDocuments)

  system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
      If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
      It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
      Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
  grade_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system),
      ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
  )
  return grade_prompt | structured_llm_grader

# 3. Hallucination Grader
def hallucination_grader():
  class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
      description="Answer is grounded in the facts, 'yes' or 'no'"
    )

  structured_llm_grader = llm.with_structured_output(GradeHallucinations)

  system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
      Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
  hallucination_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system),
      ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
  )
  return hallucination_prompt | structured_llm_grader

# 4. Answer Grader
def answer_grader():
  class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

  structured_llm_grader = llm.with_structured_output(GradeAnswer)

  system = """You are a grader assessing whether an answer addresses / resolves a question \n 
      Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
  answer_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system),
      ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
  )
  return answer_prompt | structured_llm_grader

# 5. Question Rewriter
def question_rewriter():
  system = """You a question re-writer that converts an input question to a better version that is optimized \n 
      for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
  re_write_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", system),
          (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
          ),
      ]
  )
  return re_write_prompt | llm | StrOutputParser()

###########

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for d in documents:
        score = retrieval_grader().invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter().invoke({"question": question})
    return {"documents": documents, "question": better_question}

def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([str(d["content"]) for d in docs])
    web_results = Document(page_content=web_results)
    return {"documents": web_results, "question": question}

### Edges ###

def route_question(state):
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router().invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    
def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    
def grade_generation_v_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader().invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader().invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
### Compile Graph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()

def get_graph_app_response(user_question):
  inputs = {
    "question": user_question
  }
  output = app.invoke(inputs)
  print(output)