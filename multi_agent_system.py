"""
AAIDC Module 2 Project: Multi-Agent Publication Assistant
Agents:
1. RepoAnalyzer - Reads and understands the GitHub repo
2. MetadataRecommender - Suggests tags, title, summary
3. Reviewer - Checks for missing sections or improvements
Tools: Web Search, Math, RAG
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# Load environment variables
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "fake_key")

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Tools
search = TavilySearchResults(max_results=2)
math_tool = Tool(
    name="Calculator",
    func=lambda x: eval(x),
    description="Useful for math calculations"
)

# Simulated RAG (for README content)
def create_rag_tool(repo_content: str):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts([repo_content], embeddings)
    retriever = vectorstore.as_retriever()
    return Tool(
        name="RepoRAG",
        func=lambda q: retriever.get_relevant_documents(q)[0].page_content,
        description="Retrieve info from the GitHub repo README"
    )

# Agent State
class AgentState(TypedDict):
    messages: List
    repo_url: str
    repo_content: str
    final_report: str

# Agent Functions
def repo_analyzer_node(state: AgentState):
    content = "Simulated README content for " + state["repo_url"]
    state["repo_content"] = content
    return state

def metadata_recommender_node(state: AgentState):
    prompt = f"""
    You are a Metadata Recommender Agent.
    Analyze this repo content and suggest:
    - A better title (max 6 words)
    - 3 relevant tags
    - A 1-sentence summary
    
    Repo: {state['repo_content']}
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    state["messages"].append(HumanMessage(content=f"Metadata Recommender: {response.content}"))
    return state

def reviewer_node(state: AgentState):
    prompt = f"""
    You are a Reviewer Agent.
    Check the README for:
    - Missing sections (like Installation, Usage, License)
    - Clarity issues
    - Suggest 2 improvements
    
    Repo: {state['repo_content']}
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    state["messages"].append(HumanMessage(content=f"Reviewer: {response.content}"))
    return state

def final_report_node(state: AgentState):
    report = "\n".join([msg.content for msg in state["messages"]])
    state["final_report"] = report
    return state

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("analyzer", repo_analyzer_node)
workflow.add_node("metadata", metadata_recommender_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("report", final_report_node)

workflow.set_entry_point("analyzer")
workflow.add_edge("analyzer", "metadata")
workflow.add_edge("metadata", "reviewer")
workflow.add_edge("reviewer", "report")
workflow.add_edge("report", END)

app = workflow.compile()

# Run Example
if __name__ == "__main__":
    print("🚀 AAIDC Module 2: Multi-Agent Publication Assistant")
    repo_url = "https://github.com/R786P/aaidc-module1-rag"
    
    inputs = {
        "messages": [],
        "repo_url": repo_url,
        "repo_content": "",
        "final_report": ""
    }
    
    result = app.invoke(inputs)
    print("\n✅ FINAL REPORT:\n")
    print(result["final_report"])
