import os
import json
from uuid import uuid4
from typing import Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_nvidia import NVIDIAEmbeddings
from langchain_postgres import PGVector
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field


class Gatekeeper(BaseModel):
    is_relevant: bool = Field(
        description="Indicates whether the input question is relevant about getting to know about the person, their experience, and their projects."
    )
    query: Optional[str] = Field(
        description="If relevant, a refined version of the question focused on the person's experience and projects."
    )


class PortfolioState(TypedDict):
    question: str
    is_relevant: bool
    query: Optional[str]
    retrieved_context: Optional[list[dict]]
    final_answer: str


def build_graph():
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_EMBEDDINGS_MODEL = os.getenv("OPENAI_EMBEDDINGS_MODEL")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    embeddings = NVIDIAEmbeddings(
        model=OPENAI_EMBEDDINGS_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    connection = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="my_docs",
        connection=connection,
        use_jsonb=True,
    )

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.5,
    )

    gatekeeper = llm.with_structured_output(Gatekeeper)

    def gatekeeper_node(state: PortfolioState):
        result = gatekeeper.invoke(
            [
                SystemMessage(
                    content="You are a gatekeeper for a personal portfolio assistant. Only generate a relevant query when the question is about past experience, projects, or skills of the person. If the question is not relevant, return False and None. If the question is relevant but not about past experience, projects, or skills, return True and None. If the question is relevant, generate a refined version of the question that is more specific and focused on the person's experience and projects."
                ),
                HumanMessage(
                    content=f"Decide whether the following question is relevant about getting to know the person and their portfolio: {state['question']}."
                ),
            ]
        )

        if not result.is_relevant:
            state["is_relevant"] = False
            state["final_answer"] = (
                "I'm sorry, but your question is not relevant to getting to know me and my portfolio. "
                "Please ask a question that is related to my experience, projects, or skills."
            )
        else:
            state["is_relevant"] = True
            state["query"] = result.query

        return state

    def retrieval_node(state: PortfolioState):
        results = vector_store.similarity_search(state["query"], k=3)
        state["retrieved_context"] = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in results
        ]
        return state

    def saujana_node(state: PortfolioState):
        context = "\n\n".join(
            [
                f"{doc['page_content']} [metadata: {doc['metadata']}]"
                for doc in state["retrieved_context"]
            ]
        )

        final_answer = llm.invoke(
            [
                SystemMessage(
                    content=f"""
You are a personal portfolio assistant that represents "Saujana Shafi".

Give answer from the perspective of "Saujana Shafi" and provide a concise response to the question asked, using the context provided from the retrieved documents, no need to be too detailed, just make it more human-like.

Context: {context}
"""
                ),
                AIMessage(
                    content="""Hi, I'm Saujana - a Software Engineer from Indonesia.

I'm currently exploring my potential at Apple Developer Academy @ UCS. I graduated with Electrical Engineering from Universitas Jenderal Soedirman. I strongly believe that understanding how things work is important, without it success is often just luck.

I started building at 14, beginning with an Arduino ultrasonic ruler. Since then, I've continued exploring different domains and technologies.

My core areas include:
- Web Development (FastAPI, React)
- Mobile Development (Swift, React Native)
- Embedded Systems (Arduino, IoT)

I'm passionate about technology that solves small things in our daily lives that others overlooked. It actually started when I got my first iPhone and when I change my background it automatically split my picture into two layer for aesthetic lock screen. Since then, I've been deeply interested in on-device AI/ML technology, especially within the Apple Ecosystem.

Beyond software, I'm particularly interested in Smart Home solutions. Because not only lives on the Software Realms, but also able to interact with our real worlds and helps us on physical daily tasks. Also, I LOVE beautiful interior design & architecture.

On the side I'm a Chess addict (1400 ELO @ chess.com), Goddess Rockstar, and die-hard Timothy Ronald fans.

I'm always open to learning, building, and collaborating on meaningful projects."""
                ),
                HumanMessage(content=f"{state['question']}"),
            ]
        )

        state["final_answer"] = final_answer.content
        return state

    workflow = StateGraph(PortfolioState)

    workflow.add_node("gatekeeper", gatekeeper_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("saujana", saujana_node)

    workflow.set_entry_point("gatekeeper")

    def route_from_gatekeeper(state: PortfolioState):
        if not state["is_relevant"]:
            return "end"
        elif state["query"]:
            return "retrieval"
        else:
            return "saujana"

    workflow.add_conditional_edges(
        "gatekeeper",
        route_from_gatekeeper,
        {
            "end": END,
            "retrieval": "retrieval",
            "saujana": "saujana",
        },
    )

    workflow.add_edge("retrieval", "saujana")
    workflow.add_edge("saujana", END)

    return workflow.compile()
