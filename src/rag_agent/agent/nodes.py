"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages

from rag_agent.agent.prompts import (
    QUESTION_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import AgentResponse, AgentState, RetrievedChunk
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Natural language questions are often poorly suited for vector
    similarity search. This node rephrases the query into a form
    that produces better embedding matches against the corpus.

    Example
    -------
    Input:  "I'm confused about how LSTMs remember things long-term"
    Output: "LSTM long-term memory cell state forget gate mechanism"

    Interview talking point: query rewriting is a production RAG pattern
    that significantly improves retrieval recall. It acknowledges that
    users do not phrase queries the way documents are written.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: messages (for context).

    Returns
    -------
    dict
        Updates: original_query, rewritten_query.
    """
    if not state["messages"]:
        return {"original_query": "", "rewritten_query": ""}
        
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_message = msg
                break
                
    original_query = last_message.content
    
    try:
        settings = get_settings()
        llm = LLMFactory(settings).create()
        prompt = (
            "You are an AI assistant that optimizes search queries. "
            "Rewrite the user's question into keywords for vector search. "
            "Return ONLY the rewritten query with no preamble.\n\n"
            f"User Query: {original_query}"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        rewritten_query = str(response.content).strip()
        return {"original_query": original_query, "rewritten_query": rewritten_query}
    except Exception:
        return {"original_query": original_query, "rewritten_query": original_query}


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.

    Sets the no_context_found flag if no chunks meet the similarity
    threshold. This flag is checked by generation_node to trigger
    the hallucination guard.

    Interview talking point: separating retrieval into its own node
    makes it independently testable and replaceable — you could swap
    ChromaDB for Pinecone or Weaviate by changing only this node.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: rewritten_query, topic_filter, difficulty_filter.

    Returns
    -------
    dict
        Updates: retrieved_chunks, no_context_found.
    """
    manager = VectorStoreManager(get_settings())
    chunks = manager.query(
        query_text=state.get("rewritten_query", ""),
        topic_filter=state.get("topic_filter"),
        difficulty_filter=state.get("difficulty_filter")
    )
    
    if not chunks:
        return {"retrieved_chunks": [], "no_context_found": True}
        
    return {"retrieved_chunks": chunks, "no_context_found": False}


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.

    Implements the hallucination guard: if no_context_found is True,
    returns a clear "no relevant context" message rather than allowing
    the LLM to answer from parametric memory.

    Implements token-aware conversation memory trimming: when the
    message history approaches max_context_tokens, the oldest
    non-system messages are removed.

    Interview talking point: the hallucination guard is the most
    commonly asked about production RAG pattern. Interviewers want
    to know how you prevent the model from confidently making up
    information when the retrieval step finds nothing relevant.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: retrieved_chunks, no_context_found, messages,
               original_query, topic_filter.

    Returns
    -------
    dict
        Updates: final_response, messages (with new AIMessage appended).
    """
    settings = get_settings()
    llm = LLMFactory(settings).create()

    # ---- Hallucination Guard -----------------------------------------------
    if state.no_context_found:
        no_context_message = (
            "I was unable to find relevant information in the corpus for your query. "
            "This may mean the topic is not yet covered in the study material, or "
            "your query may need to be rephrased. Please try a more specific "
            "deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."
        )
        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=state.rewritten_query,
        )
        return {
            "final_response": response,
            "messages": [AIMessage(content=no_context_message)],
        }

    context_parts = []
    sources = set()
    total_score = 0.0
    
    for chunk in state.get("retrieved_chunks", []):
        citation = chunk.to_citation()
        sources.add(citation)
        total_score += chunk.score
        context_parts.append(f"{citation}\n{chunk.chunk_text}\n")
        
    context_str = "\n".join(context_parts)
    confidence = total_score / len(state["retrieved_chunks"]) if state.get("retrieved_chunks") else 0.0
    
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    context_msg = HumanMessage(content=f"Available Context from Corpus:\n{context_str}")
    
    history = state.get("messages", [])[:-1]
    current_query = state.get("messages", [])[-1]
    
    if len(history) > 10:
        history = history[-10:]
        
    messages_for_llm = [system_msg, context_msg] + history + [current_query]
    
    llm_response = llm.invoke(messages_for_llm)
    answer = str(llm_response.content)
    
    response = AgentResponse(
        answer=answer,
        sources=list(sources),
        confidence=confidence,
        no_context_found=False,
        rewritten_query=state.get("rewritten_query", "")
    )
    
    return {
        "final_response": response,
        "messages": [AIMessage(content=answer)]
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge function: decide whether to retry retrieval or generate.

    Called by the graph after retrieval_node. If no context was found,
    the graph routes back to query_rewrite_node for one retry with a
    broader query before triggering the hallucination guard.

    Interview talking point: conditional edges in LangGraph enable
    agentic behaviour — the graph makes decisions about its own
    execution path rather than following a fixed sequence.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: no_context_found, retrieved_chunks.

    Returns
    -------
    str
        "generate" — proceed to generation_node.
        "end"      — skip generation, return no_context response directly.

    Notes
    -----
    Retry logic should be limited to one attempt to prevent infinite loops.
    Track retry count in AgentState if implementing retry behaviour.
    """
    return "generate"
