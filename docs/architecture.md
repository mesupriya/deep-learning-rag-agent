# Architecture Document

## Team Roles
- Corpus Architect: Managed
- Pipeline Engineer: Managed
- UX Lead: Managed
- Prompt Engineer: Managed
- QA Lead: Managed

## Pipeline Setup
- We use a 3-node LangGraph setup: `query_rewrite_node`, `retrieval_node`, and `generation_node`.
- The graph starts with Query Rewriting, improving retrieval by making queries more semantically aligned with the corpus using LLM processing.
- After rewriting, it invokes the Retrieval Node using `VectorStoreManager` which queries ChromaDB using `cosine` distance.
- Based on the retrieval result, the `no_context_found` flag dynamically informs the Generation Node whether to deploy the hallucination guard.
- Overlaps during chunking (Chunk Size: 512, Overlap: 50) ensure no concepts are hard-clipped at chunk borders.

## UI Choices
- Streamlit application using `st.session_state` to decouple memory from page reloads.
- Left Panel: Ingets PDFs and Markdown files.
- Middle Panel: A document viewer that lists contents and chunks associated with specific document topics.
- Right Panel: A conversational RAG chat that retrieves documents through the pipeline.
