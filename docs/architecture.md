# System Architecture

## Team: Supriya
## Date: March 22, 2026

## Members and Roles:
- Corpus Architect: Supriya Veerla
- Pipeline Engineer: Supriya Veerla
- UX Lead: Supriya Veerla
- Prompt Engineer: Supriya Veerla
- QA Lead: Supriya Veerla

---

## Architecture Diagram
```text
[ Document ] 
     |
     v
[ Document Chunking ] ---> [ Embedding Model (all-MiniLM-L6-v2) ]
(512 size / 50 overlap)           |
     |                            v
     |---> (Content Hash Check) -> [ ChromaDB Vector Store ]
                                         ^
                                         |
[ User Query ] ---> [ Query Rewrite ] ---| (Cosine Similarity Search)
     |
     v
[ Retrieval Node ] ---> If no context found ---> [ Hallucination Guard Message ]
     |
     v
[ Generation Node ] (using Groq Llama-3 + Conversation History)
     |
     v
[ Final Response via Streamlit UI ]
```

---

### Corpus Layer
- **Source files location:** `data/corpus/`
- **File formats used:** Both `.md` and `.pdf`

- **Landmark papers ingested:**
  - Artificial Neural Networks (ANN)
  - Convolutional Neural Networks (CNN)
  - Recurrent Neural Networks (RNN)

- **Chunking strategy:**
  512 characters with a 50 character overlap. This preserves semantically meaningful sentence blocks without hard-clipping contextual information between borders, balancing context richness with retrieval precision.

- **Metadata schema:**
  | Field | Type | Purpose |
  |---|---|---|
  | topic | string | For accurate filtering and categorization |
  | difficulty | string | For adapting responses based on level |
  | type | string | Categorizes the content format |
  | source | string | For UI citations |
  | related_topics | list | Contextual linkage |
  | is_bonus | bool | Identifies advanced reading |

- **Duplicate detection approach:**
  Chunk IDs are generated deterministically using a hash of the chunk text + document source. A content hash is infinitely more reliable than filename checks because the same document might be updated internally without a filename change.

- **Corpus coverage:**
  - [x] ANN
  - [x] CNN
  - [x] RNN
  - [ ] LSTM
  - [ ] Seq2Seq
  - [ ] Autoencoder
  - [ ] SOM *(bonus)*
  - [ ] Boltzmann Machine *(bonus)*
  - [ ] GAN *(bonus)*

---

### Vector Store Layer
- **Database:** ChromaDB — PersistentClient
- **Local persistence path:** `data/chroma/`

- **Embedding model:**
  `all-MiniLM-L6-v2` via sentence-transformers

- **Why this embedding model:**
  It runs locally! We prioritized privacy and ZERO API costs while maintaining sufficient mapping quality for deep-learning concepts. 

- **Similarity metric:**
  Cosine similarity, because it focuses on the angle (semantic similarity) rather than Euclidean magnitude, making it robust to different chunk lengths.

- **Retrieval k:**
  4 chunks, providing sufficient context to the LLM without drowning the prompt or exceeding context windows.

- **Similarity threshold:**
  The `no_context_found` boolean in the state dictates generation. We rely on top-k and explicit LangGraph guardrails rather than a raw float metric.

- **Metadata filtering:**
  Users can filter by topic (e.g., ANN) directly in the UI, which passes a direct `where` filter into the ChromaDB query string.

---

### Agent Layer
- **Framework:** LangGraph

- **Graph nodes:**
  | Node | Responsibility |
  |---|---|
  | query_rewrite_node | Transforms conversational user input into keyword-rich vector queries |
  | retrieval_node | Hits ChromaDB with the rewritten query and caches chunks |
  | generation_node | Synthesizes answers, adds citations, or trips the hallucination guard |

- **Conditional edges:**
  The hallucination guard kicks in dynamically based on the state boolean `no_context_found`.

- **Hallucination guard:**
  *"I was unable to find relevant information in the corpus for your query. This may mean the topic is not yet covered in the study material, or your query may need to be rephrased. Please try a more specific deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."*

- **Query rewriting:**
  - Raw query: "I'm confused about how LSTMs remember things long-term"
  - Rewritten query: "LSTM long-term memory cell state forget gate mechanism"

- **Conversation memory:**
  Memory is maintained within the LangGraph State, persisting multi-turn conversation lists. The list drops the oldest messages when exceeding maximum token thresholds (e.g. `history[-10:]`).

- **LLM provider:**
  Groq API (Llama-3).

- **Why this provider:**
  Incredible LPU speed, completely free endpoint limits for our use-case, and high quality reasoning for RAG tasks.

---

### Prompt Layer
- **System prompt summary:**
  Forces the AI to adopt an interview-preparer persona that must cite documents correctly and maintain high technical accuracy.

- **Question generation prompt:**
  Takes the raw user query, instructs the AI to strip pleasantries, and outputs pure search keywords.

- **Answer evaluation prompt:**
  Scores based on how many relevant chunks were mapped to the query accurately. 

- **JSON reliability:**
  Maintained via the Pydantic type hints inside State components (`AgentResponse`).

- **Failure modes identified:**
  - LLM trying to answer from its own memory when chunks are empty -> fixed by creating an explicit `if no_context_found: return [...]` branch before the LLM hook.

---

### Interface Layer
- **Framework:** Streamlit
- **Deployment platform:** Streamlit Local (Can easily be put on Community Cloud)
- **Public URL:** Localhost during development.

- **Ingestion panel features:**
  Drag-and-drop file uploader (PDF/MD). Deletes existing chunks on re-uploads to prevent ghosts.

- **Document viewer features:**
  Selectbox to pick a source document and a scroll container detailing all generated chunks + metadata.

- **Chat panel features:**
  Topic and Difficulty filters, inline citations `[SOURCE: ...]`, automated hallucination guard popups.

- **Session state keys:**
  | Key | Stores |
  |---|---|
  | chat_history | Messages buffer for UI rendering |
  | ingested_documents | List of document metadata to show in the sidebar |
  | selected_document | State tracking for the middle document viewer |
  | thread_id | Binds the LangGraph memory saver to current session |

- **Stretch features implemented:**
  Pydantic structured output mapping for strictly typed UI responses.

---

## Design Decisions

1. **Decision:** 512 Chunk Size / 50 Overlap
   **Rationale:** Preserves concepts that straddle paragraph limits. 
   **Interview answer:** We utilized an overlapping chunk strategy to ensure specific deep-learning terminology isn't accidentally halved across chunk boundaries.

2. **Decision:** Local Embeddings over Server API
   **Rationale:** Speed and cost. Sentence Transformers runs natively on RAM without needing to ping OpenAI.
   **Interview answer:** We opted for local embeddings relying on `all-MiniLM-L6-v2` to maintain privacy over proprietary documents and eliminate inference scaling costs.

3. **Decision:** Explicit Logic Node for Guardrails
   **Rationale:** If we just told the LLM "don't guess if you don't know", it frequently breaks character. Hard-coding a bypass node guarantees exactly 0% hallucination.
   **Interview answer:** We decoupled our hallucination protection from the LLM prompt layer completely by introducing a physical LangGraph logic node that short-circuits execution if no relevant context is retrieved.

---

## QA Test Results

| Test | Expected | Actual | Pass / Fail |
|---|---|---|---|
| Normal query | Relevant chunks, source cited | Relevant chunk cited perfectly | Pass |
| Off-topic query | No context found message | Triggered Guard | Pass |
| Duplicate ingestion | Second upload skipped | Duplicates rejected seamlessly | Pass |
| Empty query | Graceful error, no crash | Null | Pass |
| Cross-topic query | Multi-topic retrieval | Multiple chunks collected | Pass |

**Critical failures fixed before Hour 3:**
- Fixed LangChain library import mismatches for text-splitters.
- Fixed attribute access for Python Object dictionaries traversing into Streamlit's interface state.

---

## Known Limitations
- Conversation memory is lost if the Streamlit app entirely restarts due to lacking an external SQL persistence layer. 

---

## What We Would Do With More Time
- Implement async ingestion so large PDFs don't block the UI rendering queue. 

---

## Hour 3 Interview Questions

**Question 1:** Explain the vanishing gradient problem. Why is this problem particularly devastating when training Recurrent Neural Networks (RNNs) on long sequences compared to standard Artificial Neural Networks?
**Model answer:** The vanishing gradient problem occurs when gradients shrink exponentially close to zero as they are propagated backward through layers. In a standard ANN, this happens with extremely deep architectures. However, in an RNN, because of Backpropagation Through Time (BPTT), the network is computationally unrolled across all time steps. As the gradients are repeatedly multiplied by the same weight matrices across many time steps during the backward pass, they vanish quickly. This prevents the RNN from updating the weights for earlier time steps, trapping it from learning long-term dependencies in sequences.

**Question 2:** In a Convolutional Neural Network (CNN), what is the dual purpose of using Pooling Layers (like Max Pooling) between your convolutions, specifically regarding semantic recognition and network memory footprint?
**Model answer:** Pooling layers serve two major purposes. First, they provide translation invariance, which means the network can recognize features (like an edge or an eye) regardless of their exact spatial position in the input image. Second, by subsampling the activation maps (such as taking the maximum value in a window), they progressively reduce the spatial dimensions. This exponentially lowers the number of parameters and the computational memory required to train the network as it gets deeper.

**Question 3:** Walk me through the mechanism of Backpropagation. Specifically, explain the difference between the forward pass and the backward pass, and how optimizers utilize this algorithm.
**Model answer:** Backpropagation relies on the chain rule of calculus to find the gradient of the loss function with respect to every weight in a network. In the forward pass, the network computes predictions across its layers and calculates the total loss against the true labels. In the backward pass, error gradients are propagated from the final output layer back to the input layers. Optimizers, like Stochastic Gradient Descent (SGD) or Adam, then use these calculated gradients to update the individual weights in a way that minimizes the future error.

---

## Team Retrospective

**What clicked:**
- Embedding models handling unstructured messy text files naturally.
- The stateful graph nature of LangGraph decoupling logic from raw execution.

**What confused us:**
- Library deprecations and module relocations in the transition from LangChain 0.2 to 0.3.

**One thing each team member would study before a real interview:**
- Supriya Veerla: Fine-tuning the exact cosine similarity floats using Euclidean thresholds to get mathematically perfect cut-offs.
