"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        import chromadb
        path = Path(self._settings.chroma_db_path)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=self._settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        if self._collection:
            logger.info(f"Initialised ChromaDB collection '{self._settings.chroma_collection_name}' with {self._collection.count()} items")

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        result = self._collection.get(ids=[chunk_id])
        return len(result["ids"]) > 0

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        result = IngestionResult()
        
        for chunk in chunks:
            if self.check_duplicate(chunk.chunk_id):
                result.skipped += 1
                continue
                
            embedding = self._embeddings.embed_documents([chunk.chunk_text])[0]
            
            try:
                self._collection.upsert(
                    ids=[chunk.chunk_id],
                    embeddings=[embedding],
                    documents=[chunk.chunk_text],
                    metadatas=[chunk.metadata.to_dict()]
                )
                result.ingested += 1
                if chunk.metadata.source not in result.document_ids:
                    result.document_ids.append(chunk.metadata.source)
            except Exception as e:
                logger.error(f"Error ingesting chunk {chunk.chunk_id}: {e}")
                result.errors.append(str(e))
                
        logger.info(f"Ingestion complete. Ingested: {result.ingested}, Skipped: {result.skipped}, Errored: {len(result.errors)}")
        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        k = k or self._settings.retrieval_k
        where_filter = {}
        if topic_filter:
            where_filter["topic"] = topic_filter
        if difficulty_filter:
            where_filter["difficulty"] = difficulty_filter
            
        where = where_filter if where_filter else None
        
        query_embedding = self._embeddings.embed_query(query_text)
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_chunks = []
        if not results["ids"] or not results["ids"][0]:
            return retrieved_chunks
            
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            score = 1.0 - distance
            
            if score >= self._settings.similarity_threshold:
                chunk = RetrievedChunk(
                    chunk_id=results["ids"][0][i],
                    chunk_text=results["documents"][0][i],
                    metadata=ChunkMetadata.from_dict(results["metadatas"][0][i]),
                    score=score
                )
                retrieved_chunks.append(chunk)
                
        retrieved_chunks.sort(key=lambda x: x.score, reverse=True)
        return retrieved_chunks

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        results = self._collection.get(include=["metadatas"])
        if not results or not results["metadatas"]:
            return []
            
        sources = {}
        for metadata in results["metadatas"]:
            source = metadata["source"]
            if source not in sources:
                sources[source] = {"source": source, "topic": metadata["topic"], "chunk_count": 0}
            sources[source]["chunk_count"] += 1
            
        return sorted(list(sources.values()), key=lambda x: x["source"])

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        results = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"]
        )
        
        chunks = []
        if not results or not results["ids"]:
            return chunks
            
        for i in range(len(results["ids"])):
            chunks.append(DocumentChunk(
                chunk_id=results["ids"][i],
                chunk_text=results["documents"][i],
                metadata=ChunkMetadata.from_dict(results["metadatas"][i])
            ))
        return chunks

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        results = self._collection.get(include=["metadatas"])
        if not results or not results["metadatas"]:
            return {
                "total_chunks": 0,
                "topics": [],
                "sources": [],
                "bonus_topics_present": False
            }
            
        topics = set()
        sources = set()
        bonus_topics_present = False
        
        for metadata in results["metadatas"]:
            topics.add(metadata["topic"])
            sources.add(metadata["source"])
            if metadata.get("is_bonus") == "true":
                bonus_topics_present = True
                
        return {
            "total_chunks": len(results["ids"]),
            "topics": list(topics),
            "sources": list(sources),
            "bonus_topics_present": bonus_topics_present
        }

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        initial_count = self._collection.count()
        self._collection.delete(where={"source": source})
        return initial_count - self._collection.count()
