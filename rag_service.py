import re
from typing import List, Dict, Tuple
import numpy as np
import faiss
import logging
from sagemaker_client import SageMakerEmbeddingClient

class RAGService:
    def __init__(self, embedding_client=None, debug: bool = True):

        self.embedding_client = embedding_client or SageMakerEmbeddingClient()
        self.vector_index = None
        self.chunks = []
        self.chunk_headers = []
        self.chunk_embeddings = None
        self.debug = debug

    def split_by_headers(self, text: str) -> List[Dict]:
        """
        Splits the document into chunks using Markdown-style headers as delimiters.
        Returns a list of dicts with 'header' and 'content'.
        """
        header_pattern = re.compile(r'^(#+)\s+(.*)', re.MULTILINE)
        matches = list(header_pattern.finditer(text))
        chunks = []
        
        if self.debug:
            print(f"[DEBUG] Found {len(matches)} headers in document")
        
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            header = match.group(2).strip()
            content = text[start:end].strip()
            if content:
                chunks.append({'header': header, 'content': content})
                
                if self.debug:
                    print(f"[DEBUG] Chunk {len(chunks)}: Header='{header}'")
                    print(f"         Content length: {len(content)} characters")
                    print(f"         Content preview: '{content[:150]}...'")
                    print("-" * 50)
        
        if self.debug:
            print(f"[DEBUG] Total chunks created: {len(chunks)}")
            
        return chunks

    def build_vector_store(self, document_text: str):
        """
        Splits the document, computes embeddings using SageMaker, and builds the FAISS index.
        """
        if self.debug:
            print("[DEBUG] Starting document processing...")
            print(f"[DEBUG] Document length: {len(document_text)} characters")
        
        self.chunks = self.split_by_headers(document_text)
        self.chunk_headers = [chunk['header'] for chunk in self.chunks]
        texts = [chunk['content'] for chunk in self.chunks]
        
        if self.debug:
            print(f"[DEBUG] Generating embeddings for {len(texts)} chunks using SageMaker...")
        
        # Generate embeddings using SageMaker embedding client
        embeddings_list = []
        for i, text in enumerate(texts):
            if self.debug:
                print(f"[DEBUG] Processing chunk {i+1}/{len(texts)}: '{self.chunk_headers[i]}'")
            
            embedding = self.embedding_client.invoke(text)
            if embedding is not None:
                embeddings_list.append(embedding)
                if self.debug:
                    print(f"         Successfully generated embedding (dim: {len(embedding)})")
            else:
                logging.warning(f"Failed to get embedding for chunk {i}: '{self.chunk_headers[i]}'")
                if self.debug:
                    print(f"         Failed to generate embedding")
        
        if not embeddings_list:
            raise ValueError("No embeddings were generated successfully")
        
        # Convert to numpy array
        self.chunk_embeddings = np.array(embeddings_list, dtype=np.float32)
        
        # Build FAISS index
        dim = self.chunk_embeddings.shape[1]
        
        if self.debug:
            print(f"[DEBUG] Embedding dimensions: {self.chunk_embeddings.shape}")
            print(f"[DEBUG] Building FAISS index with dimension: {dim}")
        
        self.vector_index = faiss.IndexFlatL2(dim)
        self.vector_index.add(self.chunk_embeddings)
        
        if self.debug:
            print(f"[DEBUG] FAISS index built successfully with {self.vector_index.ntotal} vectors")
            print(f"[DEBUG] All chunk headers: {self.chunk_headers}")

    def retrieve_relevant_chunks(self, top_k: int = 10) -> List[Dict]:
        """
        Uses chunk headers as the query. Retrieves the top_k most relevant chunks.
        """
        if self.vector_index is None:
            raise ValueError("Vector store not built. Call build_vector_store() first.")
        
        # Join all headers to form a single query string
        query = " ".join(self.chunk_headers)
        
        if self.debug:
            print(f"[DEBUG] RAG Query: '{query}'")
            print(f"[DEBUG] Searching for top {min(top_k, len(self.chunks))} most relevant chunks...")
        
        # Generate query embedding using SageMaker
        query_embedding = self.embedding_client.invoke(query)
        if query_embedding is None:
            logging.error("Failed to generate query embedding")
            return []
        
        # Ensure query embedding is in the right format for FAISS
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Adjust top_k if it's larger than available chunks
        actual_top_k = min(top_k, len(self.chunks))
        
        # Search for similar chunks
        D, I = self.vector_index.search(query_embedding, actual_top_k)
        
        selected_chunks = [self.chunks[idx] for idx in I[0]]
        
        if self.debug:
            print(f"[DEBUG] Retrieved {len(selected_chunks)} chunks:")
            print(f"[DEBUG] Similarity distances: {D[0]}")
            print(f"[DEBUG] Selected chunk indices: {I[0]}")
            
            for rank, (idx, chunk, distance) in enumerate(zip(I[0], selected_chunks, D[0])):
                print(f"[DEBUG] Rank {rank + 1}: Index {idx}, Distance {distance:.4f}")
                print(f"         Header: '{chunk['header']}'")
                print(f"         Content preview: '{chunk['content'][:100]}...'")
                print("-" * 40)
        
        return selected_chunks

    def get_context_for_llm(self, top_k: int = 10) -> str:
        """
        Returns the concatenated content of the top_k retrieved chunks.
        """
        relevant_chunks = self.retrieve_relevant_chunks(top_k=top_k)
        context = "\n\n".join([f"{chunk['header']}:\n{chunk['content']}" for chunk in relevant_chunks])
        
        if self.debug:
            print(f"[DEBUG] Final context length: {len(context)} characters")
            print(f"[DEBUG] Context preview: '{context[:200]}...'")
        
        return context

    def set_debug(self, debug: bool):
        """Toggle debug mode on/off"""
        self.debug = debug

    async def async_build_vector_store(self, document_text: str):
        """
        Asynchronous version of build_vector_store for better performance.
        """
        if self.debug:
            print("[DEBUG] Starting async document processing...")
            print(f"[DEBUG] Document length: {len(document_text)} characters")
        
        self.chunks = self.split_by_headers(document_text)
        self.chunk_headers = [chunk['header'] for chunk in self.chunks]
        texts = [chunk['content'] for chunk in self.chunks]
        
        if self.debug:
            print(f"[DEBUG] Generating embeddings for {len(texts)} chunks using SageMaker (async)...")
        
        # Generate embeddings asynchronously
        embeddings_list = []
        for i, text in enumerate(texts):
            if self.debug:
                print(f"[DEBUG] Processing chunk {i+1}/{len(texts)}: '{self.chunk_headers[i]}'")
            
            embedding = await self.embedding_client.async_invoke(text)
            if embedding is not None:
                embeddings_list.append(embedding)
                if self.debug:
                    print(f"         Successfully generated embedding (dim: {len(embedding)})")
            else:
                logging.warning(f"Failed to get embedding for chunk {i}: '{self.chunk_headers[i]}'")
                if self.debug:
                    print(f"         Failed to generate embedding")
        
        if not embeddings_list:
            raise ValueError("No embeddings were generated successfully")
        
        # Convert to numpy array
        self.chunk_embeddings = np.array(embeddings_list, dtype=np.float32)
        
        # Build FAISS index
        dim = self.chunk_embeddings.shape[1]
        
        if self.debug:
            print(f"[DEBUG] Embedding dimensions: {self.chunk_embeddings.shape}")
            print(f"[DEBUG] Building FAISS index with dimension: {dim}")
        
        self.vector_index = faiss.IndexFlatL2(dim)
        self.vector_index.add(self.chunk_embeddings)
        
        if self.debug:
            print(f"[DEBUG] FAISS index built successfully with {self.vector_index.ntotal} vectors")

    async def async_retrieve_relevant_chunks(self, top_k: int = 10) -> List[Dict]:
        """
        Asynchronous version of retrieve_relevant_chunks.
        """
        if self.vector_index is None:
            raise ValueError("Vector store not built. Call build_vector_store() first.")
        
        query = " ".join(self.chunk_headers)
        
        if self.debug:
            print(f"[DEBUG] RAG Query (async): '{query}'")
        
        # Generate query embedding asynchronously
        query_embedding = await self.embedding_client.async_invoke(query)
        if query_embedding is None:
            logging.error("Failed to generate query embedding")
            return []
        
        query_embedding = np.array([query_embedding], dtype=np.float32)
        actual_top_k = min(top_k, len(self.chunks))
        
        D, I = self.vector_index.search(query_embedding, actual_top_k)
        selected_chunks = [self.chunks[idx] for idx in I[0]]
        
        if self.debug:
            print(f"[DEBUG] Retrieved {len(selected_chunks)} chunks (async)")
        
        return selected_chunks

    async def async_get_context_for_llm(self, top_k: int = 10) -> str:
        """
        Asynchronous version of get_context_for_llm.
        """
        relevant_chunks = await self.async_retrieve_relevant_chunks(top_k=top_k)
        context = "\n\n".join([f"{chunk['header']}:\n{chunk['content']}" for chunk in relevant_chunks])
        
        if self.debug:
            print(f"[DEBUG] Final context length (async): {len(context)} characters")
        
        return context
