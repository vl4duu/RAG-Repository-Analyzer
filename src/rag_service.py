import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .github_parser import get_repo_files, chunk_repository_files
from .embedding import embed_textual_metadata, generate_code_embedding
from .chromaDB_setup import setup_chroma_collections
import openai


logger = logging.getLogger(__name__)


class RAGService:
    """Service class that encapsulates all RAG functionality"""
    
    def __init__(self):
        self.current_repository: Optional[str] = None
        self.collections: Optional[Dict] = None
        self.is_ready: bool = False
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def analyze_repository(self, repo_path: str) -> Dict[str, Any]:
        """
        Analyze a GitHub repository by fetching files, generating embeddings, and setting up ChromaDB collections.
        
        Args:
            repo_path: Repository path in format 'username/repo_name'
            
        Returns:
            Dictionary with status and message
        """
        try:
            logger.info(f"Starting analysis of repository: {repo_path}")
            
            # Reset state
            self.is_ready = False
            self.collections = None
            
            # Step 1: Fetch repository files
            logger.info("Fetching repository files...")
            repo_files = await asyncio.get_event_loop().run_in_executor(
                self.executor, get_repo_files, repo_path
            )
            
            if not repo_files:
                raise ValueError(f"No files found in repository {repo_path}")
            
            # Step 2: Chunk repository files
            logger.info("Chunking repository files...")
            chunked_docs = await asyncio.get_event_loop().run_in_executor(
                self.executor, chunk_repository_files, repo_files
            )
            
            # Step 3: Generate embeddings
            logger.info("Generating embeddings...")
            embedded_chunks = await self._generate_embeddings(chunked_docs)
            
            # Step 4: Setup ChromaDB collections
            logger.info("Setting up ChromaDB collections...")
            self.collections = await asyncio.get_event_loop().run_in_executor(
                self.executor, setup_chroma_collections, chunked_docs, embedded_chunks
            )
            
            # Update state
            self.current_repository = repo_path
            self.is_ready = True
            
            logger.info(f"Successfully analyzed repository: {repo_path}")
            return {
                "status": "success",
                "message": f"Repository {repo_path} analyzed successfully",
                "repository": repo_path
            }
            
        except Exception as e:
            logger.error(f"Error analyzing repository {repo_path}: {str(e)}")
            self.is_ready = False
            raise e
    
    async def _generate_embeddings(self, chunked_docs: Dict) -> Dict[str, np.ndarray]:
        """Generate embeddings for textual and code chunks"""
        
        # Generate textual embeddings
        textual_embeddings = []
        for doc in chunked_docs['textual_chunks']:
            embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor, embed_textual_metadata, doc["content"]
            )
            textual_embeddings.append(embedding)
        
        # Generate code embeddings
        code_embeddings = []
        for doc in chunked_docs['code_chunks']:
            embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor, generate_code_embedding, doc["content"]
            )
            code_embeddings.append(embedding)
        
        return {
            'textual_embeddings': np.array(textual_embeddings),
            'code_embeddings': np.array(code_embeddings)
        }
    
    async def query_repository(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Query the analyzed repository with a question.
        
        Args:
            question: The question to ask
            top_k: Number of top results to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Query repository called with question: {question}")
        logger.info(f"Service ready status: {self.is_ready}")
        logger.info(f"Collections available: {self.collections is not None}")
        
        if not self.is_ready:
            logger.error("Service is not ready - repository not analyzed")
            raise ValueError("Repository analysis not completed. Service is not ready.")
        
        if not self.collections:
            logger.error("Collections are None - ChromaDB setup failed")
            raise ValueError("ChromaDB collections are not available. Repository analysis failed.")
        
        try:
            logger.info(f"Processing query: {question}")
            
            # Step 1: Retrieve relevant chunks
            logger.info("Step 1: Retrieving relevant chunks...")
            relevant_chunks = await self._retrieve_relevant_chunks(question, top_k)
            logger.info(f"Retrieved chunks: textual={len(relevant_chunks['textual'])}, code={len(relevant_chunks['code'])}")
            
            # Step 2: Construct RAG prompt
            logger.info("Step 2: Constructing RAG prompt...")
            rag_prompt = self._construct_rag_prompt(question, relevant_chunks)
            logger.info(f"RAG prompt constructed, length: {len(rag_prompt)}")
            
            # Step 3: Query AI model
            logger.info("Step 3: Querying AI model...")
            ai_answer = await self._query_ai_model(rag_prompt)
            logger.info(f"AI answer received, length: {len(ai_answer)}")
            
            # Step 4: Format sources
            logger.info("Step 4: Formatting sources...")
            sources = self._format_sources(relevant_chunks)
            logger.info(f"Sources formatted: {len(sources)} sources")
            
            logger.info("Query processing completed successfully")
            return {
                "answer": ai_answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error processing query '{question}': {str(e)}", exc_info=True)
            raise e
    
    async def _retrieve_relevant_chunks(self, query: str, top_k: int) -> Dict[str, List]:
        """Retrieve relevant chunks from ChromaDB collections"""
        
        # Get textual embeddings
        textual_embedding = await asyncio.get_event_loop().run_in_executor(
            self.executor, embed_textual_metadata, query
        )
        
        # Get code embeddings
        code_embedding = await asyncio.get_event_loop().run_in_executor(
            self.executor, generate_code_embedding, query
        )
        
        # Query textual collection
        textual_results = self.collections['textual_collection'].query(
            query_embeddings=[textual_embedding],
            n_results=top_k * 2,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Query code collection
        code_results = self.collections['code_collection'].query(
            query_embeddings=code_embedding.tolist(),
            n_results=top_k * 2,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Process results
        top_textual = self._process_results(textual_results, top_k)
        top_code = self._process_results(code_results, top_k)
        
        return {"textual": top_textual, "code": top_code}
    
    def _process_results(self, results: Dict, top_k: int) -> List[Tuple]:
        """Process ChromaDB query results and return top scored chunks"""
        if "distances" not in results or not results["distances"]:
            logger.warning("'distances' key missing or empty in results")
            return []
        
        # Calculate scores based on distances
        distances = np.array(results["distances"][0])
        scores = 1 - distances  # Invert distance to get similarity score
        
        # Combine scores with documents and metadata
        combined_results = list(zip(scores, results["documents"][0], results["metadatas"][0]))
        
        # Sort by score and return top_k
        return sorted(combined_results, key=lambda x: x[0], reverse=True)[:top_k]
    
    def _construct_rag_prompt(self, query: str, relevant_chunks: Dict) -> str:
        """Construct a RAG-style prompt for the AI model"""
        prompt = f"You are a repository analyser, use the provided chunks to answer any related questions about the repository:\n\nQuestion: {query}\n\nContext:\n"
        
        for chunk_type, chunks in relevant_chunks.items():
            if chunks:
                prompt += f"\n--- {chunk_type.capitalize()} Chunks ---\n"
                for score, text, metadata in chunks:
                    prompt += f"Score: {score:.4f}\n"
                    prompt += f"Content: {text}\n"
                    prompt += f"Metadata: {metadata}\n\n"
            else:
                prompt += f"\n--- No {chunk_type} chunks found ---\n"
        
        prompt += "\nAnswer:"
        return prompt
    
    async def _query_ai_model(self, prompt: str) -> str:
        """Query the OpenAI model with the RAG prompt"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                lambda: openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer the question using only the provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            logger.error(f"Error querying AI model: {str(e)}")
            raise e
    
    def _format_sources(self, relevant_chunks: Dict) -> List[Dict]:
        """Format relevant chunks as sources for the response"""
        sources = []
        
        for chunk_type, chunks in relevant_chunks.items():
            for score, text, metadata in chunks:
                file_name = metadata.get("file_name", "unknown")
                sources.append({
                    # Original fields for backward compatibility
                    "file_name": file_name,
                    "content": text[:500] + "..." if len(text) > 500 else text,  # Keep original truncation
                    "content_type": metadata.get("content_type", chunk_type),
                    "score": float(score),
                    # New fields for frontend as requested in issue
                    "fileName": file_name,  # Full file path for frontend
                    "file_contents": text  # Full content without truncation for frontend
                })
        
        return sources
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the RAG service"""
        return {
            "repository": self.current_repository,
            "ready": self.is_ready,
            "message": f"Repository '{self.current_repository}' is ready for queries" if self.is_ready else "No repository analyzed"
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)