# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Veterinary Nutrition RAG Engine

This engine provides a Retrieval-Augmented Generation (RAG) capability
focused on veterinary nutrition. It uses a local, offline sentence-transformer
model and a FAISS vector index to retrieve relevant information from a
built-in knowledge base of WSAVA and AAFCO guidelines. This engine is
100% free, requires no API keys, and is fully self-contained.
"""

import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from .knowledge_base.vet_nutrition_kb import WSAVA_NUTRITION_GUIDELINES, AAFCO_PET_FOOD_LABEL_GUIDELINES
except ImportError as e:
    logger.error(f"Missing dependencies for VeterinaryNutritionRAGEngine: {e}. Please run 'pip install sentence-transformers faiss-cpu'.")
    raise

class VeterinaryNutritionRAGEngine:
    """
    Implements a RAG system for veterinary nutrition.
    """
    
    def __init__(self):
        self.name = "Veterinary Nutrition RAG Engine"
        self.version = "1.0.0"
        self.supported_calculations = ["query_knowledge_base"]
        self.model = None
        self.index = None
        self.knowledge_chunks = []
        self._initialize_engine()

    def _initialize_engine(self):
        """
        Initializes the model, processes the knowledge base, and builds the FAISS index.
        This is a one-time setup cost when the engine is instantiated.
        """
        try:
            logger.info("Initializing Veterinary Nutrition RAG Engine...")
            
            # 1. Load the Sentence Transformer model
            # This will be downloaded and cached on the first run.
            logger.info("Loading sentence-transformer model 'all-MiniLM-L6-v2'...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', "cpu")
            logger.info("✅ Model loaded.")

            # 2. Prepare the knowledge base chunks
            self.knowledge_chunks = WSAVA_NUTRITION_GUIDELINES + AAFCO_PET_FOOD_LABEL_GUIDELINES
            chunk_contents = [chunk['content'] for chunk in self.knowledge_chunks]

            # 3. Encode the knowledge base
            logger.info(f"Encoding {len(chunk_contents)} knowledge base chunks...")
            embeddings = self.model.encode(chunk_contents, convert_to_tensor=False, show_progress_bar=False)
            logger.info("✅ Knowledge base encoded.")

            # 4. Build the FAISS index
            embedding_dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.index.add(np.array(embeddings, dtype=np.float32))
            logger.info(f"✅ FAISS index built successfully. Index contains {self.index.ntotal} vectors.")

        except Exception as e:
            logger.error(f"❌ Critical error during VeterinaryNutritionRAGEngine initialization: {e}")
            # This will prevent the engine from being used if setup fails.
            self.model = None
            self.index = None

    def query_knowledge_base(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Searches the knowledge base for text chunks relevant to the query.
        """
        if not self.index or not self.model:
            return {"error": "RAG Engine is not initialized. Please check logs for initialization errors."}
            
        try:
            query = parameters['query']
            top_k = parameters.get('top_k', 3)
            
            if not isinstance(query, str) or not query:
                return {"error": "Query must be a non-empty string."}

            logger.info(f"Executing RAG query: '{query}'")
            
            # Encode the query
            query_embedding = self.model.encode([query])
            
            # Search the index
            distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k=min(top_k, self.index.ntotal))
            
            # Format results
            retrieved_chunks = []
            for i, idx in enumerate(indices[0]):
                chunk = self.knowledge_chunks[idx]
                retrieved_chunks.append({
                    "source": chunk['source'],
                    "content": chunk['content'],
                    "relevance_score": float(1 / (1 + distances[0][i])) # Convert L2 distance to a similarity score
                })
            
            return {
                "query": query,
                "retrieved_chunks": retrieved_chunks
            }

        except KeyError as e:
            return {"error": f"Missing required parameter: {e}"}
        except Exception as e:
            logger.error(f"Error during RAG query: {e}")
            return {"error": f"An unexpected error occurred: {e}"} 