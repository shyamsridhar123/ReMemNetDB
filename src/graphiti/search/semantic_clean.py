"""
Semantic Search Engine using pgvector for vector similarity search.
Part of Graphiti E-commerce Agent Memory Platform.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import text
from openai import AsyncOpenAI

from ..core.config import get_settings
from ..core.database import get_database_session
from ..core.models import TemporalNode

logger = logging.getLogger(__name__)

@dataclass
class SemanticSearchResult:
    """Result from semantic search."""
    node_id: str
    entity_type: str
    identifier: str
    properties: Dict[str, Any]
    similarity_score: float
    valid_from: datetime
    valid_to: Optional[datetime]
    embedding_distance: float

class SemanticSearchEngine:
    """Vector similarity search using pgvector."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        if self.settings.use_azure_openai:
            # Construct proper Azure OpenAI URL
            base_url = f"{self.settings.azure_openai_endpoint.rstrip('/')}"
            self.client = AsyncOpenAI(
                api_key=self.settings.azure_openai_api_key,
                base_url=base_url,
                default_query={"api-version": "2024-02-01"}
            )
        else:
            self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        
    async def generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for search query."""
        try:
            if self.settings.use_azure_openai:
                # For Azure, use the deployment name in the URL path
                response = await self.client.embeddings.create(
                    model=self.settings.azure_openai_embedding_deployment_name,
                    input=query_text.strip()
                )
            else:
                response = await self.client.embeddings.create(
                    model=self.settings.openai_embedding_model,
                    input=query_text.strip()
                )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated query embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    async def search_similar_nodes(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        entity_types: Optional[List[str]] = None,
        time_filter: Optional[datetime] = None
    ) -> List[SemanticSearchResult]:
        """Search for nodes similar to query embedding."""
        
        # Build the query
        conditions = ["embedding IS NOT NULL"]
        params = {
            "query_embedding": str(query_embedding),
            "limit": limit,
            "threshold": 1.0 - similarity_threshold  # Convert similarity to distance
        }
        
        if entity_types:
            conditions.append("entity_type = ANY(:entity_types)")
            params["entity_types"] = entity_types
            
        if time_filter:
            conditions.append("valid_from <= :time_filter")
            conditions.append("(valid_to IS NULL OR valid_to > :time_filter)")
            params["time_filter"] = time_filter
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
        SELECT 
            node_id,
            entity_type,
            identifier,
            properties,
            valid_from,
            valid_to,
            (embedding <=> :query_embedding::vector) as distance,
            1 - (embedding <=> :query_embedding::vector) as similarity
        FROM temporal_graph.temporal_nodes
        WHERE {where_clause}
            AND (embedding <=> :query_embedding::vector) < :threshold
        ORDER BY embedding <=> :query_embedding::vector
        LIMIT :limit
        """
        
        try:
            with get_database_session() as db:
                result = db.execute(text(query), params)
                rows = result.fetchall()
                
                search_results = []
                for row in rows:
                    search_results.append(SemanticSearchResult(
                        node_id=row.node_id,
                        entity_type=row.entity_type,
                        identifier=row.identifier,
                        properties=row.properties,
                        similarity_score=row.similarity,
                        valid_from=row.valid_from,
                        valid_to=row.valid_to,
                        embedding_distance=row.distance
                    ))
                
                logger.info(f"Found {len(search_results)} similar nodes")
                return search_results
                
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    async def search(
        self,
        query_text: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        entity_types: Optional[List[str]] = None,
        time_filter: Optional[datetime] = None
    ) -> List[SemanticSearchResult]:
        """End-to-end semantic search."""
        
        # Generate embedding for query
        query_embedding = await self.generate_query_embedding(query_text)
        
        # Search for similar nodes
        results = await self.search_similar_nodes(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=similarity_threshold,
            entity_types=entity_types,
            time_filter=time_filter
        )
        
        logger.info(f"Semantic search for '{query_text}' returned {len(results)} results")
        return results
    
    async def find_related_entities(
        self,
        entity_id: str,
        relation_types: Optional[List[str]] = None,
        max_hops: int = 2,
        limit: int = 20
    ) -> List[SemanticSearchResult]:
        """Find entities related to a given entity through semantic similarity."""
        
        try:
            # First, get the source entity's embedding
            with get_database_session() as db:
                source_query = text("""
                SELECT embedding, entity_type, identifier
                FROM temporal_graph.temporal_nodes 
                WHERE node_id = :entity_id AND embedding IS NOT NULL
                """)
                
                result = db.execute(source_query, {"entity_id": entity_id})
                source_row = result.fetchone()
                
                if not source_row or not source_row.embedding:
                    logger.warning(f"No embedding found for entity {entity_id}")
                    return []
                
                # Use the source entity's embedding to find similar entities
                return await self.search_similar_nodes(
                    query_embedding=source_row.embedding,
                    limit=limit,
                    similarity_threshold=0.6  # Lower threshold for related entities
                )
                
        except Exception as e:
            logger.error(f"Failed to find related entities for {entity_id}: {e}")
            return []
