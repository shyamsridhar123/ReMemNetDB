"""
Simple graph search implementation for testing.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import text
from ..core.database import get_database_session

logger = logging.getLogger(__name__)

@dataclass
class GraphSearchResult:
    """Result from graph search."""
    node_id: str
    entity_type: str
    identifier: str
    properties: Dict[str, Any]
    valid_from: datetime
    valid_to: Optional[datetime]
    distance_from_source: int
    path_to_source: Optional[Any]
    relationship_types: List[str]
    traversal_score: float

class GraphSearchEngine:
    """Simple graph search implementation."""
    
    def __init__(self, settings=None):
        self.settings = settings
        
    async def find_connected_entities(
        self,
        source_entity_id: str,
        max_hops: int = 3,
        relationship_types: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[GraphSearchResult]:
        """Find connected entities (simplified implementation)."""
        
        try:
            # Simple query to find entities connected via edges
            conditions = ["e.source_node_id = :source_id OR e.target_node_id = :source_id"]
            params = {"source_id": source_entity_id, "limit": limit}
            
            if relationship_types:
                conditions.append("e.relationship_type = ANY(:relationship_types)")
                params["relationship_types"] = relationship_types
                
            if entity_types:
                conditions.append("n.entity_type = ANY(:entity_types)")
                params["entity_types"] = entity_types
            
            where_clause = " AND ".join(conditions)
            
            query = f"""
            SELECT DISTINCT
                n.node_id,
                n.entity_type,
                n.identifier,
                n.properties,
                n.valid_from,
                n.valid_to,
                1 as distance,
                ARRAY[e.relationship_type] as relationship_types,
                e.confidence as traversal_score
            FROM temporal_graph.temporal_edges e
            JOIN temporal_graph.temporal_nodes n ON (
                CASE 
                    WHEN e.source_node_id = :source_id THEN n.node_id = e.target_node_id
                    ELSE n.node_id = e.source_node_id
                END
            )
            WHERE {where_clause}
                AND n.node_id != :source_id
            ORDER BY e.confidence DESC
            LIMIT :limit
            """
            
            with get_database_session() as db:
                result = db.execute(text(query), params)
                rows = result.fetchall()
                
                search_results = []
                for row in rows:
                    search_results.append(GraphSearchResult(
                        node_id=row.node_id,
                        entity_type=row.entity_type,
                        identifier=row.identifier,
                        properties=row.properties,
                        valid_from=row.valid_from,
                        valid_to=row.valid_to,
                        distance_from_source=row.distance,
                        path_to_source=None,
                        relationship_types=row.relationship_types,
                        traversal_score=row.traversal_score
                    ))
                
                logger.info(f"Graph search from {source_entity_id} returned {len(search_results)} results")
                return search_results
                
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
