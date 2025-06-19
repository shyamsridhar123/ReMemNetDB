"""
Simple keyword search implementation for testing.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import text
from ..core.database import get_database_session
from ..core.models import TemporalNode

logger = logging.getLogger(__name__)

@dataclass
class KeywordSearchResult:
    """Result from keyword search."""
    node_id: str
    entity_type: str
    identifier: str
    properties: Dict[str, Any]
    valid_from: datetime
    valid_to: Optional[datetime]
    relevance_score: float
    matched_fields: List[str]
    search_rank: float

class KeywordSearchEngine:
    """Simple keyword search using PostgreSQL ILIKE."""
    
    def __init__(self, settings=None):
        self.settings = settings
    
    async def search(
        self,
        query_text: str,
        limit: int = 10,
        entity_types: Optional[List[str]] = None,
        time_filter: Optional[datetime] = None,
        min_rank: float = 0.1
    ) -> List[KeywordSearchResult]:
        """Simple keyword search."""
        try:
            # Initialize params dictionary
            params = {"limit": limit}
            
            # Smart keyword search - split terms and use OR logic
            search_terms = query_text.strip().split()
            if len(search_terms) > 1:
                # Multiple terms - search for each term separately
                term_conditions = []
                for i, term in enumerate(search_terms):
                    params[f"term_{i}"] = f"%{term}%"
                    term_conditions.append(f"(COALESCE(properties->>'identifier', CAST(id as text)) ILIKE :term_{i} OR properties::text ILIKE :term_{i})")
                conditions = [f"({' OR '.join(term_conditions)})"]
            else:
                # Single term
                conditions = ["(COALESCE(properties->>'identifier', CAST(id as text)) ILIKE :query OR properties::text ILIKE :query)"]
                params["query"] = f"%{query_text}%"
            
            if entity_types:
                conditions.append("type = ANY(:entity_types)")
                params["entity_types"] = entity_types
                
            if time_filter:
                conditions.append("valid_from <= :time_filter")
                conditions.append("(valid_to IS NULL OR valid_to > :time_filter)")
                params["time_filter"] = time_filter
            
            where_clause = " AND ".join(conditions)
            
            query = f"""
            SELECT 
                id as node_id,
                type as entity_type,
                COALESCE(properties->>'identifier', CAST(id as text)) as identifier,
                properties,
                valid_from,
                valid_to,
                1.0 as search_rank
            FROM {TemporalNode.__table__.fullname}
            WHERE {where_clause}
            ORDER BY valid_from DESC
            LIMIT :limit
            """
            
            with get_database_session() as db:
                result = db.execute(text(query), params)
                rows = result.fetchall()
                
                search_results = []
                for row in rows:
                    search_results.append(KeywordSearchResult(
                        node_id=row.node_id,
                        entity_type=row.entity_type,
                        identifier=row.identifier,
                        properties=row.properties,
                        valid_from=row.valid_from,
                        valid_to=row.valid_to,
                        relevance_score=1.0,
                        matched_fields=["identifier", "properties"],
                        search_rank=1.0
                    ))
                
                logger.info(f"Keyword search for '{query_text}' returned {len(search_results)} results")
                return search_results
                
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
