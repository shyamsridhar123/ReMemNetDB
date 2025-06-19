"""
Hybrid Search Engine combining semantic, keyword, and graph search.
Part of Graphiti E-commerce Agent Memory Platform.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from .semantic import SemanticSearchEngine, SemanticSearchResult
from .keyword import KeywordSearchEngine, KeywordSearchResult
from .graph_search import GraphSearchEngine, GraphSearchResult

logger = logging.getLogger(__name__)

@dataclass
class HybridSearchResult:
    """Unified result from hybrid search combining all search types."""
    node_id: str
    entity_type: str
    identifier: str
    properties: Dict[str, Any]
    valid_from: datetime
    valid_to: Optional[datetime]
    
    # Scoring from different search types
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    graph_score: float = 0.0
    
    # Combined score
    hybrid_score: float = 0.0
    
    # Source information
    found_by: List[str] = field(default_factory=list)  # ['semantic', 'keyword', 'graph']
    
    # Detailed results from each search type
    semantic_result: Optional[SemanticSearchResult] = None
    keyword_result: Optional[KeywordSearchResult] = None
    graph_result: Optional[GraphSearchResult] = None
    
    # Additional metadata
    matched_fields: List[str] = field(default_factory=list)
    relationship_types: List[str] = field(default_factory=list)
    distance_from_query: Optional[int] = None

@dataclass
class SearchWeights:
    """Weights for combining different search types."""
    semantic: float = 0.4
    keyword: float = 0.3
    graph: float = 0.3
    
    def normalize(self):
        """Normalize weights to sum to 1.0."""
        total = self.semantic + self.keyword + self.graph
        if total > 0:
            self.semantic /= total
            self.keyword /= total
            self.graph /= total

class HybridSearchEngine:
    """Main search engine combining semantic, keyword, and graph search."""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.semantic_engine = SemanticSearchEngine(settings)
        self.keyword_engine = KeywordSearchEngine(settings)
        self.graph_engine = GraphSearchEngine(settings)
        
    async def search(
        self,
        query: str,
        limit: int = 20,
        weights: Optional[SearchWeights] = None,
        entity_types: Optional[List[str]] = None,
        time_filter: Optional[datetime] = None,
        enable_semantic: bool = True,
        enable_keyword: bool = True,
        enable_graph: bool = False,  # Graph search requires a source entity
        source_entity_id: Optional[str] = None,
        semantic_threshold: float = 0.4,
        keyword_min_rank: float = 0.1,
        graph_max_hops: int = 3
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining multiple search types.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            weights: Weights for combining search types
            entity_types: Filter by entity types
            time_filter: Filter by time
            enable_semantic: Enable semantic search
            enable_keyword: Enable keyword search
            enable_graph: Enable graph search (requires source_entity_id)
            source_entity_id: Source entity for graph search
            semantic_threshold: Minimum similarity threshold for semantic search
            keyword_min_rank: Minimum rank for keyword search
            graph_max_hops: Maximum hops for graph traversal
        """
        
        if weights is None:
            weights = SearchWeights()
        weights.normalize()
        
        # Run searches in parallel
        search_tasks = []
        
        if enable_semantic:
            search_tasks.append(self._run_semantic_search(
                query, limit, entity_types, time_filter, semantic_threshold
            ))
        else:
            search_tasks.append(asyncio.create_task(self._empty_semantic_results()))
            
        if enable_keyword:
            search_tasks.append(self._run_keyword_search(
                query, limit, entity_types, time_filter, keyword_min_rank
            ))
        else:
            search_tasks.append(asyncio.create_task(self._empty_keyword_results()))
            
        if enable_graph and source_entity_id:
            search_tasks.append(self._run_graph_search(
                source_entity_id, graph_max_hops, None, entity_types, limit
            ))
        else:
            search_tasks.append(asyncio.create_task(self._empty_graph_results()))
        
        # Wait for all searches to complete
        try:
            semantic_results, keyword_results, graph_results = await asyncio.gather(*search_tasks)
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
        
        # Combine and rank results
        hybrid_results = self._combine_results(
            semantic_results, keyword_results, graph_results, weights
        )
        
        # Sort by hybrid score and limit results
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        final_results = hybrid_results[:limit]
        
        logger.info(f"Hybrid search for '{query}' returned {len(final_results)} results")
        logger.info(f"Search breakdown: {len(semantic_results)} semantic, {len(keyword_results)} keyword, {len(graph_results)} graph")
        
        return final_results
    
    async def search_related_entities(
        self,
        entity_id: str,
        query: Optional[str] = None,
        limit: int = 20,
        weights: Optional[SearchWeights] = None,
        max_hops: int = 2,
        semantic_threshold: float = 0.6
    ) -> List[HybridSearchResult]:
        """Find entities related to a specific entity."""
        
        if weights is None:
            weights = SearchWeights(semantic=0.3, keyword=0.2, graph=0.5)
        weights.normalize()
        
        # Run all search types
        search_tasks = [
            self._run_graph_search(entity_id, max_hops, None, None, limit),
            self.semantic_engine.find_related_entities(entity_id, None, max_hops, limit),
        ]
        
        if query:
            search_tasks.append(self._run_keyword_search(query, limit, None, None, 0.1))
        else:
            search_tasks.append(asyncio.create_task(self._empty_keyword_results()))
        
        try:
            graph_results, semantic_results, keyword_results = await asyncio.gather(*search_tasks)
            
            # Convert semantic results to the expected format
            semantic_search_results = [
                SemanticSearchResult(
                    node_id=r.node_id,
                    entity_type=r.entity_type,
                    identifier=r.identifier,
                    properties=r.properties,
                    similarity_score=r.similarity_score,
                    valid_from=r.valid_from,
                    valid_to=r.valid_to,
                    embedding_distance=r.embedding_distance
                ) for r in semantic_results
            ]
            
        except Exception as e:
            logger.error(f"Related entity search failed: {e}")
            return []
        
        # Combine results
        hybrid_results = self._combine_results(
            semantic_search_results, keyword_results, graph_results, weights
        )
        
        # Sort and limit
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return hybrid_results[:limit]
    
    async def _run_semantic_search(
        self, query: str, limit: int, entity_types: Optional[List[str]], 
        time_filter: Optional[datetime], threshold: float
    ) -> List[SemanticSearchResult]:
        """Run semantic search."""
        try:
            return await self.semantic_engine.search(
                query_text=query,
                limit=limit,
                similarity_threshold=threshold,
                entity_types=entity_types,
                time_filter=time_filter
            )
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _run_keyword_search(
        self, query: str, limit: int, entity_types: Optional[List[str]], 
        time_filter: Optional[datetime], min_rank: float
    ) -> List[KeywordSearchResult]:
        """Run keyword search."""
        try:
            return await self.keyword_engine.search(
                query_text=query,
                limit=limit,
                entity_types=entity_types,
                time_filter=time_filter,
                min_rank=min_rank
            )
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    async def _run_graph_search(
        self, source_id: str, max_hops: int, relationship_types: Optional[List[str]], 
        entity_types: Optional[List[str]], limit: int
    ) -> List[GraphSearchResult]:
        """Run graph search."""
        try:
            return await self.graph_engine.find_connected_entities(
                source_entity_id=source_id,
                max_hops=max_hops,
                relationship_types=relationship_types,
                entity_types=entity_types,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    async def _empty_semantic_results(self) -> List[SemanticSearchResult]:
        """Return empty semantic results."""
        return []
    
    async def _empty_keyword_results(self) -> List[KeywordSearchResult]:
        """Return empty keyword results."""
        return []    
    async def _empty_graph_results(self) -> List[GraphSearchResult]:
        """Return empty graph results."""
        return []
    
    def _combine_results(
        self,
        semantic_results: List[SemanticSearchResult],
        keyword_results: List[KeywordSearchResult],
        graph_results: List[GraphSearchResult],
        weights: SearchWeights
    ) -> List[HybridSearchResult]:
        """Combine results from all search types."""
        
        # Log the inputs
        logger.debug(f"🔄 Combining results: {len(semantic_results)} semantic, {len(keyword_results)} keyword, {len(graph_results)} graph")
        
        # Index results by node_id
        combined_results = defaultdict(lambda: {
            'semantic': None,
            'keyword': None,
            'graph': None,
            'node_data': None
        })
        
        # Process semantic results
        for result in semantic_results:
            logger.debug(f"🔄 Processing semantic result: {result.node_id}")
            combined_results[result.node_id]['semantic'] = result
            combined_results[result.node_id]['node_data'] = {
                'node_id': result.node_id,
                'entity_type': result.entity_type,
                'identifier': result.identifier,
                'properties': result.properties,
                'valid_from': result.valid_from,
                'valid_to': result.valid_to
            }
        
        # Process keyword results
        for result in keyword_results:
            logger.debug(f"🔄 Processing keyword result: {result.node_id}")
            combined_results[result.node_id]['keyword'] = result
            if not combined_results[result.node_id]['node_data']:
                combined_results[result.node_id]['node_data'] = {
                    'node_id': result.node_id,
                    'entity_type': result.entity_type,
                    'identifier': result.identifier,
                    'properties': result.properties,
                    'valid_from': result.valid_from,
                    'valid_to': result.valid_to
                }
        
        # Process graph results
        for result in graph_results:
            logger.debug(f"🔄 Processing graph result: {result.node_id}")
            combined_results[result.node_id]['graph'] = result
            if not combined_results[result.node_id]['node_data']:
                combined_results[result.node_id]['node_data'] = {
                    'node_id': result.node_id,
                    'entity_type': result.entity_type,
                    'identifier': result.identifier,
                    'properties': result.properties,
                    'valid_from': result.valid_from,
                    'valid_to': result.valid_to
                }
        
        logger.debug(f"🔄 Combined node data for {len(combined_results)} unique nodes")
        
        # Create hybrid results
        hybrid_results = []
        for node_id, data in combined_results.items():
            if not data['node_data']:
                logger.warning(f"🔄 Skipping node {node_id} - no node data")
                continue
                
            # Calculate scores
            semantic_score = data['semantic'].similarity_score if data['semantic'] else 0.0
            keyword_score = data['keyword'].relevance_score if data['keyword'] else 0.0
            graph_score = data['graph'].traversal_score if data['graph'] else 0.0
            
            # Normalize scores to 0-1 range
            semantic_score = min(1.0, max(0.0, semantic_score))
            keyword_score = min(1.0, max(0.0, keyword_score))
            graph_score = min(1.0, max(0.0, graph_score))
            
            # Calculate hybrid score
            hybrid_score = (
                weights.semantic * semantic_score +
                weights.keyword * keyword_score +
                weights.graph * graph_score            )
            
            # Calculate hybrid score
            hybrid_score = (
                weights.semantic * semantic_score +
                weights.keyword * keyword_score +
                weights.graph * graph_score
            )
            
            logger.debug(f"🔄 Node {node_id}: sem={semantic_score:.3f}, key={keyword_score:.3f}, graph={graph_score:.3f}, hybrid={hybrid_score:.3f}")
            
            # Determine which search types found this result
            found_by = []
            if data['semantic']:
                found_by.append('semantic')
            if data['keyword']:
                found_by.append('keyword')
            if data['graph']:
                found_by.append('graph')
            
            # Collect metadata
            matched_fields = data['keyword'].matched_fields if data['keyword'] else []
            relationship_types = data['graph'].relationship_types if data['graph'] else []
            distance_from_query = data['graph'].distance_from_source if data['graph'] else None
            
            hybrid_result = HybridSearchResult(
                node_id=data['node_data']['node_id'],
                entity_type=data['node_data']['entity_type'],
                identifier=data['node_data']['identifier'],
                properties=data['node_data']['properties'],
                valid_from=data['node_data']['valid_from'],
                valid_to=data['node_data']['valid_to'],
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                graph_score=graph_score,
                hybrid_score=hybrid_score,
                found_by=found_by,
                semantic_result=data['semantic'],
                keyword_result=data['keyword'],
                graph_result=data['graph'],
                matched_fields=matched_fields,
                relationship_types=relationship_types,
                distance_from_query=distance_from_query
            )
            
            hybrid_results.append(hybrid_result)
            logger.debug(f"🔄 Added hybrid result for {node_id}")
        
        logger.debug(f"🔄 Final hybrid results: {len(hybrid_results)} total results")
        if len(hybrid_results) == 0 and (semantic_results or keyword_results or graph_results):
            logger.warning("🔄 No hybrid results found despite having individual search results!")
            logger.debug(f"🔄 Semantic results: {[r.node_id for r in semantic_results]}")
            logger.debug(f"🔄 Keyword results: {[r.node_id for r in keyword_results]}")
            logger.debug(f"🔄 Graph results: {[r.node_id for r in graph_results]}")
        
        return hybrid_results
    
    async def analyze_search_coverage(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        time_filter: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze search coverage across different search types."""
        
        # Run all search types
        tasks = [
            self._run_semantic_search(query, 50, entity_types, time_filter, 0.5),
            self._run_keyword_search(query, 50, entity_types, time_filter, 0.05),
        ]
        
        semantic_results, keyword_results = await asyncio.gather(*tasks)
        
        # Analyze overlap
        semantic_ids = {r.node_id for r in semantic_results}
        keyword_ids = {r.node_id for r in keyword_results}
        
        total_unique = len(semantic_ids | keyword_ids)
        overlap = len(semantic_ids & keyword_ids)
        
        analysis = {
            'total_results': {
                'semantic': len(semantic_results),
                'keyword': len(keyword_results),
                'unique_total': total_unique
            },
            'overlap': {
                'count': overlap,
                'percentage': (overlap / total_unique * 100) if total_unique > 0 else 0
            },
            'coverage': {
                'semantic_only': len(semantic_ids - keyword_ids),
                'keyword_only': len(keyword_ids - semantic_ids),
                'both': overlap
            }
        }
        
        return analysis
