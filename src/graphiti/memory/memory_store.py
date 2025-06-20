"""
memory/memory_store.py - Memory Storage and Retrieval (Step 4 Implementation)

Implements the MemoryStore class as specified in the implementation plan.
This is the central integration point that brings together:
- Temporal Graph Engine (Step 3)
- Semantic Search (Step 5)
- Keyword Search (Step 6)
- Entity Extraction
"""

import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import or_, String

from .temporal_graph import TemporalGraph, TimeRange
from .extraction import EntityExtractor
from ..core.models import Event, TemporalNode, TemporalEdge, NodeSchema, EdgeSchema
from ..core.database import db_manager
from ..search.semantic import SemanticSearchEngine
from ..search.keyword import KeywordSearchEngine
from ..search.hybrid import HybridSearchEngine

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Represents a complete episode in memory."""
    id: str
    customer_id: str
    events: List[Event]
    start_time: datetime
    end_time: datetime
    summary: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]


@dataclass
class Memory:
    """Represents a retrieved memory with context."""
    id: str
    content: Dict[str, Any]
    timestamp: datetime
    relevance_score: float
    memory_type: str  # 'episode', 'entity', 'relationship'
    context: Dict[str, Any]


@dataclass
class MemoryQuery:
    """Query parameters for memory retrieval."""
    query_text: str
    query_type: str  # 'semantic', 'keyword', 'temporal', 'hybrid'
    time_range: Optional[TimeRange] = None
    entity_filter: Optional[str] = None
    max_results: int = 20
    include_context: bool = True


class MemoryStore:
    """
    Memory Storage and Retrieval System (Step 4 Implementation)
    
    This class implements the exact interface specified in the implementation plan
    and serves as the central hub for all memory operations.
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize the memory store with all required components."""
        logger.info("🧠 Initializing MemoryStore...")
        
        # Initialize temporal graph engine
        self.graph = TemporalGraph(db_session)
        
        # Initialize entity extraction
        self.extractor = EntityExtractor()
        
        # Initialize search engines
        self.hybrid_search = HybridSearchEngine()
        self.semantic_search = SemanticSearchEngine()
        self.keyword_search = KeywordSearchEngine()
        
        # Store external session flag
        self.external_session = db_session is not None
        
        logger.info("✅ MemoryStore initialized with all components")
    
    def store_episode(self, episode: Episode) -> None:
        """
        Store complete episode in memory.
        
        This processes all events in the episode through the temporal graph
        and ensures proper indexing for search.
        """
        logger.info(f"📝 Storing episode {episode.id} with {len(episode.events)} events")
        try:
            # Store each event in the temporal graph
            for event in episode.events:
                self.graph.add_event(event)
            
            logger.info(f"✅ Successfully stored episode {episode.id}")
        except Exception as e:
            logger.error(f"❌ Failed to store episode {episode.id}: {str(e)}")            
            raise
    
    def store_event(self, customer_id: str, event_data: Dict[str, Any], 
                   event_type: str, timestamp: datetime) -> str:
        """Store a single event in memory."""
        print(f"🚀 DEBUG: store_event called with event_type={event_type}")
        logger.info(f"📝 Storing single event: {event_type}")
        
        try:
            logger.debug(f"🔧 Creating Event and EventSchema objects...")
            # Create Event object with correct parameters
            event = Event(
                event_type=event_type,
                event_data={**event_data, 'customer_id': customer_id},
                timestamp=timestamp
            )
              # Create EventSchema for extraction
            from ..core.models import EventSchema
            event_schema = EventSchema(
                event_type=event_type,
                event_data={**event_data, 'customer_id': customer_id},
                timestamp=timestamp
            )
              # Extract entities with embeddings
            print(f"🔧 DEBUG: About to start EntityExtractor...")
            import asyncio
            import concurrent.futures
            
            def extract_entities_sync():
                """Run entity extraction in a new event loop."""
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    print(f"🔧 DEBUG: Running extractor.extract_entities...")
                    result = loop.run_until_complete(self.extractor.extract_entities(event_schema))
                    print(f"🔧 DEBUG: Extraction completed successfully, got {len(result)} entities")
                    return result
                except Exception as e:
                    print(f"❌ ERROR in extract_entities_sync: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                finally:
                    loop.close()
            
            # Run extraction in a separate thread to avoid event loop conflicts
            try:
                print(f"🔧 DEBUG: Starting ThreadPoolExecutor...")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(extract_entities_sync)
                    entities = future.result(timeout=60)  # Increased timeout
                print(f"🔧 DEBUG: ThreadPoolExecutor completed successfully")
            except Exception as e:
                print(f"❌ ERROR in ThreadPoolExecutor: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            logger.info(f"📊 Extracted {len(entities)} entities: {[e.type + ':' + (e.identifier or 'None') for e in entities]}")
            
            # Debug: Check if entities have embeddings
            for entity in entities:
                has_embedding = hasattr(entity, 'embedding') and entity.embedding is not None
                embedding_len = len(entity.embedding) if has_embedding else 0
                logger.debug(f"Entity {entity.identifier}: has_embedding={has_embedding}, embedding_len={embedding_len}")
            
            # Add the event to the temporal graph with the extracted entities
            self.graph.add_event(event, entities)
            
            episode_id = f"episode_{customer_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"✅ Successfully stored event {event_type}")
            return episode_id
        except Exception as e:
            logger.error(f"❌ Failed to store event: {str(e)}")
            raise
    
    def retrieve_memories(self, query: MemoryQuery) -> List[Memory]:
        """
        Retrieve relevant memories based on query.
        
        Implements hybrid search across semantic, keyword, and temporal dimensions.
        """
        logger.info(f"🔍 Retrieving memories for query: {query.query_text[:50]}...")
        
        try:
            memories = []
            
            if query.query_type == 'semantic':
                memories = self._semantic_memory_search(query)
            elif query.query_type == 'keyword':
                memories = self._keyword_memory_search(query)
            elif query.query_type == 'temporal':
                memories = self._temporal_memory_search(query)
            elif query.query_type == 'hybrid':
                memories = self._hybrid_memory_search(query)
            else:
                raise ValueError(f"Unknown query type: {query.query_type}")
            
            logger.info(f"📊 Retrieved {len(memories)} memories")
            return memories[:query.max_results]
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve memories: {str(e)}")
            raise
            
    def get_temporal_sequence(self, entity_id: str, include_relations: bool = True) -> Dict[str, Any]:
        """Get a temporal sequence of events for an entity."""
        logger.info(f"📅 Getting temporal sequence for entity {entity_id}")
        logger.debug(f"📅 Entity ID format: {entity_id} (length: {len(entity_id)}, dashes: {entity_id.count('-')})")
        
        try:
            with db_manager.get_session() as session:
                # Multi-strategy approach to find the customer
                nodes = []
                
                # Strategy 1: Direct queries on known fields
                query = session.query(TemporalNode).filter(
                    or_(
                        TemporalNode.properties['customer_id'].astext == entity_id,
                        TemporalNode.properties['identifier'].astext == entity_id,
                        TemporalNode.id == entity_id
                    )
                ).order_by(TemporalNode.valid_from)
                
                nodes = query.all()
                logger.info(f"📊 Strategy 1 found {len(nodes)} nodes using standard query")
                
                # Strategy 2: If no results, try broader text search
                if not nodes:
                    logger.warning(f"📊 No nodes found with direct ID match, trying text search")
                    # Search in all properties for the entity_id
                    query = session.query(TemporalNode).filter(
                        TemporalNode.properties.cast(String).like(f'%{entity_id}%')
                    ).order_by(TemporalNode.valid_from)
                    nodes = query.all()
                    logger.info(f"📊 Strategy 2 found {len(nodes)} nodes using text search")
                
                # Strategy 3: If still no results, try customer type nodes
                if not nodes:
                    logger.warning(f"📊 Still no nodes found, searching customer type nodes")
                    query = session.query(TemporalNode).filter(
                        TemporalNode.type == 'customer'
                    ).order_by(TemporalNode.valid_from)
                    all_customers = query.all()
                    logger.info(f"📊 Found {len(all_customers)} total customer nodes")
                    
                    # Log customer details for debugging
                    for customer in all_customers[:5]:  # Log first 5 customers
                        logger.debug(f"📊 Customer: {customer.id}, props: {customer.properties}")
                
                # Convert nodes to dictionaries within session
                entity_history = []
                for node in nodes:
                    node_dict = {
                        'id': str(node.id),
                        'type': node.type,
                        'properties': node.properties or {},
                        'valid_from': node.valid_from,
                        'valid_to': node.valid_to,
                        'recorded_at': node.recorded_at,
                        'embedding': node.embedding
                    }
                    entity_history.append(node_dict)
                    logger.debug(f"📊 Added node: {node.type} - {node.properties.get('identifier', str(node.id)) if node.properties else str(node.id)}")
                
                logger.info(f"📊 Found {len(entity_history)} events in temporal sequence")
                
                return {
                    'entity_id': entity_id,
                    'nodes': entity_history,
                    'total_events': len(entity_history)
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get temporal sequence: {str(e)}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return {
                'entity_id': entity_id,
                'nodes': [],
                'total_events': 0,
                'error': str(e)
            }    
    def get_customer_episodes(self, customer_id: str, days_back: int = 30) -> List[Episode]:
        """Get all episodes for a specific customer."""
        logger.info(f"👤 Getting episodes for customer {customer_id}")
        
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days_back)
            
            # Query events directly from the events table
            with db_manager.get_session() as session:
                db_events = session.query(Event).filter(
                    Event.event_data['customer_id'].astext == customer_id,
                    Event.timestamp >= start_time,
                    Event.timestamp <= end_time
                ).order_by(Event.timestamp).all()
                
                logger.info(f"📊 Found {len(db_events)} events for customer {customer_id}")
                
                # Convert to our Event objects
                events = []
                for db_event in db_events:
                    try:
                        event = Event(
                            id=db_event.id,
                            event_type=db_event.event_type,
                            event_data=db_event.event_data,
                            timestamp=db_event.timestamp,
                            processed=db_event.processed
                        )
                        events.append(event)
                        logger.debug(f"📊 Event: {event.event_type} at {event.timestamp}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to convert event: {e}")
                        continue
                
                logger.info(f"📊 Converted {len(events)} database events to Event objects")
            
            # Group events into episodes (simple implementation)
            episodes = self._group_events_into_episodes(events, customer_id)
            
            logger.info(f"📊 Found {len(episodes)} episodes for customer {customer_id}")
            return episodes
            
        except Exception as e:
            logger.error(f"❌ Failed to get customer episodes: {str(e)}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _semantic_memory_search(self, query: MemoryQuery) -> List[Memory]:
        """Perform semantic search across memories."""
        logger.debug("🔍 Performing semantic memory search")
        
        try:
            # Use semantic search engine
            search_results = self.semantic_search.search(
                query_text=query.query_text,
                limit=query.max_results
            )
            
            # Convert to Memory objects
            memories = []
            for result in search_results:
                memory = Memory(
                    id=str(uuid.uuid4()),
                    content=result.get('content', {}),
                    timestamp=result.get('timestamp', datetime.now(timezone.utc)),
                    relevance_score=result.get('similarity_score', 0.0),
                    memory_type='semantic',
                    context=result.get('context', {})
                )
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {str(e)}")
            return []
    
    def _keyword_memory_search(self, query: MemoryQuery) -> List[Memory]:
        """Perform keyword search across memories."""
        logger.debug("🔍 Performing keyword memory search")
        
        try:
            # Use keyword search engine
            search_results = self.keyword_search.search(
                query_text=query.query_text,
                limit=query.max_results
            )
            
            # Convert to Memory objects
            memories = []
            for result in search_results:
                memory = Memory(
                    id=str(uuid.uuid4()),
                    content=result.get('content', {}),
                    timestamp=result.get('timestamp', datetime.now(timezone.utc)),
                    relevance_score=result.get('score', 0.0),
                    memory_type='keyword',
                    context=result.get('context', {})
                )
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"❌ Keyword search failed: {str(e)}")
            return []
    
    def _temporal_memory_search(self, query: MemoryQuery) -> List[Memory]:
        """Perform temporal search across memories."""
        logger.debug("🔍 Performing temporal memory search")
        
        try:
            memories = []
            
            if query.time_range:
                # Query temporal graph at specific time range
                nodes = self.graph.query_in_time_range(query.query_text, query.time_range)
                
                for node in nodes:
                    memory = Memory(
                        id=str(uuid.uuid4()),
                        content={'node': node.to_dict()},
                        timestamp=node.valid_from,
                        relevance_score=1.0,  # Temporal matches are exact
                        memory_type='temporal',
                        context={'query_time_range': str(query.time_range)}
                    )
                    memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"❌ Temporal search failed: {str(e)}")
            return []
    
    def _hybrid_memory_search(self, query: MemoryQuery) -> List[Memory]:
        """Perform hybrid search combining all methods."""
        logger.debug("🔍 Performing hybrid memory search")
        
        try:
            # Get results from all search types
            semantic_memories = self._semantic_memory_search(query)
            keyword_memories = self._keyword_memory_search(query)
            temporal_memories = self._temporal_memory_search(query)
            
            # Combine and rank results
            all_memories = semantic_memories + keyword_memories + temporal_memories
            
            # Simple ranking by relevance score (can be improved)
            all_memories.sort(key=lambda m: m.relevance_score, reverse=True)
            
            # Remove duplicates (simple implementation)
            unique_memories = []
            seen_content = set()
            
            for memory in all_memories:
                content_hash = str(hash(str(memory.content)))
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_memories.append(memory)
            
            return unique_memories
            
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {str(e)}")
            return []
    
    def _time_ranges_overlap(self, range1: TimeRange, range2: TimeRange) -> bool:
        """Check if two time ranges overlap."""        # Simple overlap check
        start1, end1 = range1.start, range1.end or datetime.now(timezone.utc)
        start2, end2 = range2.start, range2.end or datetime.now(timezone.utc)
        
        return start1 <= end2 and start2 <= end1
    def _get_events_for_node(self, node: Union[TemporalNode, Dict[str, Any]]) -> List[Event]:
        """Get events associated with a temporal node."""
        try:
            # Get node properties and type, handling both TemporalNode objects and dictionaries
            node_type = node['type'] if isinstance(node, dict) else node.type
            node_properties = node['properties'] if isinstance(node, dict) else node.properties
            node_valid_from = node['valid_from'] if isinstance(node, dict) else node.valid_from
            
            # Look for events with timestamp matching the node's valid_from
            # This is a simple approach - in production we'd want a more robust relationship
            with db_manager.get_session() as session:
                events = session.query(Event).filter(
                    Event.timestamp == node_valid_from
                ).all()
                
                # Convert to list of dictionaries to avoid session issues
                return events                
        except Exception as e:
            logger.error(f"❌ Failed to get events for node: {str(e)}")
            return []

    def _group_events_into_episodes(self, events: List[Event], customer_id: str) -> List[Episode]:
        """Group events into logical episodes."""
        logger.debug(f"📊 Grouping {len(events)} events into episodes")
        
        if not events:
            return []
        
        # Debug the events we received
        for i, event in enumerate(events):
            logger.debug(f"🔧 Event {i}: type={type(event)}, has_timestamp={hasattr(event, 'timestamp')}")
            if hasattr(event, 'timestamp'):
                logger.debug(f"🔧 Event {i} timestamp: {event.timestamp}")
            else:
                logger.error(f"❌ Event {i} missing timestamp attribute: {event}")
        
        # Simple episode grouping: group events within 1 hour of each other
        episodes = []
        current_episode_events = []
        episode_threshold = timedelta(hours=1)
        
        try:
            # Sort events by timestamp
            events_sorted = sorted(events, key=lambda x: x.timestamp)
        except AttributeError as e:
            logger.error(f"❌ Failed to sort events by timestamp: {e}")
            logger.error(f"❌ Events received: {[type(e) for e in events]}")
            return []
        
        for event in events_sorted:
            if not current_episode_events:
                current_episode_events = [event]
            else:
                last_event_time = current_episode_events[-1].timestamp
                if event.timestamp - last_event_time <= episode_threshold:
                    current_episode_events.append(event)
                else:
                    # Create episode from current events
                    if current_episode_events:
                        episode = self._create_episode_from_events(current_episode_events, customer_id)
                        episodes.append(episode)
                    current_episode_events = [event]
        
        # Don't forget the last episode
        if current_episode_events:
            episode = self._create_episode_from_events(current_episode_events, customer_id)
            episodes.append(episode)
        
        return episodes
    
    def _create_episode_from_events(self, events: List[Event], customer_id: str) -> Episode:
        """Create an Episode object from a list of events."""
        if not events:
            raise ValueError("Cannot create episode from empty events list")
        events.sort(key=lambda e: e.timestamp)
        
        # Create summary from event types
        event_types = [e.event_type for e in events]
        summary = f"Episode with {len(events)} events: {', '.join(set(event_types))}"
        
        # Create simple episode (entities are already stored in the graph)
        episode = Episode(
            id=str(uuid.uuid4()),
            customer_id=customer_id,
            events=events,
            start_time=events[0].timestamp,
            end_time=events[-1].timestamp,
            summary=summary,
            entities=[],  # Entities are already in the temporal graph
            relationships=[]  # Relationships are already in the temporal graph
        )
        
        return episode

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics for monitoring."""
        logger.debug("📊 Getting system stats")
        
        try:
            stats = {
                'total_nodes': 0,
                'total_edges': 0,
                'total_events': 0,
                'memory_store_status': 'active',
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            # Get stats from temporal graph if available
            if hasattr(self.graph, 'get_stats'):
                graph_stats = self.graph.get_stats()
                stats.update(graph_stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get system stats: {str(e)}")
            return {
                'error': str(e),
                'memory_store_status': 'error',
                'last_updated': datetime.now(timezone.utc).isoformat()
            }

    def query_memory(self, query, query_type: str = "hybrid", max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Query memory using the existing search engines.
        
        Args:
            query: Search query string or MemoryQuery object
            query_type: Type of search ('semantic', 'keyword', 'hybrid', 'temporal')
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        # Handle both string queries and MemoryQuery objects
        if hasattr(query, 'query_text'):  # MemoryQuery object
            query_text = query.query_text
            query_type = query.query_type
            max_results = query.max_results
            time_range = getattr(query, 'time_range', None)
        else:  # String query
            query_text = query
            time_range = None
            
        logger.info(f"🔍 Querying memory: '{query_text}' (type: {query_type})")        
        try:
            if query_type == "hybrid":
                # Use the existing hybrid search engine
                import asyncio
                results = asyncio.run(self.hybrid_search.search(
                    query=query_text,
                    limit=max_results,
                    enable_semantic=True,
                    enable_keyword=True,
                    enable_graph=False  # No source entity provided
                ))                # Convert to Memory objects
                memories = [self._convert_hybrid_result_to_memory(r) for r in results]            
            elif query_type == "temporal":
                # For temporal search, use the temporal graph directly
                if time_range:
                    node_dicts = self.graph.query_in_time_range(query_text, time_range)
                    memories = []
                    for node_dict in node_dicts:
                        memory = Memory(
                            id=str(node_dict['id']),
                            content={
                                'entity_type': node_dict['type'],
                                'identifier': node_dict['properties'].get('identifier', str(node_dict['id'])) if node_dict['properties'] else str(node_dict['id']),
                                'properties': node_dict['properties'] or {}
                            },
                            timestamp=node_dict['valid_from'],
                            relevance_score=1.0,
                            memory_type='temporal',
                            context={'valid_to': node_dict['valid_to']}
                        )
                        memories.append(memory)
                else:
                    logger.warning("Temporal search requires time_range parameter")
                    memories = []
            else:
                # For semantic and keyword searches
                import asyncio
                results = asyncio.run(self.hybrid_search.search(
                    query=query_text,
                    limit=max_results,
                    enable_semantic=(query_type == "semantic"),
                    enable_keyword=(query_type == "keyword"),
                    enable_graph=False
                ))
                memories = [self._convert_hybrid_result_to_memory(r) for r in results]
            logger.info(f"✅ Query completed: found {len(memories)} results")
            return memories[:max_results]
            
        except Exception as e:
            logger.error(f"❌ Query failed: {str(e)}")
            return []

    def _convert_hybrid_result_to_memory(self, result) -> Memory:
        """Convert hybrid search result to Memory object."""
        # Handle both dict and object results
        if isinstance(result, dict):
            node_id = result.get('node_id') or result.get('id')
            entity_type = result.get('entity_type') or result.get('type')
            identifier = result.get('identifier')
            properties = result.get('properties', {})
            valid_from = result.get('valid_from')
            combined_score = result.get('combined_score', result.get('relevance_score', 0.0))
        else:
            node_id = getattr(result, 'node_id', None) or getattr(result, 'id', None)
            entity_type = getattr(result, 'entity_type', None) or getattr(result, 'type', None)
            identifier = getattr(result, 'identifier', None)
            properties = getattr(result, 'properties', {})
            valid_from = getattr(result, 'valid_from', None)
            combined_score = getattr(result, 'combined_score', getattr(result, 'relevance_score', 0.0))
        
        return Memory(
            id=str(node_id),
            content={
                'entity_type': entity_type,
                'identifier': identifier,
                'properties': properties
            },
            timestamp=valid_from,
            relevance_score=combined_score,
            memory_type='entity',
            context={'search_type': 'hybrid'}
        )
    
    def _convert_temporal_node_to_memory(self, node) -> Memory:
        """Convert temporal node to Memory object."""
        return Memory(
            id=str(node.id),
            content={
                'entity_type': node.type,
                'properties': node.properties or {}
            },
            timestamp=node.valid_from,
            relevance_score=1.0,  # Temporal matches are exact
            memory_type='entity',
            context={'search_type': 'temporal'}
        )
        logger.error(f"❌ Keyword search failed: {str(e)}")
        return []
    
    def _temporal_search_on_entities(self, query_text: str, entities: List[Dict[str, Any]], 
                                   time_range, max_results: int) -> List:
        """Perform temporal search on entities from temporal graph."""
        try:
            results = []
            
            for entity in entities:
                valid_from = entity['valid_from']
                
                # Check if entity is within time range
                if time_range:
                    if valid_from < time_range.start or (time_range.end and valid_from > time_range.end):
                        continue
                
                # Simple text matching within time range
                similarity = self._calculate_text_similarity(query_text.lower(), entity['text'].lower())
                if similarity > 0.1:
                    memory = Memory(
                        id=entity['id'],
                        content=entity,
                        timestamp=valid_from,
                        relevance_score=similarity,
                        memory_type='entity',
                        context={'search_type': 'temporal', 'time_range': str(time_range) if time_range else None}
                    )
                    results.append(memory)
            
            # Sort by timestamp (most recent first)
            results.sort(key=lambda x: x.timestamp, reverse=True)
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"❌ Temporal search failed: {str(e)}")
            return []
    
    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """Simple text similarity calculation."""
        query_words = set(query.split())
        text_words = set(text.split())
        
        if not query_words:
            return 0.0
            
        intersection = query_words.intersection(text_words)
        return len(intersection) / len(query_words)
    
    def _convert_semantic_result_to_memory(self, result) -> Memory:
        """Convert SemanticSearchResult to Memory."""
        return Memory(
            id=result.node_id,
            content={
                'entity_type': result.entity_type,
                'identifier': result.identifier,
                'properties': result.properties
            },
            timestamp=result.valid_from,
            relevance_score=result.similarity_score,
            memory_type='semantic',
            context={'embedding_distance': result.embedding_distance}
        )
    
    def _convert_keyword_result_to_memory(self, result) -> Memory:
        """Convert KeywordSearchResult to Memory."""
        return Memory(
            id=result.node_id,
            content={
                'entity_type': result.entity_type,
                'identifier': result.identifier,
                'properties': result.properties
            },
            timestamp=result.valid_from,
            relevance_score=result.relevance_score,
            memory_type='keyword',
            context={'matched_fields': result.matched_fields}
        )
    
    def _convert_hybrid_result_to_memory(self, result) -> Memory:
        """Convert HybridSearchResult to Memory."""
        return Memory(
            id=result.node_id,
            content={
                'entity_type': result.entity_type,
                'identifier': result.identifier,
                'properties': result.properties
            },
            timestamp=result.valid_from,
            relevance_score=result.hybrid_score,
            memory_type='hybrid',            context={
                'semantic_score': result.semantic_score,
                'keyword_score': result.keyword_score,
                'graph_score': result.graph_score,
                'found_by': result.found_by
            }
        )

    def _convert_temporal_node_to_memory(self, node) -> Memory:
        """Convert TemporalNode to Memory."""
        return Memory(
            id=str(node.id),
            content={
                'entity_type': node.type,
                'identifier': node.properties.get('identifier', str(node.id)) if node.properties else str(node.id),
                'properties': node.properties or {}
            },
            timestamp=node.valid_from,
            relevance_score=1.0,
            memory_type='temporal',
            context={'valid_to': node.valid_to}
        )

    def get_existing_customer_ids(self) -> List[str]:
        """Get all unique customer IDs from the database."""
        logger.debug("Getting existing customer IDs from database")
        
        try:
            if self.external_session:
                return self._get_customer_ids_with_session(self.graph.db)
            else:
                with db_manager.get_session() as session:
                    return self._get_customer_ids_with_session(session)
        except Exception as e:
            logger.error(f"❌ Failed to get customer IDs: {str(e)}")
            return []

    def _get_customer_ids_with_session(self, session: Session) -> List[str]:
        """Get customer IDs using a database session."""
        try:
            # Query customer IDs from EVENTS table (where the actual customer activity is)
            event_customer_ids = session.query(
                Event.event_data['customer_id'].astext.distinct()
            ).filter(
                Event.event_data['customer_id'].astext.isnot(None)
            ).all()
            
            # Also get customer IDs from nodes table (entities)
            node_customer_ids = session.query(
                TemporalNode.properties['customer_id'].astext.distinct()
            ).filter(
                TemporalNode.properties['customer_id'].astext.isnot(None)
            ).all()
            
            # Combine both sources and remove duplicates
            all_customer_ids = set()
            for row in event_customer_ids:
                if row[0]:
                    all_customer_ids.add(row[0])
            for row in node_customer_ids:
                if row[0]:
                    all_customer_ids.add(row[0])
            
            customer_ids = list(all_customer_ids)
            logger.info(f"📊 Found {len(customer_ids)} unique customer IDs ({len(event_customer_ids)} from events, {len(node_customer_ids)} from nodes)")
            return customer_ids
            
        except Exception as e:
            logger.error(f"❌ Database query for customer IDs failed: {str(e)}")
            return []
