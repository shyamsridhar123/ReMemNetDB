"""
memory/temporal_graph.py - Temporal Knowledge Graph Engine for Step 3 Implementation
"""

import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple, Union
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, text, func

from ..core.models import TemporalNode, TemporalEdge, Event, NodeSchema, EdgeSchema
from ..core.database import db_manager

logger = logging.getLogger(__name__)


class TimeRange:
    """Represents a time range for temporal queries."""
    
    def __init__(self, start: datetime, end: Optional[datetime] = None):
        self.start = start
        self.end = end
    
    def __repr__(self):
        return f"TimeRange({self.start} to {self.end or 'now'})"


class TemporalGraph:
    """
    Temporal Knowledge Graph Engine (Step 3 Implementation)
    
    Manages bi-temporal data with both:
    - Valid time: when the fact was true in reality
    - Transaction time: when the fact was recorded in the database
    
    Implements the exact interface specified in the implementation plan.
    """
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize with optional database session."""
        if db_session:
            self.db = db_session
            self.external_session = True
        else:
            self.db_manager = db_manager
            self.external_session = False
            
    def add_event(self, event: Event, extracted_entities: Optional[List] = None) -> None:
        """
        Process and store event in temporal graph.
        
        This is the main entry point for adding new information to the graph.
        Implements the exact interface from the implementation plan.
        """
        logger.info(f"🔄 Processing event: {event.event_type} at {event.timestamp}")
        
        if self.external_session:
            self._process_event_with_session(self.db, event, extracted_entities)
        else:
            with self.db_manager.get_session() as session:
                self._process_event_with_session(session, event, extracted_entities)
    
    def _process_event_with_session(self, session: Session, event: Event, extracted_entities: Optional[List] = None) -> None:
        """Process event within a database session."""
        try:
            # Store the raw event
            session.add(event)
            session.flush()            # Use extracted entities if provided, otherwise extract from event
            logger.debug(f"🔍 extracted_entities parameter: {extracted_entities}")
            if extracted_entities:
                logger.debug(f"📊 Got {len(extracted_entities)} extracted entities")
                # Convert ExtractedEntity objects to the format expected by temporal graph
                entities = []                
                for entity in extracted_entities:
                    has_embedding = entity.embedding is not None
                    embedding_size = len(entity.embedding) if entity.embedding else 0
                    logger.debug(f"🔍 Processing entity: {entity.type} - {entity.identifier}, embedding: {has_embedding}, size: {embedding_size}")
                    entities.append({
                        'type': entity.type,
                        'identifier': entity.identifier,
                        'properties': entity.properties,
                        'embedding': entity.embedding  # This is the key - pass the embedding!
                    })
                logger.debug(f"📊 Using {len(entities)} pre-extracted entities with embeddings")
            else:
                # Fallback to old extraction method (without embeddings)
                entities = self._extract_entities_from_event(event)
                logger.debug(f"📊 Extracted {len(entities)} entities using fallback method")
            
            relationships = self._extract_relationships_from_event(event, entities)
            
            logger.debug(f"📊 Extracted {len(entities)} entities, {len(relationships)} relationships")
            
            # Store entities as temporal nodes
            node_map = {}
            for entity in entities:
                node = self._create_or_update_temporal_node(session, entity, event.timestamp)
                node_map[entity['identifier']] = node
            
            # Store relationships as temporal edges
            for relationship in relationships:
                if relationship['source'] in node_map and relationship['target'] in node_map:
                    self._create_temporal_edge(
                        session=session,
                        source_node=node_map[relationship['source']],
                        target_node=node_map[relationship['target']],
                        relationship_type=relationship['type'],
                        properties=relationship.get('properties', {}),
                        valid_from=event.timestamp
                    )
            
            if not self.external_session:
                session.commit()
                
            logger.info(f"✅ Successfully processed event with {len(entities)} entities and {len(relationships)} relationships")
            
        except Exception as e:
            logger.error(f"❌ Error processing event: {e}")
            if not self.external_session:
                session.rollback()
            raise
    
    def query_at_time(self, query: str, timestamp: datetime) -> List[TemporalNode]:
        """
        Query graph state at specific time.
        
        Implements the exact interface from the implementation plan.
        Args:
            query: Entity type or search criteria  
            timestamp: Time to query at
        Returns:
            List of nodes valid at the specified timestamp
        """
        logger.debug(f"🔍 Querying '{query}' at time {timestamp}")
        
        if self.external_session:
            return self._query_at_time_with_session(self.db, query, timestamp)
        else:
            with self.db_manager.get_session() as session:
                return self._query_at_time_with_session(session, query, timestamp)
    
    def _query_at_time_with_session(self, session: Session, query: str, timestamp: datetime) -> List[TemporalNode]:
        """Query implementation within a session."""
        # For now, treat query as entity type - can be enhanced later
        entity_type = query
        
        nodes = session.query(TemporalNode).filter(
            and_(
                TemporalNode.type == entity_type,
                TemporalNode.valid_from <= timestamp,
                or_(
                    TemporalNode.valid_to.is_(None),
                    TemporalNode.valid_to > timestamp
                )
            )
        ).all()
        
        logger.debug(f"📋 Found {len(nodes)} {entity_type} nodes valid at {timestamp}")
        return nodes
    
    def query_in_time_range(self, query: str, time_range: TimeRange) -> List[Dict[str, Any]]:
        """
        Query graph state within a time range.
        
        Args:
            query: Entity type or search criteria  
            time_range: Time range to query within
        Returns:
            List of nodes valid within the specified time range
        """
        logger.debug(f"🔍 Querying '{query}' in time range {time_range}")
        
        if self.external_session:
            return self._query_in_time_range_with_session(self.db, query, time_range)
        else:
            with self.db_manager.get_session() as session:
                return self._query_in_time_range_with_session(session, query, time_range)
    def _query_in_time_range_with_session(self, session: Session, query: str, time_range: TimeRange) -> List[Dict[str, Any]]:
        """Query implementation within a session for time range."""
        # For now, treat query as entity type - can be enhanced later
        entity_type = query
        
        # Build time range filters
        filters = [TemporalNode.type == entity_type]
        
        # Node must be valid during at least part of the time range
        if time_range.end:
            # Time range has an end - node must start before range ends and end after range starts (or still be valid)
            filters.append(TemporalNode.valid_from < time_range.end)
            filters.append(
                or_(
                    TemporalNode.valid_to.is_(None),
                    TemporalNode.valid_to > time_range.start
                )
            )
        else:
            # Time range is open-ended - node must start before now and still be valid or end after range start
            filters.append(
                or_(
                    TemporalNode.valid_to.is_(None),
                    TemporalNode.valid_to > time_range.start
                )
            )
        
        nodes = session.query(TemporalNode).filter(and_(*filters)).all()
        
        # Convert to dictionaries within the session to avoid detachment issues
        node_dicts = []
        for node in nodes:
            node_dict = {
                'id': node.id,
                'type': node.type,
                'properties': node.properties,
                'valid_from': node.valid_from,
                'valid_to': node.valid_to,
                'recorded_at': node.recorded_at,
                'embedding': node.embedding
            }
            node_dicts.append(node_dict)
        
        logger.debug(f"📋 Found {len(node_dicts)} {entity_type} nodes valid in time range {time_range}")
        return node_dicts
    
    def get_entity_history(self, entity_id: str) -> List[Tuple[TemporalNode, TimeRange]]:
        """
        Get complete history of entity changes.
        
        Implements the exact interface from the implementation plan.
        Args:
            entity_id: Entity identifier (not UUID, but business identifier)
        Returns:
            List of (node, time_range) tuples showing entity evolution
        """
        logger.debug(f"📜 Getting history for entity: {entity_id}")
        
        if self.external_session:
            return self._get_entity_history_with_session(self.db, entity_id)
        else:
            with self.db_manager.get_session() as session:
                return self._get_entity_history_with_session(session, entity_id)
    
    def _get_entity_history_with_session(self, session: Session, entity_id: str) -> List[Tuple[TemporalNode, TimeRange]]:
        """Get entity history implementation within a session."""
        # Find all node versions for this entity identifier
        nodes = session.query(TemporalNode).filter(
            TemporalNode.properties['identifier'].astext == entity_id
        ).order_by(TemporalNode.valid_from).all()
        
        history = []
        for node in nodes:
            time_range = TimeRange(node.valid_from, node.valid_to)
            history.append((node, time_range))
        
        logger.debug(f"📊 Found {len(history)} versions in entity history")
        return history    
    def traverse_graph(self, start_node_id: str, relationship_types: List[str], 
                      max_depth: int = 2, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Traverse the graph from a starting node following specific relationship types.
        
        Enhanced graph traversal for relationship analysis.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        logger.debug(f"🕸️ Traversing graph from {start_node_id}, depth {max_depth}, types {relationship_types}")
        
        if self.external_session:
            return self._traverse_graph_with_session(self.db, start_node_id, relationship_types, max_depth, timestamp)
        else:
            with self.db_manager.get_session() as session:
                return self._traverse_graph_with_session(session, start_node_id, relationship_types, max_depth, timestamp)
    
    def _traverse_graph_with_session(self, session: Session, start_node_id: str, relationship_types: List[str],
                                   max_depth: int, timestamp: datetime) -> Dict[str, Any]:
        """Graph traversal implementation within a session."""
        visited = set()
        result = {
            'nodes': {},
            'edges': [],
            'paths': []
        }
        
        def _traverse_recursive(node_id: str, depth: int, path: List[str]):
            if depth > max_depth or node_id in visited:
                return
            
            visited.add(node_id)
            
            # Get the node
            node = session.query(TemporalNode).filter(TemporalNode.id == node_id).first()
            if node:
                result['nodes'][node_id] = {
                    'id': str(node.id),
                    'type': node.type,
                    'properties': node.properties,
                    'valid_from': node.valid_from.isoformat(),
                    'valid_to': node.valid_to.isoformat() if node.valid_to else None
                }
            
            # Get outgoing relationships
            edges = self._get_relationships_at_time_with_session(session, node_id, timestamp, 'outgoing')
            
            for edge in edges:
                if edge.relationship_type in relationship_types:
                    result['edges'].append({
                        'id': str(edge.id),
                        'source': str(edge.source_node_id),
                        'target': str(edge.target_node_id),
                        'type': edge.relationship_type,
                        'properties': edge.properties
                    })
                    
                    new_path = path + [str(edge.target_node_id)]
                    result['paths'].append(new_path)
                    
                    # Recurse to target node
                    _traverse_recursive(str(edge.target_node_id), depth + 1, new_path)
        
        _traverse_recursive(start_node_id, 0, [start_node_id])
        
        logger.debug(f"🔍 Graph traversal found {len(result['nodes'])} nodes, {len(result['edges'])} edges")
        return result
    
    def _get_relationships_at_time_with_session(self, session: Session, node_id: str, timestamp: datetime,
                                              direction: str = 'both') -> List[TemporalEdge]:
        """Get relationships for a node at a specific time within a session."""
        query = session.query(TemporalEdge).filter(
            and_(
                TemporalEdge.valid_from <= timestamp,
                or_(
                    TemporalEdge.valid_to.is_(None),
                    TemporalEdge.valid_to > timestamp
                )
            )
        )
        
        if direction == 'outgoing':
            query = query.filter(TemporalEdge.source_node_id == node_id)
        elif direction == 'incoming':
            query = query.filter(TemporalEdge.target_node_id == node_id)
        else:  # both
            query = query.filter(
                or_(
                    TemporalEdge.source_node_id == node_id,
                    TemporalEdge.target_node_id == node_id
                )
            )
        
        return query.all()
    
    def _extract_entities_from_event(self, event: Event) -> List[Dict[str, Any]]:
        """
        Extract entities from event data.
        
        This implements basic entity extraction as specified in Step 3.
        Will be enhanced with LLM extraction in the EntityExtractor integration.
        """
        entities = []
        event_data = event.event_data
        
        logger.debug(f"🔍 Extracting entities from {event.event_type} event")
        
        # Extract based on event type
        if event.event_type == "order_placed":
            # Customer entity
            if "customer_name" in event_data:
                entities.append({
                    'type': 'customer',
                    'identifier': event_data['customer_name'],
                    'properties': {
                        'identifier': event_data['customer_name'],
                        'name': event_data['customer_name'],
                        'source_event': event.event_type,
                        'extraction_confidence': 0.95
                    }
                })
            
            # Product entity
            if "product_name" in event_data:
                entities.append({
                    'type': 'product',
                    'identifier': event_data['product_name'],
                    'properties': {
                        'identifier': event_data['product_name'],
                        'name': event_data['product_name'],
                        'category': event_data.get('category', 'Unknown'),
                        'description': event_data.get('description', ''),
                        'source_event': event.event_type,
                        'extraction_confidence': 0.95
                    }
                })
            
            # Order entity
            order_id = f"order_{event.timestamp.isoformat()}_{event_data.get('customer_name', 'unknown')}"
            entities.append({
                'type': 'order',
                'identifier': order_id,
                'properties': {
                    'identifier': order_id,
                    'order_id': order_id,
                    'customer_name': event_data.get('customer_name'),
                    'product_name': event_data.get('product_name'),
                    'order_timestamp': event.timestamp.isoformat(),
                    'source_event': event.event_type,
                    'extraction_confidence': 0.90
                }
            })
        
        elif event.event_type == "product_review":
            # Customer entity
            if "customer_name" in event_data:
                entities.append({
                    'type': 'customer',
                    'identifier': event_data['customer_name'],
                    'properties': {
                        'identifier': event_data['customer_name'],
                        'name': event_data['customer_name'],
                        'source_event': event.event_type,
                        'extraction_confidence': 0.95
                    }
                })
            
            # Product entity
            if "product_name" in event_data:
                entities.append({
                    'type': 'product',
                    'identifier': event_data['product_name'],
                    'properties': {
                        'identifier': event_data['product_name'],
                        'name': event_data['product_name'],
                        'category': event_data.get('category', 'Unknown'),
                        'source_event': event.event_type,
                        'extraction_confidence': 0.95
                    }
                })
                
            # Review entity
            review_id = f"review_{event_data.get('customer_name', 'unknown')}_{event_data.get('product_name', 'unknown')}_{event.timestamp.isoformat()}"
            entities.append({
                'type': 'review',
                'identifier': review_id,
                'properties': {
                    'identifier': review_id,
                    'review_id': review_id,
                    'customer_name': event_data.get('customer_name'),
                    'product_name': event_data.get('product_name'),
                    'rating': event_data.get('rating'),
                    'review_text': event_data.get('review', ''),
                    'review_timestamp': event.timestamp.isoformat(),
                    'source_event': event.event_type,
                    'extraction_confidence': 0.90
                }
            })
        
        logger.debug(f"📊 Extracted {len(entities)} entities from {event.event_type} event")
        return entities
    
    def _extract_relationships_from_event(self, event: Event, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities from event data.
        
        This implements relationship detection as specified in Step 3.
        """
        relationships = []
        event_data = event.event_data
        
        logger.debug(f"🔗 Extracting relationships from {event.event_type} event")
        
        # Create entity lookup by type
        entity_map = {}
        for entity in entities:
            entity_type = entity['type']
            if entity_type not in entity_map:
                entity_map[entity_type] = []
            entity_map[entity_type].append(entity['identifier'])
        
        if event.event_type == "order_placed":
            # Customer -> Product (purchased)
            if 'customer' in entity_map and 'product' in entity_map:
                relationships.append({
                    'source': entity_map['customer'][0],
                    'target': entity_map['product'][0],
                    'type': 'purchased',
                    'properties': {
                        'relationship_timestamp': event.timestamp.isoformat(),
                        'order_id': entity_map.get('order', [None])[0],
                        'extraction_confidence': 0.95
                    }
                })
            
            # Customer -> Order (placed)
            if 'customer' in entity_map and 'order' in entity_map:
                relationships.append({
                    'source': entity_map['customer'][0],
                    'target': entity_map['order'][0],
                    'type': 'placed_order',
                    'properties': {
                        'relationship_timestamp': event.timestamp.isoformat(),
                        'extraction_confidence': 0.95
                    }
                })
            
            # Order -> Product (contains)
            if 'order' in entity_map and 'product' in entity_map:
                relationships.append({
                    'source': entity_map['order'][0],
                    'target': entity_map['product'][0],
                    'type': 'contains_product',
                    'properties': {
                        'relationship_timestamp': event.timestamp.isoformat(),
                        'extraction_confidence': 0.90
                    }
                })
        
        elif event.event_type == "product_review":
            # Customer -> Product (reviewed)
            if 'customer' in entity_map and 'product' in entity_map:
                relationships.append({
                    'source': entity_map['customer'][0],
                    'target': entity_map['product'][0],
                    'type': 'reviewed_product',
                    'properties': {
                        'relationship_timestamp': event.timestamp.isoformat(),
                        'rating': event_data.get('rating'),
                        'extraction_confidence': 0.95
                    }
                })
            
            # Customer -> Review (authored)
            if 'customer' in entity_map and 'review' in entity_map:
                relationships.append({
                    'source': entity_map['customer'][0],
                    'target': entity_map['review'][0],
                    'type': 'authored_review',
                    'properties': {
                        'relationship_timestamp': event.timestamp.isoformat(),
                        'extraction_confidence': 0.95
                    }
                })
            
            # Review -> Product (about)
            if 'review' in entity_map and 'product' in entity_map:
                relationships.append({
                    'source': entity_map['review'][0],
                    'target': entity_map['product'][0],
                    'type': 'review_about_product',
                    'properties': {
                        'relationship_timestamp': event.timestamp.isoformat(),
                        'extraction_confidence': 0.90
                    }
                })
        
        logger.debug(f"🔗 Extracted {len(relationships)} relationships from {event.event_type} event")
        return relationships
    
    def _create_or_update_temporal_node(self, session: Session, entity: Dict[str, Any], valid_from: datetime) -> TemporalNode:
        """
        Create a new temporal node or update existing one.
        
        Handles temporal versioning by closing previous version and creating new one.
        """
        identifier = entity['identifier']
        entity_type = entity['type']
        
        logger.debug(f"📝 Creating/updating temporal node: {entity_type} - {identifier}")
        
        # Debug: Check embedding
        embedding = entity.get('embedding')
        has_embedding = embedding is not None
        embedding_size = len(embedding) if embedding else 0
        logger.debug(f"🔍 Entity embedding info: has_embedding={has_embedding}, size={embedding_size}")
        
        # Check if node already exists
        existing_node = session.query(TemporalNode).filter(
            and_(
                TemporalNode.type == entity_type,                TemporalNode.properties['identifier'].astext == identifier,
                TemporalNode.valid_to.is_(None)  # Currently valid
            )
        ).first()
        
        if existing_node:
            # Check if properties have changed
            if existing_node.properties != entity['properties']:
                # Close the existing node
                existing_node.valid_to = valid_from
                
                # Create new version with embedding
                new_node = TemporalNode(
                    type=entity_type,
                    properties=entity['properties'],
                    valid_from=valid_from,
                    embedding=entity.get('embedding')  # Include embedding if available
                )
                session.add(new_node)
                session.flush()
                
                logger.debug(f"✏️ Updated temporal node: {identifier} (new version)")
                return new_node
            else:
                logger.debug(f"📋 No changes for temporal node: {identifier}")
                return existing_node
        else:
            # Create new node with embedding
            new_node = TemporalNode(
                type=entity_type,
                properties=entity['properties'],
                valid_from=valid_from,
                embedding=entity.get('embedding')  # Include embedding if available
            )
            session.add(new_node)
            session.flush()
            
            logger.debug(f"✨ Created new temporal node: {identifier}")
            return new_node
    
    def _create_temporal_edge(self, session: Session, source_node: TemporalNode, target_node: TemporalNode,
                            relationship_type: str, properties: Dict[str, Any], valid_from: datetime) -> TemporalEdge:
        """
        Create a temporal edge between two nodes.
        """
        edge = TemporalEdge(
            source_node_id=source_node.id,
            target_node_id=target_node.id,
            relationship_type=relationship_type,
            properties=properties,
            valid_from=valid_from
        )
        
        session.add(edge)
        session.flush()
        
        logger.debug(f"🔗 Created temporal edge: {source_node.properties.get('identifier')} --{relationship_type}--> {target_node.properties.get('identifier')}")
        return edge
        
    # Keep existing methods for backward compatibility
        """Add an event to the temporal graph."""
        with self.db_manager.get_session() as session:
            event = Event(
                event_type=event_data.get("type", "unknown"),
                event_data=event_data,
                timestamp=timestamp
            )
            session.add(event)
            session.flush()
            
            # Process event to extract entities and relationships
            self._process_event(session, event)
            
            return str(event.id)
    
    def _process_event(self, session: Session, event: Event) -> None:
        """Process event to extract entities and relationships."""
        # Basic event processing - in a real implementation, this would use LLM
        event_type = event.event_type
        event_data = event.event_data
        
        if event_type == "customer_action":
            self._process_customer_action(session, event_data, event.timestamp)
        elif event_type == "product_update":
            self._process_product_update(session, event_data, event.timestamp)
        elif event_type == "order_placed":
            self._process_order(session, event_data, event.timestamp)
            
        # Mark event as processed
        event.processed = True
    
    def _process_customer_action(self, session: Session, data: Dict[str, Any], timestamp: datetime) -> None:
        """Process customer-related events."""
        customer_id = data.get("customer_id")
        action = data.get("action")
        
        if customer_id and action:
            # Create or update customer node
            self.insert_or_update_node(
                session,
                node_type="Customer",
                properties={"id": customer_id, "last_action": action},
                valid_from=timestamp
            )
    
    def _process_product_update(self, session: Session, data: Dict[str, Any], timestamp: datetime) -> None:
        """Process product-related events."""
        product_id = data.get("product_id")
        if product_id:
            self.insert_or_update_node(
                session,
                node_type="Product", 
                properties=data,
                valid_from=timestamp
            )
    
    def _process_order(self, session: Session, data: Dict[str, Any], timestamp: datetime) -> None:
        """Process order events and create relationships."""
        customer_id = data.get("customer_id")
        product_ids = data.get("product_ids", [])
        
        if customer_id and product_ids:
            # Find customer node
            customer_node = self.get_node_by_property(session, "Customer", "id", customer_id, timestamp)
            
            for product_id in product_ids:
                # Find product node
                product_node = self.get_node_by_property(session, "Product", "id", product_id, timestamp)
                
                if customer_node and product_node:
                    # Create PURCHASED relationship
                    self.insert_edge(
                        session,
                        source_node_id=customer_node.id,
                        target_node_id=product_node.id,
                        relationship_type="PURCHASED",
                        properties={"order_data": data},
                        valid_from=timestamp
                    )
    
    def insert_node(
        self, 
        session: Session,
        node_type: str, 
        properties: Dict[str, Any], 
        valid_from: datetime,
        valid_to: Optional[datetime] = None,
        embedding: Optional[List[float]] = None
    ) -> TemporalNode:
        """Insert a new temporal node."""
        node = TemporalNode(
            type=node_type,
            properties=properties,
            valid_from=valid_from,
            valid_to=valid_to,
            embedding=embedding
        )
        session.add(node)
        session.flush()
        return node
    
    def insert_edge(
        self,
        session: Session,
        source_node_id: uuid.UUID,
        target_node_id: uuid.UUID,
        relationship_type: str,
        properties: Dict[str, Any],
        valid_from: datetime,
        valid_to: Optional[datetime] = None
    ) -> TemporalEdge:
        """Insert a new temporal edge."""
        edge = TemporalEdge(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_type=relationship_type,
            properties=properties,
            valid_from=valid_from,
            valid_to=valid_to
        )
        session.add(edge)
        session.flush()
        return edge
    
    def insert_or_update_node(
        self,
        session: Session,
        node_type: str,
        properties: Dict[str, Any],
        valid_from: datetime,
        embedding: Optional[List[float]] = None
    ) -> TemporalNode:
        """Insert or update a node with temporal versioning."""
        # Check if node exists with same entity identifier
        entity_id = properties.get("id")
        if entity_id:
            existing_node = self.get_node_by_property(session, node_type, "id", entity_id, valid_from)
            if existing_node:
                return self.update_node(session, existing_node.id, properties, valid_from, embedding)
        
        return self.insert_node(session, node_type, properties, valid_from, embedding=embedding)
    
    def update_node(
        self,
        session: Session,
        node_id: uuid.UUID,
        new_properties: Dict[str, Any],
        effective_from: datetime,
        new_embedding: Optional[List[float]] = None
    ) -> TemporalNode:
        """Update a node with temporal versioning using database function."""
        # Use the database function we created earlier
        result = session.execute(
            text("SELECT temporal_graph.update_node(:node_id, :props, :effective_from, :embedding)"),
            {
                "node_id": node_id,
                "props": new_properties,
                "effective_from": effective_from,
                "embedding": new_embedding
            }
        ).scalar()
        
        # Return the new node
        return session.query(TemporalNode).filter(TemporalNode.id == result).first()
    
    def query_at_time(self, entity_type: str, timestamp: datetime) -> List[NodeSchema]:
        """Query graph state at specific time."""
        with self.db_manager.get_session() as session:
            nodes = session.query(TemporalNode).filter(
                and_(
                    TemporalNode.type == entity_type,
                    TemporalNode.valid_from <= timestamp,
                    or_(TemporalNode.valid_to.is_(None), TemporalNode.valid_to > timestamp)
                )
            ).all()
            
            return [NodeSchema.from_attributes(node) for node in nodes]
    
    def get_node_by_property(
        self, 
        session: Session,
        node_type: str, 
        property_name: str, 
        property_value: Any,
        as_of_time: Optional[datetime] = None
    ) -> Optional[TemporalNode]:
        """Get a node by a specific property value."""
        query = session.query(TemporalNode).filter(
            and_(
                TemporalNode.type == node_type,
                TemporalNode.properties[property_name].astext == str(property_value)
            )
        )
        
        if as_of_time:
            query = query.filter(
                and_(
                    TemporalNode.valid_from <= as_of_time,
                    or_(TemporalNode.valid_to.is_(None), TemporalNode.valid_to > as_of_time)
                )
            )
        
        return query.first()
    
    def get_entity_history(self, entity_id: uuid.UUID) -> List[Tuple[NodeSchema, datetime, Optional[datetime]]]:
        """Get complete history of entity changes."""
        with self.db_manager.get_session() as session:
            nodes = session.query(TemporalNode).filter(
                TemporalNode.id == entity_id
            ).order_by(TemporalNode.valid_from).all()
            
            return [
                (NodeSchema.from_attributes(node), node.valid_from, node.valid_to) 
                for node in nodes
            ]
    
    def get_neighbors_at_time(
        self, 
        node_id: uuid.UUID, 
        timestamp: datetime,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get neighbors of a node at a specific point in time."""
        with self.db_manager.get_session() as session:
            # Use the database function we created
            result = session.execute(
                text("SELECT * FROM temporal_graph.get_neighbors_at_time(:node_id, :timestamp)"),
                {"node_id": node_id, "timestamp": timestamp}
            ).fetchall()
            
            neighbors = []
            for row in result:
                if not relationship_types or row.relationship_type in relationship_types:
                    neighbors.append({
                        "edge_id": row.edge_id,
                        "neighbor_id": row.neighbor_id,
                        "relationship_type": row.relationship_type,
                        "direction": row.direction,
                        "properties": row.properties
                    })
            
            return neighbors
    
    def semantic_search(
        self,
        query_embedding: List[float],
        threshold: float = 0.7,
        limit: int = 10,
        as_of_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity."""
        with self.db_manager.get_session() as session:
            if as_of_time is None:
                as_of_time = datetime.utcnow()
            
            # Use the database function we created
            result = session.execute(
                text("SELECT * FROM temporal_graph.semantic_search(:embedding, :threshold, :limit, :as_of_time)"),
                {
                    "embedding": query_embedding,
                    "threshold": threshold,
                    "limit": limit,
                    "as_of_time": as_of_time
                }
            ).fetchall()
            
            return [
                {
                    "node_id": row.node_id,
                    "node_type": row.node_type,
                    "properties": row.properties,
                    "similarity": row.similarity
                }
                for row in result
            ]
    
    def find_contradictions(
        self, 
        entity_id: uuid.UUID, 
        property_name: str
    ) -> List[Dict[str, Any]]:
        """Find contradicting facts for a given entity."""
        with self.db_manager.get_session() as session:
            # Use the database function we created
            result = session.execute(
                text("SELECT * FROM temporal_graph.find_contradictions(:entity_id, :property_name)"),
                {"entity_id": entity_id, "property_name": property_name}
            ).fetchall()
            
            return [
                {
                    "valid_from": row.valid_from,
                    "valid_to": row.valid_to,
                    "property_value": row.property_value,
                    "recorded_at": row.recorded_at
                }
                for row in result
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics for the temporal graph."""
        with self.db_manager.get_session() as session:
            try:
                # Count nodes, edges, and events
                node_count = session.execute(text("SELECT COUNT(*) FROM temporal_graph.nodes")).scalar()
                edge_count = session.execute(text("SELECT COUNT(*) FROM temporal_graph.edges")).scalar()
                event_count = session.execute(text("SELECT COUNT(*) FROM temporal_graph.events")).scalar()
                
                return {
                    'total_nodes': node_count,
                    'total_edges': edge_count,
                    'total_events': event_count,
                    'database_stats': {
                        'total_nodes': {'row_count': node_count, 'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                        'total_edges': {'row_count': edge_count, 'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                        'total_events': {'row_count': event_count, 'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    }
                }
            except Exception as e:
                logger.error(f"❌ Failed to get database stats: {str(e)}")
                return {
                    'total_nodes': 0,
                    'total_edges': 0,
                    'total_events': 0,
                    'error': str(e)
                }
