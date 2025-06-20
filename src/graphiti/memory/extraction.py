"""
memory/extraction.py - Part of Graphiti E-commerce Agent Memory Platform

LLM-based entity and relationship extraction pipeline for temporal knowledge graphs.
Extracts entities and relationships from raw text/events and converts them into 
temporal nodes and edges with embeddings.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from uuid import uuid4

import openai
import tiktoken
import numpy as np
from pydantic import BaseModel, Field

from ..core.config import get_settings
from ..core.models import NodeSchema, EdgeSchema, EventSchema

# Set up logging
logger = logging.getLogger(__name__)

# Entity extraction prompts
ENTITY_EXTRACTION_PROMPT = """
You are an expert at extracting entities and their properties from e-commerce events.

Extract entities from the following event data. For each entity, provide:
1. Type (customer, product, order, review, support_ticket, etc.)
2. Identifier (unique ID or name)
3. Properties (key-value pairs of relevant attributes)
4. Confidence score (0-1)

Event Type: {event_type}
Event Data: {event_data}
Timestamp: {timestamp}

Return your response as a JSON list of entities:
[
  {{
    "type": "customer",
    "identifier": "customer_123",
    "properties": {{
      "name": "Alice Johnson",
      "email": "alice@example.com",
      "segment": "premium"
    }},
    "confidence": 0.95
  }},
  ...
]

Focus on extracting meaningful business entities. Be conservative - only include entities you're confident about.
"""

RELATIONSHIP_EXTRACTION_PROMPT = """
You are an expert at extracting relationships between entities from e-commerce events.

Given these entities extracted from an event, identify relationships between them:

Entities: {entities}
Event Type: {event_type}
Event Data: {event_data}
Timestamp: {timestamp}

Return relationships as a JSON list:
[
  {{
    "source_entity": "customer_123",
    "target_entity": "product_456",
    "relationship_type": "purchased",
    "properties": {{
      "quantity": 2,
      "price": 59.99,
      "channel": "web"
    }},
    "confidence": 0.9
  }},
  ...
]

Common e-commerce relationship types:
- purchased, viewed, reviewed, returned
- belongs_to (category), similar_to (product)
- contacted_support, recommended, wishlisted

Focus on meaningful business relationships. Include temporal and transactional details in properties.
"""

@dataclass
class ExtractedEntity:
    """Represents an entity extracted from raw data."""
    type: str
    identifier: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_event_id: Optional[str] = None
    embedding: Optional[List[float]] = None  # Add embedding field
    
    def to_node_schema(self, valid_from: datetime, embedding: Optional[List[float]] = None) -> NodeSchema:
        """Convert to NodeSchema for database storage."""
        # Add metadata to properties
        props = self.properties.copy()
        props.update({
            "identifier": self.identifier,
            "confidence": self.confidence,
            "extraction_source": self.source_event_id,
            "extracted_at": datetime.now(timezone.utc).isoformat()
        })
        
        return NodeSchema(
            type=self.type,
            properties=props,
            valid_from=valid_from,
            embedding=embedding
        )

@dataclass
class ExtractedRelationship:
    """Represents a relationship extracted from raw data."""
    source_entity: str
    target_entity: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_event_id: Optional[str] = None
    
    def to_edge_schema(self, source_node_id: str, target_node_id: str, 
                      valid_from: datetime) -> EdgeSchema:
        """Convert to EdgeSchema for database storage."""
        # Add metadata to properties
        props = self.properties.copy()
        props.update({
            "confidence": self.confidence,
            "extraction_source": self.source_event_id,
            "extracted_at": datetime.now(timezone.utc).isoformat()
        })
        
        return EdgeSchema(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_type=self.relationship_type,
            properties=props,
            valid_from=valid_from
        )

class ExtractionResult(BaseModel):
    """Result of entity/relationship extraction."""
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    processing_time: float = 0.0
    token_usage: Dict[str, int] = Field(default_factory=dict)

class EntityExtractor:
    """LLM-based entity and relationship extraction engine."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._client = None
        self._tokenizer = None
        
        # Initialize token counting
        try:
            self._tokenizer = tiktoken.encoding_for_model(self.settings.openai_model)
        except Exception as e:
            logger.warning(f"Could not load tokenizer for {self.settings.openai_model}: {e}")
            self._tokenizer = tiktoken.get_encoding("cl100k_base")  # fallback
    
    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            if self.settings.use_azure_openai:
                self._client = openai.AzureOpenAI(
                    api_key=self.settings.azure_openai_api_key,
                    api_version=self.settings.azure_openai_api_version,
                    azure_endpoint=self.settings.azure_openai_endpoint
                )
            else:
                self._client = openai.OpenAI(
                    api_key=self.settings.openai_api_key
                )
        return self._client    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer is None:
            return len(text) // 4  # rough estimate
        return len(self._tokenizer.encode(text))
    
    async def extract_entities(self, event: EventSchema) -> List[ExtractedEntity]:
        """Extract entities from a single event."""
        print(f"🔧 DEBUG EntityExtractor: extract_entities called with event_type={event.event_type}")
        logger.info(f"🔧 EntityExtractor: Starting entity extraction for {event.event_type}")
        logger.debug(f"🔧 EntityExtractor: Event data: {event.event_data}")
        
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            event_type=event.event_type,
            event_data=json.dumps(event.event_data, indent=2),
            timestamp=event.timestamp.isoformat()
        )
        
        print(f"🔧 DEBUG EntityExtractor: About to call OpenAI API...")
        logger.debug(f"🔧 EntityExtractor: Prompt length: {len(prompt)} chars")
        logger.debug(f"🔧 EntityExtractor: Full prompt: {prompt}")
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.settings.azure_openai_gpt4_deployment_name if self.settings.use_azure_openai else self.settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert entity extraction system for e-commerce data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.settings.openai_temperature,
                max_tokens=self.settings.openai_max_tokens // 2  # Reserve tokens for relationships
            )
            content = response.choices[0].message.content.strip()
            logger.debug(f"🔧 EntityExtractor: Raw OpenAI response: {content}")
            
            # Clean the JSON response (remove markdown code blocks if present)
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.endswith('```'):
                content = content[:-3]  # Remove ```
            content = content.strip()
            
            # Parse JSON response
            try:
                logger.debug(f"Entity parsing JSON content: {repr(content)}")
                entities_data = json.loads(content)
                entities = []
                
                for entity_data in entities_data:
                    try:                        # Handle None/null identifiers properly
                        identifier = entity_data.get("identifier")
                        if identifier is None or identifier == "null":
                            # Generate a fallback identifier based on entity type and timestamp
                            identifier = f"{entity_data.get('type', 'entity')}_{uuid4().hex[:8]}"
                        
                        entity = ExtractedEntity(
                            type=entity_data.get("type", "unknown"),
                            identifier=identifier,
                            properties=entity_data.get("properties", {}),
                            confidence=entity_data.get("confidence", 0.5),
                            source_event_id=str(event.id) if event.id else None
                        )
                        
                        # Enhanced customer entity handling
                        if entity.type == "customer":
                            # Always ensure identifier is in properties
                            if "identifier" not in entity.properties:
                                entity.properties["identifier"] = entity.identifier
                            # Also ensure customer_id is set for better retrieval
                            if "customer_id" not in entity.properties and event.event_data.get("customer_id"):
                                entity.properties["customer_id"] = event.event_data.get("customer_id")
                        
                        # Generate embedding for the entity
                        entity_text = f"{entity.type}: {entity.identifier}"
                        if entity.properties:
                            entity_text += f" - {json.dumps(entity.properties)}"
                        
                        print(f"🔧 DEBUG: Generating embedding for entity: {entity_text[:100]}...")
                        logger.debug(f"🔧 EntityExtractor: Entity text for embedding: {entity_text}")
                        
                        entity.embedding = await self._generate_embedding(entity_text)
                        print(f"🔧 DEBUG: Generated embedding size: {len(entity.embedding) if entity.embedding else 0}")
                        
                        if entity.embedding:
                            logger.debug(f"🔧 EntityExtractor: Successfully generated embedding for {entity.type} entity")
                        else:
                            logger.warning(f"🔧 EntityExtractor: Failed to generate embedding for {entity.type} entity")
                        
                        entities.append(entity)
                        logger.debug(f"🔧 EntityExtractor: Added entity: {entity.type} - {entity.identifier}")
                        
                    except Exception as entity_error:
                        logger.error(f"🔧 EntityExtractor: Failed to process individual entity: {entity_error}")
                        logger.debug(f"🔧 EntityExtractor: Problematic entity data: {entity_data}")
                        continue
                
                logger.info(f"Extracted {len(entities)} entities from event {event.id}")
                return entities
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse entity extraction response: {e}")
                logger.debug(f"Raw response: {content}")
                return []
                
        except Exception as e:
            logger.error(f"Entity extraction failed for event {event.id}: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return []
    
    async def extract_relationships(self, entities: List[ExtractedEntity], 
                                  event: EventSchema) -> List[ExtractedRelationship]:
        """Extract relationships between entities from an event."""
        if len(entities) < 2:
            return []  # Need at least 2 entities for relationships
        
        # Prepare entities for prompt
        entities_json = [
            {
                "type": e.type,
                "identifier": e.identifier,
                "properties": e.properties
            }
            for e in entities
        ]
        
        prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
            entities=json.dumps(entities_json, indent=2),
            event_type=event.event_type,
            event_data=json.dumps(event.event_data, indent=2),
            timestamp=event.timestamp.isoformat()
        )
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.settings.azure_openai_gpt4_deployment_name if self.settings.use_azure_openai else self.settings.openai_model,                messages=[
                    {"role": "system", "content": "You are an expert relationship extraction system for e-commerce data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.settings.openai_temperature,
                max_tokens=self.settings.openai_max_tokens // 2
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean the JSON response (remove markdown code blocks if present)
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.endswith('```'):
                content = content[:-3]  # Remove ```
            content = content.strip()
              # Parse JSON response
            try:
                logger.debug(f"Relationship parsing JSON content: {repr(content)}")
                relationships_data = json.loads(content)
                relationships = []
                
                for rel_data in relationships_data:
                    relationship = ExtractedRelationship(
                        source_entity=rel_data.get("source_entity", ""),
                        target_entity=rel_data.get("target_entity", ""),
                        relationship_type=rel_data.get("relationship_type", "related_to"),
                        properties=rel_data.get("properties", {}),
                        confidence=rel_data.get("confidence", 0.5),
                        source_event_id=str(event.id) if event.id else None
                    )
                    relationships.append(relationship)
                
                logger.info(f"Extracted {len(relationships)} relationships from event {event.id}")
                return relationships
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse relationship extraction response: {e}")
                logger.debug(f"Raw response: {content}")
                return []
                
        except Exception as e:
            logger.error(f"Relationship extraction failed for event {event.id}: {e}")
            return []
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a text string."""
        try:
            if self.settings.use_azure_openai:
                response = await asyncio.to_thread(
                    self.client.embeddings.create,
                    input=text,
                    model=self.settings.azure_openai_embedding_deployment_name
                )
            else:
                response = await asyncio.to_thread(
                    self.client.embeddings.create,
                    input=text,
                    model=self.settings.openai_embedding_model
                )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding of size {len(embedding)} for text: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text '{text[:50]}...': {e}")
            return None
    
    def prepare_embedding_text(self, entity: ExtractedEntity) -> str:
        """Prepare text for embedding generation from entity."""
        # Create a rich text representation of the entity
        text_parts = [
            f"Type: {entity.type}",
            f"Identifier: {entity.identifier}"
        ]
        
        # Add important properties
        for key, value in entity.properties.items():
            if key in ['name', 'title', 'description', 'content', 'category']:
                text_parts.append(f"{key}: {value}")
        
        # Add all other properties as a single line
        other_props = {k: v for k, v in entity.properties.items() 
                      if k not in ['name', 'title', 'description', 'content', 'category']}
        if other_props:
            text_parts.append(f"Properties: {json.dumps(other_props)}")
        
        return " | ".join(text_parts)
    
    async def process_event(self, event: EventSchema) -> ExtractionResult:
        """Process a single event to extract entities and relationships."""
        start_time = datetime.now()
        
        try:
            # Extract entities
            entities = await self.extract_entities(event)
            
            # Extract relationships
            relationships = await self.extract_relationships(entities, event)
              # Generate embeddings for entities (if not already generated)
            for entity in entities:
                if entity.embedding is None:
                    embedding_text = self.prepare_embedding_text(entity)
                    embedding = await self._generate_embedding(embedding_text)
                    entity.embedding = embedding
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ExtractionResult(
                entities=entities,
                relationships=relationships,
                success=True,
                processing_time=processing_time,
                token_usage={
                    "prompt_tokens": self.count_tokens(ENTITY_EXTRACTION_PROMPT) + self.count_tokens(RELATIONSHIP_EXTRACTION_PROMPT),
                    "completion_tokens": 0  # Would need to track from API response
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Event processing failed: {e}")
            
            return ExtractionResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    async def process_events_batch(self, events: List[EventSchema], 
                                 max_concurrent: int = 3) -> List[ExtractionResult]:
        """Process multiple events in parallel with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(event):
            async with semaphore:
                return await self.process_event(event)
        
        results = await asyncio.gather(
            *[process_with_semaphore(event) for event in events],
            return_exceptions=True
        )
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Event {events[i].id} processing failed with exception: {result}")
                processed_results.append(ExtractionResult(
                    success=False,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results

class ExtractionPipeline:
    """High-level pipeline for processing events into temporal knowledge graph."""
    
    def __init__(self, extractor: EntityExtractor = None):
        self.extractor = extractor or EntityExtractor()
        self.settings = get_settings()
    
    async def process_raw_events(self, events: List[EventSchema]) -> Dict[str, Any]:
        """Process raw events into nodes and edges for the temporal graph."""
        # Process events in batches
        batch_size = self.settings.entity_extraction_batch_size
        all_results = []
        
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(events) + batch_size - 1) // batch_size}")
            
            batch_results = await self.extractor.process_events_batch(batch)
            all_results.extend(batch_results)
        
        # Aggregate results
        total_entities = 0
        total_relationships = 0
        failed_events = 0
        all_nodes = []
        all_edges = []
        
        for result in all_results:
            if result.success:
                total_entities += len(result.entities)
                total_relationships += len(result.relationships)
                
                # Convert entities to nodes
                for entity in result.entities:
                    # Use event timestamp as valid_from
                    event_idx = all_results.index(result)
                    event = events[event_idx]
                    
                    node = entity.to_node_schema(
                        valid_from=event.timestamp,
                        embedding=getattr(entity, 'embedding', None)
                    )
                    all_nodes.append(node)
                
                # Convert relationships to edges (would need node ID mapping in real implementation)
                for relationship in result.relationships:
                    event_idx = all_results.index(result)
                    event = events[event_idx]
                    
                    # This is simplified - in real implementation, need to map entity identifiers to node IDs
                    edge = relationship.to_edge_schema(
                        source_node_id=relationship.source_entity,  # Would be actual UUID
                        target_node_id=relationship.target_entity,  # Would be actual UUID
                        valid_from=event.timestamp
                    )
                    all_edges.append(edge)
            else:
                failed_events += 1
        
        summary = {
            "processed_events": len(events),
            "successful_events": len(events) - failed_events,
            "failed_events": failed_events,
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "nodes": all_nodes,
            "edges": all_edges,
            "extraction_results": all_results
        }
        
        logger.info(f"Extraction pipeline completed: {summary['successful_events']}/{summary['processed_events']} events processed, "
                   f"{summary['total_entities']} entities, {summary['total_relationships']} relationships")
        
        return summary

# Convenience functions
async def extract_from_event(event: EventSchema) -> ExtractionResult:
    """Extract entities and relationships from a single event."""
    extractor = EntityExtractor()
    return await extractor.process_event(event)

async def extract_from_events(events: List[EventSchema]) -> Dict[str, Any]:
    """Extract entities and relationships from multiple events."""
    pipeline = ExtractionPipeline()
    return await pipeline.process_raw_events(events)
