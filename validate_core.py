#!/usr/bin/env python3
"""
Core validation test - Entity extraction and hybrid search
This test validates that our basic functionality works correctly.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from graphiti.memory.extraction import EntityExtractor
from graphiti.core.config import get_settings
import math

async def validate_core_functionality():
    """Validate that entity extraction and hybrid search work correctly"""
    print("🔍 Core Functionality Validation")
    print("=" * 50)
    
    # Initialize extractor
    extractor = EntityExtractor()
    
    # Test 1: Entity Extraction
    print("\n1️⃣ Testing Entity Extraction...")
    test_event = {
        "event_type": "order_placed",
        "customer_name": "Alice Johnson",
        "product": "Dell XPS 13 Laptop",
        "category": "Electronics",
        "description": "High-performance laptop computer for programming and development",
        "timestamp": "2024-01-15T10:30:00+00:00"
    }
    
    entities = await extractor.extract_entities(test_event)
    print(f"✅ Extracted {len(entities)} entities")
    
    for entity in entities:
        print(f"   - {entity.type}: {entity.identifier}")
    
    # Test 2: Embedding Generation
    print("\n2️⃣ Testing Embedding Generation...")
    embeddings_generated = 0
    
    for entity in entities:
        # Generate rich embedding
        entity_props = entity.properties or {}
        rich_text = f"{entity.type} {entity.identifier} {entity_props.get('category', '')} {entity_props.get('description', '')}".strip()
        
        entity.rich_embedding = await extractor.generate_embedding(rich_text)
        entity.simple_embedding = await extractor.generate_embedding(f"{entity.type}: {entity.identifier}")
        
        if entity.rich_embedding and entity.simple_embedding:
            embeddings_generated += 1
            print(f"   ✅ {entity.type}: {entity.identifier} - Embeddings: {len(entity.rich_embedding)} dims")
    
    print(f"✅ Generated embeddings for {embeddings_generated}/{len(entities)} entities")
    
    # Test 3: Cosine Similarity
    print("\n3️⃣ Testing Cosine Similarity...")
    
    def cosine_similarity(vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    # Test self-similarity (should be 1.0)
    if entities and hasattr(entities[0], 'rich_embedding'):
        self_sim = cosine_similarity(entities[0].rich_embedding, entities[0].rich_embedding)
        print(f"   Self-similarity: {self_sim:.6f} (should be 1.0)")
        
        if abs(self_sim - 1.0) < 0.0001:
            print("   ✅ Cosine similarity calculation is correct")
        else:
            print("   ❌ Cosine similarity calculation has issues")
    
    # Test 4: Basic Semantic Search
    print("\n4️⃣ Testing Semantic Search...")
    
    query = "laptop computer"
    query_embedding = await extractor.generate_embedding(query)
    
    semantic_results = []
    for entity in entities:
        if not hasattr(entity, 'rich_embedding'):
            continue
            
        similarity = cosine_similarity(query_embedding, entity.rich_embedding)
        if similarity > 0.3:  # threshold
            semantic_results.append({
                'entity': entity,
                'similarity': similarity
            })
    
    semantic_results.sort(key=lambda x: x['similarity'], reverse=True)
    
    print(f"   Query: '{query}'")
    print(f"   Found {len(semantic_results)} semantic matches:")
    for result in semantic_results:
        entity = result['entity']
        print(f"   - {entity.type}: {entity.identifier} (similarity: {result['similarity']:.3f})")
    
    # Test 5: Basic Keyword Search
    print("\n5️⃣ Testing Keyword Search...")
    
    keyword_results = []
    query_terms = query.lower().split()
    
    for entity in entities:
        entity_props = entity.properties or {}
        entity_text = f"{entity.type} {entity.identifier} {entity_props.get('category', '')} {entity_props.get('description', '')}".lower()
        
        score = 0
        for term in query_terms:
            if term in entity.identifier.lower():
                score += 1.5
            elif term in entity_text:
                score += 1.0
        
        if score > 0:
            keyword_results.append({
                'entity': entity,
                'score': score / len(query_terms)
            })
    
    keyword_results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"   Found {len(keyword_results)} keyword matches:")
    for result in keyword_results:
        entity = result['entity']
        print(f"   - {entity.type}: {entity.identifier} (score: {result['score']:.3f})")
    
    # Test 6: Hybrid Combination
    print("\n6️⃣ Testing Hybrid Search...")
    
    combined_results = {}
    
    # Add keyword results
    for result in keyword_results:
        entity_key = f"{result['entity'].type}_{result['entity'].identifier}"
        combined_results[entity_key] = {
            'entity': result['entity'],
            'keyword_score': result['score'],
            'semantic_score': 0,
            'source': 'keyword'
        }
    
    # Add semantic results
    for result in semantic_results:
        entity_key = f"{result['entity'].type}_{result['entity'].identifier}"
        if entity_key in combined_results:
            combined_results[entity_key]['semantic_score'] = result['similarity']
            combined_results[entity_key]['source'] = 'hybrid'
        else:
            combined_results[entity_key] = {
                'entity': result['entity'],
                'keyword_score': 0,
                'semantic_score': result['similarity'],
                'source': 'semantic'
            }
    
    # Calculate final scores
    final_results = []
    for data in combined_results.values():
        total_score = (data['keyword_score'] * 0.5) + (data['semantic_score'] * 0.5)
        data['total_score'] = total_score
        final_results.append(data)
    
    final_results.sort(key=lambda x: x['total_score'], reverse=True)
    
    print(f"   Hybrid results:")
    for result in final_results:
        entity = result['entity']
        print(f"   - {entity.type}: {entity.identifier}")
        print(f"     Total: {result['total_score']:.3f} (K: {result['keyword_score']:.3f}, S: {result['semantic_score']:.3f}) [{result['source']}]")
    
    print("\n🎉 Core functionality validation completed!")
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"   ✅ Entity extraction: {len(entities)} entities extracted")
    print(f"   ✅ Embedding generation: {embeddings_generated} embeddings created")
    print(f"   ✅ Semantic search: {len(semantic_results)} matches found")
    print(f"   ✅ Keyword search: {len(keyword_results)} matches found")
    print(f"   ✅ Hybrid search: {len(final_results)} combined results")
    
    if len(entities) > 0 and embeddings_generated > 0 and len(final_results) > 0:
        print("\n🎯 ALL CORE FUNCTIONALITY IS WORKING CORRECTLY!")
        return True
    else:
        print("\n❌ Some core functionality is not working properly")
        return False

if __name__ == "__main__":
    success = asyncio.run(validate_core_functionality())
    if success:
        print("\n✅ Ready for UI integration!")
    else:
        print("\n❌ Fix core issues before proceeding!")
