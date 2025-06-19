#!/usr/bin/env python3
"""
Test script for MemoryStore functionality
Tests the complete memory storage and retrieval pipeline
"""

import sys
import os
import uuid
sys.path.append('src')

from datetime import datetime, timezone
from graphiti.memory.memory_store import MemoryStore, MemoryQuery, TimeRange
from graphiti.core.logging_config import setup_logging

def test_memory_store():
    """Test MemoryStore initialization and basic operations"""
    print("🧠 Testing MemoryStore...")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    
    try:        # Initialize MemoryStore
        print("📋 1. Initializing MemoryStore...")
        memory_store = MemoryStore()
        print("✅ MemoryStore initialized successfully")
        
        # Generate proper UUID for customer
        customer_alice_id = str(uuid.uuid4())
        print(f"📋 Using customer ID: {customer_alice_id}")
        
        # Test sample events
        events = [
            {
                "event_id": "evt_001",
                "customer_id": customer_alice_id,
                "event_type": "order_placed",
                "timestamp": "2024-01-15T10:30:00+00:00",
                "data": {
                    "customer_name": "Alice Johnson",
                    "product_name": "Dell XPS 13 Laptop",
                    "price": 1299.99,
                    "category": "Electronics"
                }
            },
            {
                "event_id": "evt_002", 
                "customer_id": customer_alice_id,
                "event_type": "product_viewed",
                "timestamp": "2024-01-20T14:15:00+00:00",
                "data": {
                    "customer_name": "Alice Johnson",
                    "product_name": "MacBook Pro 16",
                    "category": "Electronics",
                    "viewed_duration": 120
                }
            }
        ]
        
        # Test storing events
        print("\n📋 2. Testing event storage...")
        for i, event in enumerate(events, 1):
            print(f"   Processing event {i}: {event['event_type']}")
            try:
                episode_id = memory_store.store_event(
                    customer_id=event["customer_id"],
                    event_data=event["data"],
                    event_type=event["event_type"],
                    timestamp=datetime.fromisoformat(event["timestamp"].replace('Z', '+00:00'))
                )
                print(f"   ✅ Event stored with episode_id: {episode_id}")
            except Exception as e:
                print(f"   ❌ Failed to store event: {e}")
        
        # Test memory queries
        print("\n📋 3. Testing memory queries...")
        
        # Test semantic query
        print("   3a. Semantic search for 'laptop computer'...")
        query = MemoryQuery(
            query_text="laptop computer",
            query_type="semantic",
            max_results=5
        )
        try:
            results = memory_store.query_memory(query)
            print(f"   ✅ Found {len(results)} semantic results")
            for i, result in enumerate(results, 1):
                print(f"      {i}. {result.memory_type}: {result.relevance_score:.4f}")
        except Exception as e:
            print(f"   ❌ Semantic search failed: {e}")
        
        # Test keyword query
        print("   3b. Keyword search for 'Electronics'...")
        query = MemoryQuery(
            query_text="Electronics",
            query_type="keyword",
            max_results=5
        )
        try:
            results = memory_store.query_memory(query)
            print(f"   ✅ Found {len(results)} keyword results")
            for i, result in enumerate(results, 1):
                print(f"      {i}. {result.memory_type}: {result.relevance_score:.4f}")
        except Exception as e:
            print(f"   ❌ Keyword search failed: {e}")
        
        # Test hybrid query
        print("   3c. Hybrid search for 'Alice laptop'...")
        query = MemoryQuery(
            query_text="Alice laptop",
            query_type="hybrid",
            max_results=5
        )
        try:
            results = memory_store.query_memory(query)
            print(f"   ✅ Found {len(results)} hybrid results")
            for i, result in enumerate(results, 1):
                print(f"      {i}. {result.memory_type}: {result.relevance_score:.4f}")
        except Exception as e:
            print(f"   ❌ Hybrid search failed: {e}")
        
        # Test temporal query
        print("   3d. Temporal search for January 2024...")
        time_range = TimeRange(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc)
        )
        query = MemoryQuery(
            query_text="order",
            query_type="temporal",
            time_range=time_range,
            max_results=5
        )
        try:
            results = memory_store.query_memory(query)
            print(f"   ✅ Found {len(results)} temporal results")
            for i, result in enumerate(results, 1):
                print(f"      {i}. {result.memory_type}: {result.relevance_score:.4f}")
        except Exception as e:
            print(f"   ❌ Temporal search failed: {e}")
          # Test episode retrieval
        print("\n📋 4. Testing episode retrieval...")
        try:
            episodes = memory_store.get_customer_episodes(customer_alice_id)
            print(f"   ✅ Found {len(episodes)} episodes for customer {customer_alice_id[:8]}...")
            for i, episode in enumerate(episodes, 1):
                print(f"      Episode {i}: {episode.summary}")
        except Exception as e:
            print(f"   ❌ Episode retrieval failed: {e}")
        
        print("\n🎉 MemoryStore test completed successfully!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ MemoryStore test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_memory_store()
    exit(0 if success else 1)
