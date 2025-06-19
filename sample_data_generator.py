#!/usr/bin/env python3
"""
Sample data generator for testing the Gradio Hybrid Search UI
"""

import json
from datetime import datetime, timezone

def generate_sample_events():
    """Generate sample e-commerce events for testing"""
    
    events = [
        {
            "event_type": "order_placed",
            "customer_name": "Alice Johnson",
            "product_name": "Dell XPS 13 Laptop", 
            "category": "Electronics",
            "description": "High-performance laptop computer for programming and development",
            "price": 1299.99,
            "timestamp": "2024-01-15T10:30:00Z"
        },
        {
            "event_type": "product_review",
            "customer_name": "Alice Johnson",
            "product_name": "Dell XPS 13 Laptop",
            "rating": 5,
            "review": "Great laptop for programming and work. Excellent for software development.",
            "timestamp": "2024-01-20T14:15:00Z"
        },
        {
            "event_type": "order_placed",
            "customer_name": "Bob Smith",
            "product_name": "iPhone 15 Pro",
            "category": "Electronics",
            "description": "Latest smartphone with advanced camera and processing power",
            "price": 999.99,
            "timestamp": "2024-01-18T16:45:00Z"
        },
        {
            "event_type": "product_inquiry",
            "customer_name": "Carol Davis",
            "product_name": "Gaming Desktop Computer",
            "category": "Desktop Computer",
            "inquiry": "Looking for a powerful gaming computer for video editing and gaming",
            "timestamp": "2024-01-22T09:20:00Z"
        },
        {
            "event_type": "order_placed",
            "customer_name": "David Wilson",
            "product_name": "MacBook Pro",
            "category": "Electronics",
            "description": "Professional laptop for creative work and development",
            "price": 2499.99,
            "timestamp": "2024-01-25T11:10:00Z"
        },
        {
            "event_type": "support_ticket",
            "customer_name": "Eve Martinez",
            "product_name": "Wireless Headphones",
            "category": "Audio",
            "issue": "Bluetooth connectivity problems with smartphone pairing",
            "timestamp": "2024-01-28T13:45:00Z"
        }
    ]
    
    return events

def get_sample_queries():
    """Get sample search queries for testing"""
    
    queries = [
        "laptop computer",
        "programming development", 
        "smartphone phone",
        "Alice customer",
        "electronics technology",
        "gaming computer",
        "Bluetooth wireless",
        "creative work",
        "support issues"
    ]
    
    return queries

def print_sample_data():
    """Print sample data for easy copy-paste into the UI"""
    
    print("🎯 Sample E-commerce Events for Testing")
    print("=" * 60)
    print()
    
    events = generate_sample_events()
    
    for i, event in enumerate(events, 1):
        print(f"📊 Event {i}: {event['event_type']}")
        print("Copy this JSON into the Event Data field:")
        print("-" * 40)
        print(json.dumps(event, indent=2))
        print()
    
    print("🔍 Sample Search Queries:")
    print("-" * 30)
    queries = get_sample_queries()
    for query in queries:
        print(f"• {query}")
    
    print()
    print("🚀 Instructions:")
    print("1. Start the UI: python gradio_hybrid_search_ui.py")
    print("2. Go to 'Entity Processing' tab")
    print("3. Copy-paste each event JSON and click 'Process Event'")
    print("4. Go to 'Hybrid Search' tab") 
    print("5. Try the sample queries above")
    print("6. Check the 'Analytics' tab for search history")

if __name__ == "__main__":
    print_sample_data()
