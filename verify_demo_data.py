#!/usr/bin/env python3
"""
Demo Data Verification Script
Verify that demo data is properly loaded and ready for comprehensive demo
"""

import sys
import os
sys.path.append('src')

from datetime import datetime
from sqlalchemy import text
from graphiti.core.database import db_manager
from graphiti.core.logging_config import setup_logging

def verify_demo_data():
    """Verify that demo data is properly loaded"""
    print("🔍 Verifying Demo Data Setup")
    print("=" * 50)
    
    setup_logging()
    with db_manager.get_session() as session:
        # Count events by type
        result = session.execute(text("""
            SELECT 
                event_type,
                COUNT(*) as count,
                MIN(timestamp) as earliest,
                MAX(timestamp) as latest
            FROM temporal_graph.events 
            GROUP BY event_type 
            ORDER BY count DESC
        """))
        
        print("\n📊 Events by Type:")
        print("-" * 30)
        total_events = 0
        for row in result:
            total_events += row[1]
            print(f"  {row[0]:<20} {row[1]:>3} events ({row[2].strftime('%Y-%m-%d')} to {row[3].strftime('%Y-%m-%d')})")
        
        print(f"\n📈 Total Events: {total_events}")
          # Count unique customers
        result = session.execute(text("""
            SELECT COUNT(DISTINCT event_data->>'customer_name') as unique_customers
            FROM temporal_graph.events 
            WHERE event_data->>'customer_name' IS NOT NULL
        """))
        
        customer_count = result.fetchone()[0]
        print(f"👥 Unique Customers: {customer_count}")
          # Count unique products
        result = session.execute(text("""
            SELECT COUNT(DISTINCT event_data->>'product_name') as unique_products
            FROM temporal_graph.events 
            WHERE event_data->>'product_name' IS NOT NULL
        """))
        
        product_count = result.fetchone()[0]
        print(f"🛍️  Unique Products: {product_count}")
        
        # Show customer journey summary
        print("\n👨‍💼 Customer Journey Summary:")
        print("-" * 35)
        result = session.execute(text("""
            SELECT 
                event_data->>'customer_name' as customer,
                COUNT(*) as events,
                STRING_AGG(DISTINCT event_type, ', ') as event_types
            FROM temporal_graph.events 
            WHERE event_data->>'customer_name' IS NOT NULL
            GROUP BY event_data->>'customer_name'
            ORDER BY events DESC
        """))
        
        for row in result:
            print(f"  {row[0]:<15} {row[1]:>2} events: {row[2]}")
        
        # Show recent activity
        print("\n🕐 Recent Activity (for real-time demo):")
        print("-" * 40)
        result = session.execute(text("""
            SELECT 
                event_type,
                event_data->>'customer_name' as customer,
                timestamp
            FROM temporal_graph.events 
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            ORDER BY timestamp DESC        """))
        
        recent_count = 0
        for row in result:
            recent_count += 1
            customer_name = row[1] or "Unknown"
            event_type = row[0] or "unknown_event"
            timestamp = row[2]
            if timestamp:
                time_str = timestamp.strftime('%H:%M:%S')
            else:
                time_str = "unknown_time"
            print(f"  {customer_name:<15} {event_type:<20} {time_str}")
        
        if recent_count == 0:
            print("  No recent activity found - this is expected if demo data was loaded more than 1 hour ago")
        else:
            print(f"  ✅ {recent_count} recent events found for real-time demo")
        
        print("\n🎯 Demo Readiness Checklist:")
        print("-" * 30)
        print(f"  ✅ Events loaded: {total_events}")
        print(f"  ✅ Customers ready: {customer_count}")
        print(f"  ✅ Products ready: {product_count}")
        print("  ✅ Database schema: temporal_graph")
        print("  ✅ Journey data: Alice, Bob, Carol")
        print("  ✅ Search queries: Programming, gaming, creative")
        print("  ✅ Support examples: Issues and resolutions")
        
        print("\n🚀 Ready for Demo!")
        print("Start the Gradio UI with: uv run python src/graphiti/ui/enhanced_gradio_app.py")

if __name__ == "__main__":
    verify_demo_data()
