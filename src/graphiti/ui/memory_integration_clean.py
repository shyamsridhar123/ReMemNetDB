"""
Enhanced Memory Store Integration for Gradio UI
Clean implementation with proper backend integration
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from graphiti.memory import MemoryStore, MemoryQuery, Episode
from graphiti.core.models import Event


class MemoryStoreUIAdapter:
    """Adapter class to integrate MemoryStore with Gradio UI"""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def query_customer_episodes(self, customer_id: str, days_back: int = 30) -> tuple:
        """Query customer episodes and return formatted data for Gradio"""
        try:
            # Get episodes from memory store
            episodes = self.memory_store.get_customer_episodes(customer_id, days_back)
            
            # Format for Gradio DataFrame
            episodes_data = []
            timeline_data = []
            
            for i, episode in enumerate(episodes):
                episode_start = episode.start_time.strftime("%Y-%m-%d %H:%M:%S")
                episode_end = episode.end_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Create summary of events in episode
                event_types = [event.event_type for event in episode.events]
                event_summary = ", ".join(set(event_types))
                
                # Extract key entities
                entities = []
                for event in episode.events:
                    event_data = json.loads(event.event_data) if isinstance(event.event_data, str) else event.event_data
                    if 'customer_name' in event_data:
                        entities.append(f"customer:{event_data['customer_name']}")
                    if 'product_name' in event_data:
                        entities.append(f"product:{event_data['product_name']}")
                
                entity_summary = ", ".join(set(entities))
                
                episodes_data.append([
                    episode_start,
                    event_summary,
                    f"Episode {i+1}: {len(episode.events)} events",
                    entity_summary
                ])
                
                # Timeline data for visualization
                timeline_data.append({
                    'timestamp': episode.start_time,
                    'event_type': event_summary,
                    'episode_id': i+1,
                    'event_count': len(episode.events)
                })
            
            # Create timeline visualization
            timeline_fig = self._create_timeline_plot(timeline_data, customer_id)
            
            return episodes_data, timeline_fig
            
        except Exception as e:
            error_data = [[f"Error querying episodes: {str(e)}", "", "", ""]]
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Error loading timeline")
            return error_data, empty_fig
    
    def perform_hybrid_search(self, query: str, search_types: List[str], max_results: int = 10) -> List[List[str]]:
        """Perform hybrid search using the memory store"""
        try:
            # Map UI search types to backend types
            type_mapping = {
                "Semantic": "semantic",
                "Keyword": "keyword", 
                "Temporal": "temporal"
            }
            
            results = []
            
            for search_type_ui in search_types:
                search_type = type_mapping.get(search_type_ui, "hybrid")
                
                # Create memory query
                memory_query = MemoryQuery(
                    query_text=query,
                    query_type=search_type,
                    max_results=max_results // len(search_types)
                )
                
                # Execute search
                search_results = self.memory_store.query_memory(memory_query)
                
                # Format results
                for result in search_results:
                    if hasattr(result, 'score'):
                        score = result.score
                    elif hasattr(result, 'relevance_score'):
                        score = result.relevance_score
                    else:
                        score = 0.0
                    
                    if hasattr(result, 'content'):
                        entity_type = result.content.get('entity_type', 'unknown')
                        identifier = result.content.get('identifier', 'N/A')
                        properties = result.content.get('properties', {})
                    else:
                        entity_type = getattr(result, 'entity_type', 'unknown')
                        identifier = getattr(result, 'identifier', 'N/A')
                        properties = getattr(result, 'properties', {})
                    
                    description = f"{entity_type}: {identifier}"
                    if properties and properties.get('name'):
                        description += f" - {properties['name']}"
                    
                    timestamp = getattr(result, 'timestamp', 'N/A')
                    if hasattr(timestamp, 'strftime'):
                        timestamp = timestamp.strftime("%Y-%m-%d %H:%M")
                    
                    results.append([
                        f"{score:.3f}",
                        search_type_ui,
                        description,
                        str(timestamp)
                    ])
            
            # Sort by score and limit results
            results.sort(key=lambda x: float(x[0]), reverse=True)
            return results[:max_results]
            
        except Exception as e:
            return [[f"Search error: {str(e)}", "", "", ""]]
    
    def store_sample_events(self, customer_id: str) -> str:
        """Store sample events for demonstration"""
        try:
            # Generate sample events for this customer
            sample_events = [
                {
                    'event_type': 'product_viewed',
                    'customer_name': 'Sample Customer',
                    'product_name': 'Dell XPS 13 Laptop',
                    'category': 'Electronics',
                    'price': 1299.99,
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=2)
                },
                {
                    'event_type': 'order_placed',
                    'customer_name': 'Sample Customer',
                    'product_name': 'Dell XPS 13 Laptop',
                    'category': 'Electronics',
                    'price': 1299.99,
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=1)
                },
                {
                    'event_type': 'review_posted',
                    'customer_name': 'Sample Customer',
                    'product_name': 'Dell XPS 13 Laptop',
                    'rating': 5,
                    'review': 'Great laptop for development work!',
                    'timestamp': datetime.now(timezone.utc) - timedelta(minutes=30)
                }
            ]
            
            # Store each event
            stored_count = 0
            for event_data in sample_events:
                event_type = event_data.pop('event_type')
                timestamp = event_data.pop('timestamp')
                
                episode_id = self.memory_store.store_event(
                    customer_id=customer_id,
                    event_data=event_data,
                    event_type=event_type,
                    timestamp=timestamp
                )
                stored_count += 1
            
            return f"✅ Successfully stored {stored_count} sample events for customer {customer_id}"
            
        except Exception as e:
            return f"❌ Error storing sample events: {str(e)}"

    def get_system_analytics(self) -> tuple:
        """Get analytics data for memory store dashboard"""
        try:
            # Get system statistics from memory store
            stats = self.memory_store.get_system_stats()
            
            # Format entity statistics
            entity_stats = []
            if 'entity_counts' in stats:
                for entity_type, count in stats['entity_counts'].items():
                    entity_stats.append([
                        entity_type.title(),
                        str(count),
                        stats.get('last_updated', 'N/A')
                    ])
            else:
                entity_stats = [["No data", "0", "N/A"]]
            
            # Format relationship statistics  
            relationship_stats = []
            if 'relationship_counts' in stats:
                for rel_type, count in stats['relationship_counts'].items():
                    avg_conf = stats.get('avg_confidence', {}).get(rel_type, 0.0)
                    relationship_stats.append([
                        rel_type.replace('_', ' ').title(),
                        str(count),
                        f"{avg_conf:.2f}"
                    ])
            else:
                relationship_stats = [["No data", "0", "0.0"]]
            
            # Create memory growth chart
            growth_fig = self._create_memory_growth_chart(stats)
            
            return entity_stats, relationship_stats, growth_fig
            
        except Exception as e:
            # Return error data
            error_data = [[f"Error: {str(e)}", "0", "N/A"]]
            error_fig = go.Figure()
            error_fig.update_layout(title=f"Analytics error: {str(e)}")
            return error_data, error_data, error_fig

    def get_system_status(self) -> tuple:
        """Get detailed system status information"""
        try:
            # Get system stats from memory store
            stats = self.memory_store.get_system_stats()
            
            # Format status HTML
            if stats.get('memory_store_status') == 'healthy':
                status_html = '<div class="status-success">🟢 <strong>System Status:</strong> All components operational</div>'
            else:
                status_html = '<div class="status-error">🔴 <strong>System Status:</strong> Issues detected</div>'
            
            # Format database statistics
            db_stats = []
            if 'database_stats' in stats:
                for table, info in stats['database_stats'].items():
                    db_stats.append([
                        table,
                        str(info.get('row_count', 0)),
                        info.get('last_updated', 'N/A')
                    ])
            else:
                db_stats = [["No data", "0", "N/A"]]
            
            return status_html, db_stats
            
        except Exception as e:
            error_html = f'<div class="status-error">❌ System error: {str(e)}</div>'
            error_stats = [[f"Error: {str(e)}", "0", "N/A"]]
            return error_html, error_stats

    def _create_timeline_plot(self, timeline_data: List[Dict], customer_id: str) -> go.Figure:
        """Create timeline plot for customer journey"""
        try:
            if not timeline_data:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"No timeline data found for customer {customer_id}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16)
                )
                fig.update_layout(title=f"Customer Journey - {customer_id}")
                return fig
            
            # Extract data for plotting
            timestamps = [item['timestamp'] for item in timeline_data]
            event_types = [item['event_type'] for item in timeline_data]
            episode_ids = [item['episode_id'] for item in timeline_data]
            event_counts = [item['event_count'] for item in timeline_data]
            
            # Create scatter plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=episode_ids,
                mode='markers+lines',
                marker=dict(
                    size=[count * 5 + 10 for count in event_counts],
                    color=episode_ids,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Episode")
                ),
                text=[f"Events: {count}<br>Type: {event_type}" 
                      for count, event_type in zip(event_counts, event_types)],
                hovertemplate='<b>Episode %{y}</b><br>' +
                             'Time: %{x}<br>' +
                             '%{text}<br>' +
                             '<extra></extra>',
                name='Customer Journey'
            ))
            
            fig.update_layout(
                title=f"Customer Journey Timeline - {customer_id}",
                xaxis_title="Time",
                yaxis_title="Episode",
                template="plotly_white",
                height=400,
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            # Return error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating timeline: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            fig.update_layout(title="Timeline Error")
            return fig

    def _create_memory_growth_chart(self, stats: Dict[str, Any]) -> go.Figure:
        """Create memory growth visualization"""
        try:
            # Create a simple growth chart based on available data
            fig = go.Figure()
            
            if 'total_entities' in stats and 'total_relationships' in stats:
                fig.add_trace(go.Scatter(
                    x=['Entities', 'Relationships', 'Events'],
                    y=[
                        stats.get('total_entities', 0),
                        stats.get('total_relationships', 0),
                        stats.get('total_events', 0)
                    ],
                    mode='lines+markers',
                    name='Memory Growth',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=10, color='#764ba2')
                ))
                
                fig.update_layout(
                    title="Memory Store Growth",
                    xaxis_title="Component",
                    yaxis_title="Count",
                    template="plotly_white",
                    height=400
                )
            else:
                fig.add_annotation(
                    text="No growth data available",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=16)
                )
                fig.update_layout(title="Memory Growth - No Data")
            
            return fig
            
        except Exception as e:
            # Return error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=14, color="red")
            )
            fig.update_layout(title="Chart Error")
            return fig
