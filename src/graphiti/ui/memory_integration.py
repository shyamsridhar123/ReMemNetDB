"""
Enhanced Memory Store Integration for Gradio UI
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import func, text

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
            # Add logging to debug the issue
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"🔍 query_customer_episodes called with customer_id={customer_id}, days_back={days_back}")
            
            # Validate customer_id
            if not customer_id or not customer_id.strip():
                logger.warning("❌ Empty customer_id provided")
                error_data = [["Error: Empty customer ID", "", "", ""]]
                empty_fig = go.Figure()
                empty_fig.add_annotation(
                    text="Please enter a valid customer ID",
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False, font=dict(size=16)
                )
                empty_fig.update_layout(title="Customer Journey - Invalid ID")
                return error_data, empty_fig
            
            # Get episodes from memory store
            logger.debug(f"📋 Calling memory_store.get_customer_episodes...")
            episodes = self.memory_store.get_customer_episodes(customer_id, days_back)
            logger.info(f"📊 Retrieved {len(episodes)} episodes from memory store")
            
            # Check if we have any episodes
            if not episodes:
                logger.warning(f"⚠️ No episodes found for customer {customer_id}")
                error_data = [[f"No episodes found for customer {customer_id[:16]}...", "", "", ""]]
                empty_fig = go.Figure()
                empty_fig.add_annotation(
                    text=f"No timeline data found for customer: {customer_id}",
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False, font=dict(size=16)
                )
                empty_fig.update_layout(title=f"Customer Journey - {customer_id}")
                return error_data, empty_fig
            
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
            logger.info(f"✅ Successfully created timeline with {len(episodes_data)} episodes")
            
            return episodes_data, timeline_fig
            
        except Exception as e:
            logger.error(f"❌ Error in query_customer_episodes: {str(e)}", exc_info=True)
            error_data = [[f"Error querying episodes: {str(e)}", "", "", ""]]
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text=f"Error loading customer data: {str(e)}",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14, color="red")
            )
            empty_fig.update_layout(title="Error loading timeline")
            return error_data, empty_fig
    
    def perform_hybrid_search(self, query: str, search_types: List[str], max_results: int = 10) -> List[List[str]]:
        """Perform hybrid search using the memory store"""
        try:
            # Map UI search types to backend types
            type_mapping = {
                "Semantic": "semantic",
                "Keyword": "keyword", 
                "Graph Traversal": "temporal"
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
                    # Memory object attributes
                    score = getattr(result, 'relevance_score', 0.0)
                    
                    # Extract entity info from content
                    content = getattr(result, 'content', {})
                    entity_type = content.get('entity_type', 'unknown')
                    identifier = content.get('identifier', 'N/A')
                    properties = content.get('properties', {})
                    
                    if isinstance(properties, str):
                        try:
                            properties = json.loads(properties)
                        except:
                            properties = {}
                    
                    description = f"{entity_type}: {identifier}"
                    if properties.get('name'):
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
            from datetime import datetime, timezone, timedelta
            import uuid
            
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
                
                # Add debug logging
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"🔧 About to call store_event with:")
                logger.debug(f"  customer_id: {customer_id}")
                logger.debug(f"  event_data: {event_data}")
                logger.debug(f"  event_type: {event_type}")
                logger.debug(f"  timestamp: {timestamp}")
                logger.debug(f"  Method signature: {self.memory_store.store_event.__code__.co_varnames}")
                
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

    def get_memory_analytics(self) -> tuple:
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
                xref="paper", yref="paper",                showarrow=False,
                font=dict(size=14, color="red")
            )
            fig.update_layout(title="Chart Error")
            return fig

    def get_system_analytics(self) -> tuple:
        """Get real system analytics from the database"""
        try:
            # Import the database manager
            from graphiti.core.database import db_manager
            from graphiti.core.models import TemporalNode, TemporalEdge
            from sqlalchemy import func, text
            
            entity_stats = []
            relationship_stats = []
            
            # Use the same pattern as MemoryStore
            with db_manager.get_session() as session:
                # Get entity statistics
                entity_results = session.query(
                    TemporalNode.type,
                    func.count(TemporalNode.id).label('count'),
                    func.max(TemporalNode.valid_from).label('latest_update')
                ).filter(
                    (TemporalNode.valid_to.is_(None)) | (TemporalNode.valid_to > func.now())
                ).group_by(TemporalNode.type).order_by(text('count DESC')).all()
                
                for entity_type, count, latest_update in entity_results:
                    latest_str = latest_update.strftime("%Y-%m-%d %H:%M:%S") if latest_update else "N/A"
                    entity_stats.append([entity_type or "unknown", str(count), latest_str])
                
                # Get relationship statistics  
                relationship_results = session.query(
                    TemporalEdge.relationship_type,
                    func.count(TemporalEdge.id).label('count')
                ).filter(
                    (TemporalEdge.valid_to.is_(None)) | (TemporalEdge.valid_to > func.now())
                ).group_by(TemporalEdge.relationship_type).order_by(text('count DESC')).all()
                
                for rel_type, count in relationship_results:
                    relationship_stats.append([rel_type or "unknown", str(count), "N/A"])
            
            # Create memory growth chart
            growth_fig = self._create_memory_growth_chart()
            
            return entity_stats, relationship_stats, growth_fig
            
        except Exception as e:
            # Return error data if database query fails
            error_stats = [[f"Error: {str(e)}", "0", "N/A"]]
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Error loading analytics data")
            return error_stats, error_stats, empty_fig

    def _create_memory_growth_chart(self) -> go.Figure:
        """Create memory growth chart from real database data"""
        try:
            from graphiti.core.database import db_manager
            from graphiti.core.models import Event
            from sqlalchemy import func, text
            
            dates = []
            cumulative_counts = []
            
            with db_manager.get_session() as session:
                # Query for ALL events grouped by day using created_at (when stored in DB)
                growth_results = session.query(
                    func.date_trunc('day', Event.created_at).label('date'),
                    func.count(Event.id).label('daily_count')
                ).group_by(
                    func.date_trunc('day', Event.created_at)
                ).order_by(text('date')).all()
                
                # Calculate cumulative counts over time
                cumulative = 0
                for date, daily_count in growth_results:
                    cumulative += daily_count
                    dates.append(date)
                    cumulative_counts.append(cumulative)
            
            # Create the chart
            fig = go.Figure()
            
            if dates and cumulative_counts:
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=cumulative_counts,
                    mode='lines+markers',
                    name='Total Events',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title="Knowledge Graph Growth Over Time (Events)",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Event Count",
                    height=400,
                    hovermode='x unified'
                )
            else:
                # Show message if no data
                fig.add_annotation(
                    text="No events found in the last 30 days",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=16)
                )
                fig.update_layout(
                    title="Knowledge Graph Growth Over Time",
                    height=400
                )
            
            return fig
            
        except Exception as e:
            # Return empty chart if query fails
            fig = go.Figure()
            fig.update_layout(
                title=f"Memory Growth Chart - Error: {str(e)}",
                height=400
            )
            return fig
            return fig
    
    def get_system_status(self) -> tuple:
        """Get real system status from database"""
        try:
            # Use the memory store's built-in system stats method
            stats = self.memory_store.get_system_stats()
            
            # Format the stats for display
            db_stats = []
            for key, value in stats.items():
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                db_stats.append([key, str(value), current_time])
            
            # Generate status HTML
            status_html = self._generate_status_html(db_stats)
            
            return status_html, db_stats
            
        except Exception as e:
            error_stats = [[f"Error: {str(e)}", "0", "N/A"]]
            error_html = f'<div class="status-error">❌ Database connection failed: {str(e)}</div>'
            return error_html, error_stats
    
    def _generate_status_html(self, db_stats: List[List[str]]) -> str:
        """Generate HTML status indicator"""
        try:
            total_entities = int(db_stats[1][1])  # nodes count
            total_events = int(db_stats[0][1])    # events count
            
            if total_entities > 0 and total_events > 0:
                return '''
                <div class="status-success">
                    ✅ System Status: Operational<br>
                    🔗 Database: Connected<br>
                    📊 Memory Store: Active<br>
                    🔍 Search Engine: Ready
                </div>
                '''
            else:
                return '''
                <div class="status-error">
                    ⚠️ System Status: Limited Data<br>
                    🔗 Database: Connected<br>
                    📊 Memory Store: Empty<br>
                    🔍 Search Engine: Ready
                </div>
                '''
        except:
            return '''
            <div class="status-error">
                ❌ System Status: Error<br>
                🔗 Database: Connection Issues<br>
                📊 Memory Store: Unavailable<br>
                🔍 Search Engine: Offline
            </div>
            '''
    
    def store_sample_events(self, customer_id: str) -> str:
        """Store sample events for demonstration"""
        try:
            sample_events = [
                {
                    'event_type': 'order_placed',
                    'event_data': {
                        'customer_id': customer_id,
                        'customer_name': 'Demo Customer',
                        'product_name': 'Demo Product',
                        'price': 99.99,
                        'category': 'Electronics'
                    },
                    'timestamp': datetime.now() - timedelta(days=5)
                },
                {
                    'event_type': 'product_viewed',
                    'event_data': {
                        'customer_id': customer_id,
                        'customer_name': 'Demo Customer',
                        'product_name': 'Related Product',
                        'category': 'Electronics',
                        'viewed_duration': 120
                    },
                    'timestamp': datetime.now() - timedelta(days=2)
                }
            ]
            
            for event_data in sample_events:
                self.memory_store.store_event(
                    event_type=event_data['event_type'],
                    event_data=event_data['event_data'],
                    timestamp=event_data['timestamp']
                )
            
            return f"✅ Successfully stored {len(sample_events)} sample events for customer {customer_id}"
            
        except Exception as e:
            return f"❌ Error storing sample events: {str(e)}"
    
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
