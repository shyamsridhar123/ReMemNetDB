
# Product Requirements Document (PRD)

## E-commerce Agent Memory Platform (Graphiti) with Gradio UI


---

### 1. Executive Summary

We are building a memory layer for e-commerce applications using a temporal knowledge graph (Graphiti). This platform will allow agents and applications to track and reason about customer, product, and transaction data as it changes over time. The system will include a Gradio web interface for interactive testing and demonstration.

---

### 2. Problem Statement

Most e-commerce systems can’t “remember” how things change—what customers did last month, how product sentiment evolved, or how communities of buyers shift. They also can’t easily combine semantic, keyword, and graph-based search, or let non-technical users explore these features.

---

### 3. Objectives and Success Criteria

**Objectives:**

- Provide a real-time memory graph that records events and facts as they happen.
- Let agents and analysts query both current and historical states.
- Support fast, hybrid search (semantic, keyword, graph).
- Make all features accessible through a simple Gradio web UI.

**Success Criteria:**

- Queries (hybrid or temporal) return in under 2 seconds.
- All updates to the graph are consistent and bi-temporally accurate.
- Stakeholders use the Gradio UI for demos and testing.
- Agents can answer questions about both recent and past events.

---

### 4. Scope

**Included:**

- Synthetic data generation for realistic e-commerce events.
- Implementation using Azure PostgreSQL, Apache AGE, and pgvector.
- LLM-based extraction and contradiction handling.
- Gradio-based web UI for querying, visualization, and demo.

**Not Included:**

- Live production deployment.
- Integration with external e-commerce platforms.

---

### 5. Functional Requirements

#### A. Graphiti Memory Layer

- Store all events (orders, reviews, chats) as nodes with timestamps.
- Extract entities and relationships, tracking when each fact was true and when it was recorded.
- Cluster entities into communities (e.g., frequent buyers).
- Support real-time updates and handle conflicting facts (e.g., product in/out of stock).


#### B. Hybrid Retrieval \& Analytics

- Support semantic search with embeddings (pgvector).
- Enable keyword search (BM25).
- Allow graph traversals (e.g., “who bought similar products?”).
- Let users define new node and edge types.


#### C. Synthetic Data Generation

- Generate realistic, time-stamped e-commerce events (orders, reviews, chats, returns).
- Data must support time-based analytics and incremental updates.


#### D. Gradio UI

- Query panel for episodic/temporal questions (e.g., “What did Alice do last month?”).
- Hybrid search interface (semantic, keyword, graph).
- Graph visualization panel (showing entities and relationships over time).
- Agent memory demo (showing how agent recall evolves as new events are added).
- Real-time update panel (add events and see the graph update immediately).
- No-code: all features accessible via forms, dropdowns, and buttons.

---

### 6. User Stories

- As a developer, I want to test agent memory for evolving data.
- As a product manager, I want to see how customer journeys unfold over time.
- As a data scientist, I want to analyze how relationships and facts change.
- As a stakeholder, I want to use a web UI to try out features and scenarios.

---

### 7. Example Scenarios and Episodes

**Scenario 1: Personalized Customer Journey Recall**

- Alice browses electronics, adds a laptop to her wishlist, buys it, leaves a review, contacts support, and later buys a laptop bag after a recommendation.
- *Demo:* Query Alice’s electronics interactions for the last month. The agent shows the sequence and explains how her behavior changed.

**Scenario 2: Real-Time Trend and Community Shifts**

- A fitness tracker becomes popular, gets negative reviews for battery life, and is returned by many users who then switch to a competitor.
- *Demo:* Query product sentiment and buyer communities over three weeks. Visualize how the community and sentiment shift.

**Scenario 3: Temporal Reasoning for Support**

- Bob orders a camera on Black Friday, the order ships, he contacts support about a missing accessory, and the issue is resolved.
- *Demo:* Query Bob’s camera order events during Black Friday. Show the timeline and all related episodes.

**Scenario 4: Hybrid Search for Product Discovery**

- A user asks for “products like my noise-cancelling headphones but with better battery life.”
- *Demo:* The agent uses embeddings, attributes, and graph connections to recommend relevant products.

**Scenario 5: Fact Contradiction and Update**

- Product X is “in stock,” sells out, is marked “out of stock,” and is restocked a week later.
- *Demo:* Query Product X’s stock status over two weeks. Show how the system tracks and resolves conflicting facts.

**Scenario 6: Agentic Memory Evolution**

- Carol chats about hiking gear, buys boots after a follow-up, then gets a personalized offer for socks.
- *Demo:* Show how the agent’s knowledge about Carol’s preferences evolves.

**Scenario 7: Relationship Evolution**

- Product Y is first bought by young adults, then a campaign targets seniors, shifting the buyer community.
- *Demo:* Query and visualize how the buyer community for Product Y changes over a quarter.

---

### 8. Acceptance Criteria

- Agents can answer questions about both recent and historical events.
- Real-time updates are visible in both the graph and the UI.
- Hybrid search works as described and returns results quickly.
- All example scenarios can be run and visualized in the Gradio UI.
- The UI is usable by non-technical testers.

---

### 9. Sample Gradio UI Layout (for reference)

```python
import gradio as gr

def episodic_query(user_id, query):
    # Retrieve and display episodic/temporal memory
    return "Results..."

def hybrid_search(search_type, input_text):
    # Semantic, keyword, or graph search
    return ["Result 1", "Result 2"]

def visualize_graph(entity_id):
    # Render a subgraph visualization
    return "Graph Image or Plot"

with gr.Blocks() as demo:
    gr.Markdown("# Graphiti E-commerce Memory Demo")
    with gr.Tab("Episodic Query"):
        user_id = gr.Textbox(label="User ID")
        query = gr.Textbox(label="Query")
        output = gr.Textbox(label="Results")
        gr.Button("Submit").click(episodic_query, inputs=[user_id, query], outputs=output)
    with gr.Tab("Hybrid Search"):
        search_type = gr.Dropdown(["Semantic", "Keyword", "Graph"], label="Search Type")
        input_text = gr.Textbox(label="Search Input")
        results = gr.Textbox(label="Results")
        gr.Button("Search").click(hybrid_search, inputs=[search_type, input_text], outputs=results)
    with gr.Tab("Graph Visualization"):
        entity_id = gr.Textbox(label="Entity ID")
        graph_output = gr.Image(label="Graph Visualization")
        gr.Button("Visualize").click(visualize_graph, inputs=entity_id, outputs=graph_output)

demo.launch()
```


