import os
import json
from typing import TypedDict, List, Dict
from neo4j import GraphDatabase
from langgraph.graph import StateGraph, END

# --- 1. Configuration ---
URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# --- 2. Define State ---
class AgentState(TypedDict):
    defect_id: str
    traceability_data: Dict
    correlation_stats: Dict
    hypothesis: str
    confidence: float
    evidence_citations: List[str]

# --- 3. Define Deterministic Tools ---
def fetch_traceability_data(defect_id: str) -> Dict:
    """Queries Neo4j for the lineage of the defect."""
    query = """
    MATCH (d:DefectRecord {defect_id: $defect_id})-[:CAUSED_BY]->(b:FabricBatch)
    MATCH (b)<-[:SUPPLIED]-(s:Supplier)
    MATCH (d)-[:OCCURRED_ON]->(m:Machine)
    RETURN b.id AS batch_id, b.status AS batch_status, 
           s.name AS supplier_name, m.tension AS machine_tension
    """
    with driver.session() as session:
        result = session.run(query, defect_id=defect_id).single()
        return dict(result) if result else {}

def calculate_correlation_stats(defect_id: str, supplier_name: str) -> Dict:
    """Calculates the historical defect rate for this supplier vs overall."""
    query = """
    MATCH (s:Supplier {name: $supplier_name})<-[:SUPPLIED]-(b:FabricBatch)<-[:CAUSED_BY]-(d:DefectRecord)
    WITH count(d) AS supplier_defects
    MATCH (d_all:DefectRecord)
    WITH supplier_defects, count(d_all) AS total_defects
    RETURN supplier_defects, total_defects, 
           toFloat(supplier_defects) / total_defects AS supplier_share
    """
    with driver.session() as session:
        result = session.run(query, supplier_name=supplier_name).single()
        return dict(result) if result else {"supplier_share": 0.0}

# --- 4. Define Graph Nodes ---
def node_fetch_context(state: AgentState):
    print(f"--- NODE: Fetching Traceability for {state['defect_id']} ---")
    data = fetch_traceability_data(state['defect_id'])
    return {"traceability_data": data}

def node_analyze_correlation(state: AgentState):
    print(f"--- NODE: Analyzing Correlation ---")
    supplier = state['traceability_data'].get('supplier_name', 'Unknown')
    stats = calculate_correlation_stats(state['defect_id'], supplier)
    return {"correlation_stats": stats}

def node_synthesize_hypothesis(state: AgentState):
    print(f"--- NODE: Synthesizing Hypothesis (LLM) ---")
    
    # In a real system, this is where you call OpenAI/Anthropic.
    # Here, we simulate the deterministic reasoning of the LLM based on inputs.
    
    trace = state['traceability_data']
    stats = state['correlation_stats']
    
    if trace.get('batch_status') == 'Unstable' and stats.get('supplier_share', 0) > 0.4:
        hypothesis = f"High probability that unstable batch {trace['batch_id']} from {trace['supplier_name']} is the root cause."
        confidence = 0.85
        citations = [f"Batch:{trace['batch_id']}", f"Supplier:{trace['supplier_name']}"]
    elif trace.get('machine_tension') == 'High':
        hypothesis = "Machine tension was configured to High, likely causing mechanical stress."
        confidence = 0.70
        citations = ["Machine:Machine_Line_1"]
    else:
        hypothesis = "Insufficient evidence to determine a single root cause."
        confidence = 0.40
        citations = []

    return {
        "hypothesis": hypothesis, 
        "confidence": confidence, 
        "evidence_citations": citations
    }

# --- 5. Build the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("fetch_context", node_fetch_context)
workflow.add_node("analyze_correlation", node_analyze_correlation)
workflow.add_node("synthesize_hypothesis", node_synthesize_hypothesis)

workflow.set_entry_point("fetch_context")
workflow.add_edge("fetch_context", "analyze_correlation")
workflow.add_edge("analyze_correlation", "synthesize_hypothesis")
workflow.add_edge("synthesize_hypothesis", END)

app = workflow.compile()

# --- 6. Execution ---
if __name__ == "__main__":
    # Pick a defect from your database to test
    test_defect_id = "DEF-ShadeBand-1" # Note: change this based on your actual data
    
    print(f"🚀 Starting Agentic Diagnosis for {test_defect_id}")
    initial_state = {"defect_id": test_defect_id}
    
    final_state = app.invoke(initial_state)
    
    print("\n✅ DIAGNOSIS COMPLETE")
    print(f"Hypothesis: {final_state['hypothesis']}")
    print(f"Confidence: {final_state['confidence']}")
    print(f"Evidence: {final_state['evidence_citations']}")