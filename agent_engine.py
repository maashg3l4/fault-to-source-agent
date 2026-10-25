import os
from typing import TypedDict, List, Dict
from neo4j import GraphDatabase
from langgraph.graph import StateGraph, END
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# --- 1. Configuration ---
URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# --- 2. Reconstruct the Ground-Truth Causal Network for Inference ---
model = DiscreteBayesianNetwork([
    ('Supplier', 'FabricBatch'),
    ('FabricBatch', 'VisualMicrostructure'),
    ('FabricBatch', 'DyeLotQuality'),
    ('VisualMicrostructure', 'YarnBreakage'),
    ('DyeLotQuality', 'ShadeBand'),
    ('MachineTension', 'YarnBreakage'),
    ('Humidity', 'YarnBreakage'),
    ('MachineTension', 'ShadeBand'),
    ('Humidity', 'Pilling'),
    ('YarnQuality', 'Pilling')
])

cpd_supplier = TabularCPD(variable='Supplier', variable_card=2, values=[[0.6], [0.4]], state_names={'Supplier': ['Supplier_A', 'Supplier_B']})
cpd_tension = TabularCPD(variable='MachineTension', variable_card=2, values=[[0.7], [0.3]], state_names={'MachineTension': ['Optimal', 'High']})
cpd_humidity = TabularCPD(variable='Humidity', variable_card=2, values=[[0.8], [0.2]], state_names={'Humidity': ['Normal', 'High']})
cpd_yarn_quality = TabularCPD(variable='YarnQuality', variable_card=2, values=[[0.75], [0.25]], state_names={'YarnQuality': ['Good', 'Poor']})
cpd_batch = TabularCPD(variable='FabricBatch', variable_card=2, values=[[0.7, 0.3], [0.3, 0.7]], evidence=['Supplier'], evidence_card=[2], state_names={'FabricBatch': ['Stable', 'Unstable'], 'Supplier': ['Supplier_A', 'Supplier_B']})
cpd_visual = TabularCPD(variable='VisualMicrostructure', variable_card=2, values=[[0.8, 0.2], [0.2, 0.8]], evidence=['FabricBatch'], evidence_card=[2], state_names={'VisualMicrostructure': ['Normal', 'Degraded'], 'FabricBatch': ['Stable', 'Unstable']})
cpd_dye_lot = TabularCPD(variable='DyeLotQuality', variable_card=2, values=[[0.9, 0.4], [0.1, 0.6]], evidence=['FabricBatch'], evidence_card=[2], state_names={'DyeLotQuality': ['Good', 'Bad'], 'FabricBatch': ['Stable', 'Unstable']})
cpd_yarn_breakage = TabularCPD(variable='YarnBreakage', variable_card=2, values=[[0.98, 0.90, 0.85, 0.70, 0.60, 0.40, 0.30, 0.10], [0.02, 0.10, 0.15, 0.30, 0.40, 0.60, 0.70, 0.90]], evidence=['VisualMicrostructure', 'MachineTension', 'Humidity'], evidence_card=[2, 2, 2], state_names={'YarnBreakage': ['No', 'Yes'], 'VisualMicrostructure': ['Normal', 'Degraded'], 'MachineTension': ['Optimal', 'High'], 'Humidity': ['Normal', 'High']})
cpd_shade_band = TabularCPD(variable='ShadeBand', variable_card=2, values=[[0.9, 0.3, 0.6, 0.1], [0.1, 0.7, 0.4, 0.9]], evidence=['DyeLotQuality', 'MachineTension'], evidence_card=[2, 2], state_names={'ShadeBand': ['No', 'Yes'], 'DyeLotQuality': ['Good', 'Bad'], 'MachineTension': ['Optimal', 'High']})
cpd_pilling = TabularCPD(variable='Pilling', variable_card=2, values=[[0.85, 0.4, 0.5, 0.1], [0.15, 0.6, 0.5, 0.9]], evidence=['Humidity', 'YarnQuality'], evidence_card=[2, 2], state_names={'Pilling': ['No', 'Yes'], 'Humidity': ['Normal', 'High'], 'YarnQuality': ['Good', 'Poor']})

model.add_cpds(cpd_supplier, cpd_tension, cpd_humidity, cpd_yarn_quality, cpd_batch, cpd_visual, cpd_dye_lot, cpd_yarn_breakage, cpd_shade_band, cpd_pilling)
infer = VariableElimination(model)

# --- 3. Define State ---
class AgentState(TypedDict):
    defect_id: str
    defect_type: str
    traceability_data: Dict
    causal_probabilities: Dict
    hypothesis: str
    confidence: float
    evidence_citations: List[str]

# --- 4. Define Deterministic Tools ---
def fetch_traceability_data(defect_id: str) -> Dict:
    query = """
    MATCH (d:DefectRecord {defect_id: $defect_id})-[:CAUSED_BY]->(b:FabricBatch)
    MATCH (b)<-[:SUPPLIED]-(s:Supplier)
    MATCH (d)-[:OCCURRED_ON]->(m:Machine)
    MATCH (d)-[:OCCURRED_IN]->(e:Environment)
    MATCH (d)-[:INVOLVES_MATERIAL]->(mat:Material)
    MATCH (d)-[:AFFECTED_BY_DYE]->(dye:DyeLot)
    RETURN d.type AS defect_type, b.status AS batch_status, s.name AS supplier_name, 
           m.tension AS tension, e.humidity AS humidity, 
           mat.yarn_quality AS yarn_quality, dye.quality AS dye_lot_quality
    """
    with driver.session() as session:
        result = session.run(query, defect_id=defect_id).single()
        return dict(result) if result else {}

# --- 5. Define Graph Nodes ---
def node_fetch_context(state: AgentState):
    data = fetch_traceability_data(state['defect_id'])
    return {"traceability_data": data, "defect_type": data.get('defect_type', 'Unknown')}

def node_causal_inference(state: AgentState):
    defect_type = state['defect_type']
    evidence = {defect_type: 'Yes'}
    
    if defect_type == 'YarnBreakage':
        query_vars = ['VisualMicrostructure', 'MachineTension', 'Humidity']
    elif defect_type == 'ShadeBand':
        query_vars = ['DyeLotQuality', 'MachineTension']
    elif defect_type == 'Pilling':
        query_vars = ['Humidity', 'YarnQuality']
    else:
        return {"causal_probabilities": {}}

    probs = {}
    for var in query_vars:
        posterior = infer.query(variables=[var], evidence=evidence)
        probs[var] = float(posterior.values[1]) # Probability of the "Bad" state
        
    return {"causal_probabilities": probs}

def node_synthesize_hypothesis(state: AgentState):
    trace = state['traceability_data']
    probs = state['causal_probabilities']
    
    if not probs:
        return {"hypothesis": "Unknown defect type.", "confidence": 0.0, "evidence_citations": []}

    top_cause = max(probs, key=probs.get)
    confidence = probs[top_cause]
    
    trace_key = top_cause.lower()
    if top_cause == 'DyeLotQuality': trace_key = 'dye_lot_quality'
    elif top_cause == 'YarnQuality': trace_key = 'yarn_quality'
    elif top_cause == 'VisualMicrostructure': trace_key = 'batch_status'
    
    observed_value = trace.get(trace_key, 'Unknown')
    
    hypothesis = f"Bayesian inference identifies {top_cause} as the primary driver (p={confidence:.2f}). Observed state: {observed_value}."
    citations = [f"{top_cause}: {observed_value}"]

    return {"hypothesis": hypothesis, "confidence": confidence, "evidence_citations": citations}

# --- 6. Build the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("fetch_context", node_fetch_context)
workflow.add_node("causal_inference", node_causal_inference)
workflow.add_node("synthesize_hypothesis", node_synthesize_hypothesis)

workflow.set_entry_point("fetch_context")
workflow.add_edge("fetch_context", "causal_inference")
workflow.add_edge("causal_inference", "synthesize_hypothesis")
workflow.add_edge("synthesize_hypothesis", END)

app = workflow.compile()