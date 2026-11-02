import json
import os
from neo4j import GraphDatabase
from agent_engine import app as causal_agent

# Configuration
URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def get_latest_defect_id():
    """Fetches a real ShadeBand defect ID from the newly ingested CSV data."""
    # FIX: Query for 'ShadeBand' instead of 'YarnBreakage'
    query = "MATCH (d:DefectRecord {type: 'ShadeBand'}) RETURN d.defect_id AS id LIMIT 1"
    with driver.session() as session:
        result = session.run(query).single()
        if result:
            return result['id']
    return None

def run_end_to_end():
    print("=== STEP 1: Vision Layer (Simulated) ===")
    # In a real scenario, this would be the YOLO output. 
    # For this MVP, we simulate a visual detection of ShadeBand.
    vision_payload = {
        "defect_type": "ShadeBand",
        "confidence": 0.85,
        "causal_evidence": {
            "DyeLotQuality": "Bad" # Matches ShadeBand SCM
        }
    }
    print(f"Visual Evidence Ingested: {json.dumps(vision_payload['causal_evidence'])}")

    print("\n=== STEP 2: Traceability Layer ===")
    defect_id = get_latest_defect_id()
    if not defect_id:
        print("❌ No ShadeBand defects found in Neo4j. Run ingest_csv.py first.")
        return
    print(f"Traceability Context Found: Defect ID {defect_id}")

    print("\n=== STEP 3: Causal Inference Layer ===")
    # Map to the SCM defect type
    scm_defect_type = "ShadeBand" 
    
    print(f"Invoking LangGraph Causal Agent for {scm_defect_type}...")
    
    # Invoke the LangGraph agent
    final_state = causal_agent.invoke({"defect_id": defect_id})
    
    print("\n=== FINAL DIAGNOSIS REPORT ===")
    print(f"Visual Detection: {vision_payload['defect_type']} (Conf: {vision_payload['confidence']})")
    print(f"Root Cause:       {final_state.get('hypothesis')}")
    print(f"Confidence:       {final_state.get('confidence'):.2f}")
    print(f"Evidence Cited:   {final_state.get('evidence_citations')}")
    
    driver.close()

if __name__ == "__main__":
    run_end_to_end()