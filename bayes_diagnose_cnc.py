import os
import json
import pickle
import pandas as pd
from neo4j import GraphDatabase
from pgmpy.inference import VariableElimination

URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def load_trained_model():
    print("🧠 Loading trained Bayesian Network...")
    with open('learned_model.pkl', 'rb') as f:
        model = pickle.load(f)
    infer = VariableElimination(model)
    return model, infer

def bayesian_diagnose(cycle_id, infer, model):
    print(f"\n🔍 Extracting observed evidence from FaultCycle: {cycle_id}...")
    
    query = """
    MATCH (fc:FaultCycle {cycle_id: $cycle_id})-[r:OBSERVED_STATE]->(v)
    RETURN v.name AS node_name, r.value AS value
    """
    
    evidence = {}
    with driver.session() as session:
        results = session.run(query, cycle_id=cycle_id)
        for record in results:
            node_name = record['node_name']
            value = record['value']
            
            # Only include evidence that is actually in the trained model
            if node_name in model.nodes():
                evidence[node_name] = value
                
    print(f"   Found {len(evidence)} valid sensor observations for inference.")
    
    if not evidence:
        print("❌ No overlapping evidence found between Neo4j and the trained model.")
        return

    # Define the root causes we want to test
    root_causes_to_test = [
        'CLT_Level_lt_Min', 
        'WCL_Unlocked', 
        'LT_Level_Ok', 
        'HP_Pump_Ok', 
        'LP_Pump_Ok'
    ]
    
    valid_targets = [node for node in root_causes_to_test if node in model.nodes()]
    
    print("\n🧮 Running Exact Bayesian Inference...")
    print(f"   --- 🏆 Posterior Probabilities ---")
    
    for target in valid_targets:
        # FIX: If the target is directly observed, we don't need to infer it
        if target in evidence:
            observed_val = evidence[target]
            print(f"   ▶ {target}: DIRECTLY OBSERVED AS '{observed_val}' (100% confidence)")
            continue

        try:
            # Calculate P(Target | Evidence)
            posterior = infer.query(variables=[target], evidence=evidence)
            
            # Extract the probabilities
            states = posterior.state_names[target]
            probs = posterior.values
            
            # Format the output nicely
            prob_str = " | ".join([f"{state}: {prob:.3f}" for state, prob in zip(states, probs)])
            print(f"   ▶ {target}: {prob_str}")
            
        except Exception as e:
            print(f"   ⚠️ Could not infer {target}: {e}")

if __name__ == "__main__":
    cycle_id = "FAULT-exp22-run1" 
    
    try:
        model, infer = load_trained_model()
        bayesian_diagnose(cycle_id, infer, model)
    except FileNotFoundError:
        print("❌ learned_model.pkl not found. Run learn_cpds.py first.")