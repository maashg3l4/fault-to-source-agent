import os
import pickle
from neo4j import GraphDatabase
from pgmpy.inference import VariableElimination

URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def run_hybrid_diagnosis(cycle_id):
    print(f"\n🏥 HYBRID DIAGNOSIS REPORT: {cycle_id}")
    print("="*60)
    
    # --- 1. STRUCTURAL RCA (Graph Traversal) ---
    print("1️⃣ STRUCTURAL ROOT CAUSE (Expert Graph Traversal):")
    # We filter out generic control variables to find the true physical root causes
    graph_query = """
    MATCH (fc:FaultCycle {cycle_id: $cycle_id})-[:OBSERVED_STATE]->(symptom)
    WHERE NOT symptom.name IN ['SL_Red', 'SL_Green', 'SL_Yellow', 'Machine_On', 'Prog_Name', 'MPA_toInitPos']
    
    MATCH path = (root)-[:CAUSES*1..5]->(symptom)
    WHERE NOT ()-[:CAUSES]->(root) 
    AND NOT root:FaultCycle AND NOT root:ProcessCycle
    AND NOT root.name IN ['SL_Red', 'SL_Green', 'SL_Yellow', 'Machine_On', 'Prog_Name', 'MPA_toInitPos']
    
    RETURN DISTINCT root.name AS RootCause, count(DISTINCT symptom) AS SymptomsExplained
    ORDER BY SymptomsExplained DESC LIMIT 3
    """
    
    with driver.session() as session:
        results = session.run(graph_query, cycle_id=cycle_id)
        for r in results:
            print(f"   ▶ {r['RootCause']} (Explains {r['SymptomsExplained']} anomalies)")
            
    # --- 2. STATISTICAL RCA (Bayesian Inference) ---
    print("\n2️⃣ STATISTICAL VALIDATION (Bayesian Network):")
    try:
        with open('learned_model.pkl', 'rb') as f:
            model = pickle.load(f)
        infer = VariableElimination(model)
        
        # Fetch evidence from Neo4j
        evidence = {}
        with driver.session() as session:
            res = session.run("MATCH (fc:FaultCycle {cycle_id: $cycle_id})-[r:OBSERVED_STATE]->(v) RETURN v.name AS name, r.value AS val", cycle_id=cycle_id)
            for record in res:
                if record['name'] in model.nodes():
                    evidence[record['name']] = record['val']
        
        print(f"   (Analyzing {len(evidence)} observed sensor states...)")
        
        # Test the physical root causes that exist in our trained model
        targets = ['LT_Pump_Ok', 'LP_Pump_On', 'HP_Pump_Ok', 'Hyd_Pump_Ok']
        
        for target in targets:
            if target in model.nodes():
                if target in evidence:
                    print(f"   ▶ {target}: DIRECTLY OBSERVED AS '{evidence[target]}'")
                else:
                    posterior = infer.query(variables=[target], evidence=evidence)
                    # Format nicely
                    states = posterior.state_names[target]
                    probs = posterior.values
                    prob_str = " | ".join([f"{s}: {p:.2f}" for s, p in zip(states, probs)])
                    print(f"   ▶ {target}: {prob_str}")
            else:
                print(f"   ⚠️ {target}: Not in trained model (likely dropped due to constant state in training).")
            
    except Exception as e:
        print(f"   ⚠️ Bayesian inference skipped: {e}")

if __name__ == "__main__":
    run_hybrid_diagnosis("FAULT-exp22-run1")