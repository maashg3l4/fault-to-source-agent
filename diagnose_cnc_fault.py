import os
import json
import pandas as pd
from neo4j import GraphDatabase

URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def load_fault_scenario(fault_dir_path, cycle_id):
    print(f"\n🚨 Loading FAULT Scenario from {fault_dir_path}...")
    
    causes_path = os.path.join(fault_dir_path, "causes.json")
    with open(causes_path, 'r') as f:
        timeline = json.load(f)
    
    alarm_time = timeline.get('alarms_detected_at')
    print(f"   ⏱️ Alarm triggered at: {alarm_time}s")

    fault_type = "Coolant System Fault" if "coolant" in fault_dir_path.lower() else "Unknown"

    csv_files = [f for f in os.listdir(fault_dir_path) if f.endswith('.csv')]
    csv_path = os.path.join(fault_dir_path, csv_files[0])
    df = pd.read_csv(csv_path)
    
    df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
    df_before_alarm = df[df['time_s'] <= alarm_time]
    df_before_alarm['node'] = df_before_alarm['node'].astype(str).str.strip()
    machine_state = df_before_alarm.sort_values('time_s').drop_duplicates('node', keep='last')
    
    print(f"   📸 Reconstructed state: {len(machine_state)} variables.")

    with driver.session() as session:
        session.run("""
            MERGE (fc:FaultCycle {cycle_id: $cycle_id})
            SET fc.alarm_time = $time_s, fc.fault_type = $ft
        """, cycle_id=cycle_id, time_s=float(alarm_time), ft=fault_type)
        
        linked_count = 0
        for _, row in machine_state.iterrows():
            node_name = row['node']
            value = str(row['value'])
            
            query = """
            MATCH (v)
            WHERE v.name IS NOT NULL AND toLower(trim(v.name)) = toLower($node_name)
            AND NOT v:FaultCycle AND NOT v:ProcessCycle
            MATCH (fc:FaultCycle {cycle_id: $cycle_id})
            MERGE (fc)-[:OBSERVED_STATE {value: $value}]->(v)
            RETURN v.name
            """
            result = session.run(query, node_name=node_name, cycle_id=cycle_id, value=value)
            if result.single():
                linked_count += 1
                
        print(f"   ✅ Successfully linked {linked_count} variables.")

def trace_root_cause(cycle_id):
    print(f"\n🔍 Tracing Causal Paths for {cycle_id}...")
    
    # ADVANCED RCA LOGIC:
    # 1. Exclude generic alarms (SL_Red, Prog_Name, etc.) from being symptoms.
    # 2. Find root causes that explain the maximum number of specific symptoms.
    query = """
    MATCH (fc:FaultCycle {cycle_id: $cycle_id})-[r:OBSERVED_STATE]->(symptom)
    // Filter out generic stack lights and machine states
    WHERE NOT symptom.name IN ['SL_Red', 'SL_Green', 'SL_Yellow', 'Machine_On', 'Prog_Name', 'Prog_A_701330', 'CBC_Closed', 'CBC_open']
    
    MATCH path = (root)-[:CAUSES*1..5]->(symptom)
    // Root must be an actual root (no parents) and not a generic node
    WHERE NOT ()-[:CAUSES]->(root) 
    AND NOT root:FaultCycle AND NOT root:ProcessCycle
    AND NOT root.name IN ['SL_Red', 'SL_Green', 'SL_Yellow', 'Machine_On', 'Prog_Name', 'Prog_A_701330']
    
    // Rank by how many distinct specific symptoms this root explains
    RETURN root.name AS RootCause, 
           count(DISTINCT symptom) AS SymptomsExplained,
           collect(DISTINCT symptom.name)[0..3] AS ExampleSymptoms
    ORDER BY SymptomsExplained DESC, RootCause ASC
    LIMIT 5
    """
    
    with driver.session() as session:
        results = session.run(query, cycle_id=cycle_id)
        
        print("   --- 🏆 Top Suspected Root Causes ---")
        found = False
        for record in results:
            found = True
            print(f"   ▶ {record['RootCause']} (Explains {record['SymptomsExplained']} anomalies)")
            print(f"     └─ Evidence: Leads to {record['ExampleSymptoms']}")
            
        if not found:
            print("   ⚠️ No causal path found to specific symptoms.")

if __name__ == "__main__":
    fault_scenario = "data/raw/dig_twin/exp_coolant/exp_22/run_1" 
    
    if os.path.exists(fault_scenario):
        cycle_id = "FAULT-exp22-run1"
        load_fault_scenario(fault_scenario, cycle_id)
        trace_root_cause(cycle_id)
    else:
        print(f"❌ Could not find fault scenario at {fault_scenario}")