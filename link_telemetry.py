import os
import pandas as pd
from neo4j import GraphDatabase

URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def link_telemetry_to_graph(csv_path):
    print(f"📊 Linking telemetry to Causal Graph from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Filter for the first timestamp to get a snapshot of the machine
    first_time = df['time_s'].min()
    snapshot = df[df['time_s'] == first_time]
    
    print(f"   Taking a snapshot at time_s = {first_time} ({len(snapshot)} sensor readings)")
    
    with driver.session() as session:
        # Create the ProcessCycle node for this timestamp
        session.run("""
            MERGE (pc:ProcessCycle {cycle_id: $cycle_id})
            SET pc.time_s = $time_s
        """, cycle_id=f"REAL-OP-{first_time}", time_s=float(first_time))
        
        linked_count = 0
        
        # 2. Iterate through the sensor readings and link to the graph
        for _, row in snapshot.iterrows():
            node_name = str(row['node'])
            value = str(row['value'])
            
            # Match the node in Neo4j based on its name
            # (Note: We use the 'name' property, as that's how the GML was loaded)
            query = """
            MATCH (v:Variable {name: $node_name})
            MATCH (pc:ProcessCycle {cycle_id: $cycle_id})
            MERGE (pc)-[:OBSERVED_STATE {value: $value, time_s: $time_s}]->(v)
            RETURN v.name
            """
            result = session.run(query, 
                                 node_name=node_name, 
                                 cycle_id=f"REAL-OP-{first_time}", 
                                 value=value, 
                                 time_s=float(first_time))
            
            if result.single():
                linked_count += 1

    print(f"✅ Successfully linked {linked_count} sensor readings to the Causal Graph.")

if __name__ == "__main__":
    # Ensure this matches the file we used in the previous step
    csv_file = "data/raw/real_op/dataset_real_op_28610.csv" 
    if os.path.exists(csv_file):
        link_telemetry_to_graph(csv_file)
    else:
        print(f"❌ Could not find {csv_file}.")