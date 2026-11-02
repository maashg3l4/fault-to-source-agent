import os
import networkx as nx
import pandas as pd
from neo4j import GraphDatabase

# --- Configuration ---
URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def load_expert_graph(gml_path):
    print(f"📂 Loading expert causal graph from {gml_path}...")
    G = nx.read_gml(gml_path)
    
    with driver.session() as session:
        # Clean slate for the new benchmark
        print("🧹 Wiping existing graph data...")
        session.run("MATCH (n) DETACH DELETE n")
        
        # 1. Create Nodes
        print(f"   Creating {G.number_of_nodes()} causal nodes...")
        for node_id, attrs in G.nodes(data=True):
            # Fallback to 'Variable' if no label is provided in the GML
            label = attrs.get('label', 'Variable')
            # Clean up label to be a valid Neo4j label (no spaces)
            label = label.replace(" ", "_").replace("-", "_")
            
            session.run(
                f"MERGE (n:{label} {{id: $id, name: $name}})",
                id=str(node_id), name=attrs.get('name', str(node_id))
            )
            
        # 2. Create Edges
        print(f"   Creating {G.number_of_edges()} causal edges...")
        for source, target, attrs in G.edges(data=True):
            session.run(
                """
                MATCH (a {id: $source}), (b {id: $target})
                MERGE (a)-[:CAUSES]->(b)
                """,
                source=str(source), target=str(target)
            )
    print("✅ Expert Graph successfully loaded into Neo4j.")

def load_real_telemetry(csv_path):
    print(f"📊 Loading real telemetry from {csv_path}...")
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    print(f"   Found {len(df.columns)} columns in the dataset.")
    
    # Take a sample of the first row
    row = df.iloc[0]
    cols = list(df.columns)
    
    # Safely grab the first 3 columns (regardless of data type)
    v1_name = str(cols[0]) if len(cols) > 0 else "None"
    v1 = str(row[cols[0]]) if len(cols) > 0 else "None"
    
    v2_name = str(cols[1]) if len(cols) > 1 else "None"
    v2 = str(row[cols[1]]) if len(cols) > 1 else "None"
    
    v3_name = str(cols[2]) if len(cols) > 2 else "None"
    v3 = str(row[cols[2]]) if len(cols) > 2 else "None"
    
    with driver.session() as session:
        query = """
        CREATE (pc:ProcessCycle {
            cycle_id: 'REAL-OP-28610',
            timestamp: datetime(),
            val_1_name: $v1_name,
            val_1: $v1,
            val_2_name: $v2_name,
            val_2: $v2,
            val_3_name: $v3_name,
            val_3: $v3
        })
        """
        session.run(query, 
                    v1_name=v1_name, v1=v1,
                    v2_name=v2_name, v2=v2,
                    v3_name=v3_name, v3=v3)
        
    print("✅ Real telemetry injected into Neo4j as a ProcessCycle node.")
    print(f"   Sample data recorded: {v1_name}={v1}, {v2_name}={v2}, {v3_name}={v3}")

if __name__ == "__main__":
    # 1. Load the Expert Graph
    gml_file = "data/raw/expert_graph/expert_graph.gml"
    if os.path.exists(gml_file):
        load_expert_graph(gml_file)
    else:
        print(f"❌ Could not find {gml_file}")
        
    # 2. Load the Real Telemetry
    csv_file = "data/raw/real_op/dataset_real_op_28610.csv" 
    if os.path.exists(csv_file):
        load_real_telemetry(csv_file)
    else:
        print(f"❌ Could not find {csv_file}.")