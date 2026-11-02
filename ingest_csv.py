import os
import pandas as pd
from neo4j import GraphDatabase

URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def ingest_qc_data(filepath):
    print(f"📂 Reading messy factory data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Entity Resolution: Map messy headers to our Canonical Schema
    df = df.rename(columns={
        'Roll No': 'roll_id',
        'Batch': 'batch_id',
        'Machine': 'machine_id',
        'Defect Type': 'defect_type',
        'Temp_C': 'temperature'
    })
    
    with driver.session() as session:
        # Clean slate for this demo
        session.run("MATCH (n) DETACH DELETE n")
        
        for index, row in df.iterrows():
            if row['defect_type'] == 'No Defect':
                continue # Skip clean rolls for this MVP
                
            print(f"🔗 Linking Defect on Roll {row['roll_id']} -> Batch {row['batch_id']}")
            
            # Write to Neo4j Graph
            query = """
            MERGE (m:Machine {id: $machine_id})
            MERGE (b:DyeLot {id: $batch_id, temperature: $temperature})
            MERGE (r:FabricRoll {id: $roll_id})
            
            // Create the Defect Record
            CREATE (d:DefectRecord {
                defect_id: $defect_id,
                type: 'ShadeBand',
                severity: 'Major'
            })
            
            // Establish Traceability (The Causal Links)
            MERGE (r)-[:PROCESSED_ON]->(m)
            MERGE (r)-[:USED_DYE_BATCH]->(b)
            MERGE (d)-[:CAUSED_BY]->(b)
            MERGE (d)-[:OCCURRED_ON]->(m)
            MERGE (d)-[:DETECTED_ON]->(r)
            """
            
            session.run(query, 
                        machine_id=row['machine_id'],
                        batch_id=row['batch_id'],
                        temperature=float(row['temperature']),
                        roll_id=row['roll_id'],
                        defect_id=f"DEF-{row['roll_id']}")
                        
    print("✅ CSV successfully ingested and mapped into Neo4j.")

if __name__ == "__main__":
    ingest_qc_data("shade_band_qc.csv")