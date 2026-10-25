import os
from neo4j import GraphDatabase
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling

# --- 1. Configuration ---
URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# --- 2. Define the Ground-Truth Causal DAG ---
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

# --- 3. Define ALL CPDs ---
# Root Nodes
cpd_supplier = TabularCPD(variable='Supplier', variable_card=2, values=[[0.6], [0.4]], state_names={'Supplier': ['Supplier_A', 'Supplier_B']})
cpd_tension = TabularCPD(variable='MachineTension', variable_card=2, values=[[0.7], [0.3]], state_names={'MachineTension': ['Optimal', 'High']})
cpd_humidity = TabularCPD(variable='Humidity', variable_card=2, values=[[0.8], [0.2]], state_names={'Humidity': ['Normal', 'High']})
cpd_yarn_quality = TabularCPD(variable='YarnQuality', variable_card=2, values=[[0.75], [0.25]], state_names={'YarnQuality': ['Good', 'Poor']})

# Child Nodes
cpd_batch = TabularCPD(variable='FabricBatch', variable_card=2, values=[[0.7, 0.3], [0.3, 0.7]], evidence=['Supplier'], evidence_card=[2], state_names={'FabricBatch': ['Stable', 'Unstable'], 'Supplier': ['Supplier_A', 'Supplier_B']})
cpd_visual = TabularCPD(variable='VisualMicrostructure', variable_card=2, values=[[0.8, 0.2], [0.2, 0.8]], evidence=['FabricBatch'], evidence_card=[2], state_names={'VisualMicrostructure': ['Normal', 'Degraded'], 'FabricBatch': ['Stable', 'Unstable']})
cpd_dye_lot = TabularCPD(variable='DyeLotQuality', variable_card=2, values=[[0.9, 0.4], [0.1, 0.6]], evidence=['FabricBatch'], evidence_card=[2], state_names={'DyeLotQuality': ['Good', 'Bad'], 'FabricBatch': ['Stable', 'Unstable']})

# Defect Nodes (The Outcomes)
# YarnBreakage has 3 parents: VisualMicrostructure, MachineTension, Humidity (2^3 = 8 combinations)
cpd_yarn_breakage = TabularCPD(
    variable='YarnBreakage', 
    variable_card=2, 
    values=[
        [0.98, 0.90, 0.85, 0.70, 0.60, 0.40, 0.30, 0.10], # P(No) given parents
        [0.02, 0.10, 0.15, 0.30, 0.40, 0.60, 0.70, 0.90]  # P(Yes) given parents
    ], 
    evidence=['VisualMicrostructure', 'MachineTension', 'Humidity'], 
    evidence_card=[2, 2, 2], 
    state_names={'YarnBreakage': ['No', 'Yes'], 'VisualMicrostructure': ['Normal', 'Degraded'], 'MachineTension': ['Optimal', 'High'], 'Humidity': ['Normal', 'High']}
)

cpd_shade_band = TabularCPD(variable='ShadeBand', variable_card=2, values=[[0.9, 0.3, 0.6, 0.1], [0.1, 0.7, 0.4, 0.9]], evidence=['DyeLotQuality', 'MachineTension'], evidence_card=[2, 2], state_names={'ShadeBand': ['No', 'Yes'], 'DyeLotQuality': ['Good', 'Bad'], 'MachineTension': ['Optimal', 'High']})
cpd_pilling = TabularCPD(variable='Pilling', variable_card=2, values=[[0.85, 0.4, 0.5, 0.1], [0.15, 0.6, 0.5, 0.9]], evidence=['Humidity', 'YarnQuality'], evidence_card=[2, 2], state_names={'Pilling': ['No', 'Yes'], 'Humidity': ['Normal', 'High'], 'YarnQuality': ['Good', 'Poor']})

# Add ALL nodes to model
model.add_cpds(
    cpd_supplier, cpd_tension, cpd_humidity, cpd_yarn_quality,
    cpd_batch, cpd_visual, cpd_dye_lot,
    cpd_yarn_breakage, cpd_shade_band, cpd_pilling
)

# --- 4. Data Generation & Neo4j Injection ---
def generate_and_inject(num_samples=100):
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=num_samples)
    
    print(f"Generated {len(samples)} synthetic samples.")
    print(samples.head())

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        for idx, row in samples.iterrows():
            # Determine which defect occurred (if any)
            defects = []
            if row['YarnBreakage'] == 'Yes': defects.append('YarnBreakage')
            if row['ShadeBand'] == 'Yes': defects.append('ShadeBand')
            if row['Pilling'] == 'Yes': defects.append('Pilling')
            
            # If no defect occurred in this sample, skip creating a defect node
            if not defects:
                continue

            # Map SCM variables to Graph nodes/relationships
            query = """
            MERGE (s:Supplier {name: $supplier})
            MERGE (b:FabricBatch {id: $batch_id, status: $batch_status, supplier: $supplier})
            MERGE (m:Machine {id: 'Machine_Line_1', tension: $tension})
            
            // Create the Defect record
            CREATE (d:DefectRecord {
                defect_id: $defect_id,
                type: $defect_type,
                severity: 'Major',
                timestamp: datetime()
            })
            
            // Create Causal Relationships (The Traceability Graph)
            MERGE (s)-[:SUPPLIED]->(b)
            MERGE (d)-[:CAUSED_BY]->(b)
            MERGE (d)-[:OCCURRED_ON]->(m)
            """
            for defect_type in defects:
                session.run(query, 
                            supplier=row['Supplier'],
                            batch_id=f"LOT-{row['FabricBatch']}-{idx}",
                            batch_status=row['FabricBatch'],
                            tension=row['MachineTension'],
                            defect_type=defect_type,
                            defect_id=f"DEF-{defect_type}-{idx}")

    driver.close()
    print("✅ Synthetic data injected into Neo4j.")

if __name__ == "__main__":
    generate_and_inject(num_samples=50)