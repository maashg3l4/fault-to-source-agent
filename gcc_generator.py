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
cpd_supplier = TabularCPD(variable='Supplier', variable_card=2, values=[[0.6], [0.4]], state_names={'Supplier': ['Supplier_A', 'Supplier_B']})
cpd_tension = TabularCPD(variable='MachineTension', variable_card=2, values=[[0.7], [0.3]], state_names={'MachineTension': ['Optimal', 'High']})
cpd_humidity = TabularCPD(variable='Humidity', variable_card=2, values=[[0.8], [0.2]], state_names={'Humidity': ['Normal', 'High']})
cpd_yarn_quality = TabularCPD(variable='YarnQuality', variable_card=2, values=[[0.75], [0.25]], state_names={'YarnQuality': ['Good', 'Poor']})
cpd_batch = TabularCPD(variable='FabricBatch', variable_card=2, values=[[0.7, 0.3], [0.3, 0.7]], evidence=['Supplier'], evidence_card=[2], state_names={'FabricBatch': ['Stable', 'Unstable'], 'Supplier': ['Supplier_A', 'Supplier_B']})
cpd_visual = TabularCPD(variable='VisualMicrostructure', variable_card=2, values=[[0.8, 0.2], [0.2, 0.8]], evidence=['FabricBatch'], evidence_card=[2], state_names={'VisualMicrostructure': ['Normal', 'Degraded'], 'FabricBatch': ['Stable', 'Unstable']})
cpd_dye_lot = TabularCPD(variable='DyeLotQuality', variable_card=2, values=[[0.9, 0.4], [0.1, 0.6]], evidence=['FabricBatch'], evidence_card=[2], state_names={'DyeLotQuality': ['Good', 'Bad'], 'FabricBatch': ['Stable', 'Unstable']})

cpd_yarn_breakage = TabularCPD(
    variable='YarnBreakage', variable_card=2, 
    values=[[0.98, 0.90, 0.85, 0.70, 0.60, 0.40, 0.30, 0.10], [0.02, 0.10, 0.15, 0.30, 0.40, 0.60, 0.70, 0.90]], 
    evidence=['VisualMicrostructure', 'MachineTension', 'Humidity'], evidence_card=[2, 2, 2], 
    state_names={'YarnBreakage': ['No', 'Yes'], 'VisualMicrostructure': ['Normal', 'Degraded'], 'MachineTension': ['Optimal', 'High'], 'Humidity': ['Normal', 'High']}
)
cpd_shade_band = TabularCPD(variable='ShadeBand', variable_card=2, values=[[0.9, 0.3, 0.6, 0.1], [0.1, 0.7, 0.4, 0.9]], evidence=['DyeLotQuality', 'MachineTension'], evidence_card=[2, 2], state_names={'ShadeBand': ['No', 'Yes'], 'DyeLotQuality': ['Good', 'Bad'], 'MachineTension': ['Optimal', 'High']})
cpd_pilling = TabularCPD(variable='Pilling', variable_card=2, values=[[0.85, 0.4, 0.5, 0.1], [0.15, 0.6, 0.5, 0.9]], evidence=['Humidity', 'YarnQuality'], evidence_card=[2, 2], state_names={'Pilling': ['No', 'Yes'], 'Humidity': ['Normal', 'High'], 'YarnQuality': ['Good', 'Poor']})

model.add_cpds(cpd_supplier, cpd_tension, cpd_humidity, cpd_yarn_quality, cpd_batch, cpd_visual, cpd_dye_lot, cpd_yarn_breakage, cpd_shade_band, cpd_pilling)

# --- 4. Data Generation & Neo4j Injection ---
def generate_and_inject(num_samples=100):
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=num_samples)
    
    print(f"Generated {len(samples)} synthetic samples.")
    
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        # Wipe existing data for a clean benchmark state
        print("🧹 Wiping existing graph data...")
        session.run("MATCH (n) DETACH DELETE n")

        # Inject data
        for idx, row in samples.iterrows():
            defects = []
            if row['YarnBreakage'] == 'Yes': defects.append('YarnBreakage')
            if row['ShadeBand'] == 'Yes': defects.append('ShadeBand')
            if row['Pilling'] == 'Yes': defects.append('Pilling')
            
            if not defects:
                continue

            # Multimodal Enriched Cypher Query
            query = """
            // 1. Merge Entities (Nodes)
            MERGE (s:Supplier {name: $supplier})
            MERGE (b:FabricBatch {id: $batch_id, status: $batch_status})
            MERGE (m:Machine {id: 'Machine_Line_1', tension: $tension})
            MERGE (e:Environment {humidity: $humidity})
            MERGE (mat:Material {yarn_quality: $yarn_quality})
            MERGE (dye:DyeLot {quality: $dye_lot_quality})
            
            // 2. Create the Event (Defect)
            CREATE (d:DefectRecord {
                defect_id: $defect_id,
                type: $defect_type,
                severity: 'Major',
                timestamp: datetime()
            })
            
            // 3. Establish Topology (Traceability Links)
            MERGE (s)-[:SUPPLIED]->(b)
            MERGE (b)-[:HAS_MATERIAL]->(mat)
            MERGE (b)-[:HAS_DYE_LOT]->(dye)
            
            // 4. Establish Causal Links (Edges from SCM)
            MERGE (d)-[:CAUSED_BY]->(b)
            MERGE (d)-[:OCCURRED_ON]->(m)
            MERGE (d)-[:OCCURRED_IN]->(e)
            MERGE (d)-[:INVOLVES_MATERIAL]->(mat)
            MERGE (d)-[:AFFECTED_BY_DYE]->(dye)
            """
            
            for defect_type in defects:
                session.run(query, 
                            supplier=row['Supplier'],
                            batch_id=f"LOT-{row['FabricBatch']}-{idx}",
                            batch_status=row['FabricBatch'],
                            tension=row['MachineTension'],
                            humidity=row['Humidity'],
                            yarn_quality=row['YarnQuality'],
                            dye_lot_quality=row['DyeLotQuality'],
                            defect_type=defect_type,
                            defect_id=f"DEF-{defect_type}-{idx}")

    driver.close()
    print("✅ Multimodal synthetic data injected into Neo4j.")

if __name__ == "__main__":
    generate_and_inject(num_samples=100)