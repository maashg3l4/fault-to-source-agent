import os
from neo4j import GraphDatabase
from agent_engine import app 

URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def run_evaluation():
    print("🧪 Starting Agent Evaluation on GCC-Bench...")
    
    with driver.session() as session:
        result = session.run("MATCH (d:DefectRecord) RETURN d.defect_id AS id, d.type AS type")
        defects = [record.data() for record in result]

    if not defects:
        print("❌ No defects found.")
        return

    correct_diagnoses = 0
    total_defects = len(defects)

    for defect in defects:
        defect_id = defect['id']
        
        final_state = app.invoke({"defect_id": defect_id})
        
        hypothesis = final_state.get('hypothesis', '')
        confidence = final_state.get('confidence', 0.0)
        
        # Extract the top cause from the hypothesis string
        top_cause = ""
        if "identifies " in hypothesis:
            top_cause = hypothesis.split("identifies ")[1].split(" as")[0]
            
        is_correct = False
        # Check if the top cause is a valid parent in the causal graph
        if defect['type'] == 'ShadeBand' and top_cause in ['DyeLotQuality', 'MachineTension']:
            is_correct = True
        elif defect['type'] == 'YarnBreakage' and top_cause in ['VisualMicrostructure', 'MachineTension', 'Humidity']:
            is_correct = True
        elif defect['type'] == 'Pilling' and top_cause in ['Humidity', 'YarnQuality']:
            is_correct = True

        if is_correct:
            correct_diagnoses += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        print(f"[{status}] {defect_id} ({defect['type']}) -> Conf: {confidence:.2f} | Hyp: {hypothesis[:70]}...")

    accuracy = (correct_diagnoses / total_defects) * 100
    print("\n📊 EVALUATION RESULTS")
    print(f"Total Defects Analyzed: {total_defects}")
    print(f"Correct Diagnoses: {correct_diagnoses}")
    print(f"Agent Accuracy: {accuracy:.2f}%")
    
    driver.close()

if __name__ == "__main__":
    run_evaluation()