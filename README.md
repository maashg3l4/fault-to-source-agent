# Fault-to-Source Traceability Agent: A Neurosymbolic Causal Engine for Industrial Manufacturing

## 🧠 The Novelty: Bridging the Modality Gap & The Raw-First Reality
Traditional causal engines fail in manufacturing for two reasons:
1. The dominant variables (yarn morphology, vibration, texture) are visual/time-series, not flat tabular data.
2. Factory data is messy, siloed, and incomplete (WhatsApp, Excel, missing sensors).

This project solves both. It is a **Raw-First Neurosymbolic Causal Engine** that ingests messy factory data, resolves entities, translates visual microstructure into symbolic causal variables, and feeds a Bayesian Causal Network for exact posterior root-cause ranking with **data-completeness audits**.

## 🏗️ Architecture (The 7-Layer SAD)
1. **Reality (Zone 1):** Raw ingestion of messy CSVs, emails, and QC logs.
2. **Ingestion & Entity Resolution (Zone 2):** Maps inconsistent headers (e.g., `Roll No` -> `roll_id`) and uses confidence scoring to auto-link or flag for human review.
3. **Edge Vision (Zone 3 - Optional):** YOLOv8 trained on the BDFD9 dataset (88.8% mAP@50) translates visual pixels into causal JSON events.
4. **Canonical Store & Graph:** Neo4j Traceability Graph models the supply chain (`Supplier -> Batch -> Machine -> Defect`).
5. **Intelligence (Zone 5):** A deterministic Rule Engine + Bayesian Network (pgmpy) calculates exact posterior probabilities.
6. **Orchestration:** LangGraph manages the state machine (Fetch Context -> Infer -> Synthesize).
7. **Action (Zone 6):** Generates a diagnosis with missing-data warnings, enabling human-in-the-loop CAPA decisions.

## 🚀 Key Engineering Achievements & Benchmarks

### 1. Textile True MVP (Zone 4 Validation)
Successfully validated the end-to-end pipeline on a messy, real-world factory QC CSV (`shade_band_qc.csv`).
- **Ingestion & Entity Resolution:** Maps messy headers and builds the Neo4j traceability graph.
- **Feature Engineering:** Converts continuous process variables (e.g., `Temperature = 95.0°C`) into discrete causal states (`DyeLotQuality: Bad`).
- **Data Completeness Audit:** The agent explicitly flags missing sensor data in the final hypothesis (e.g., `⚠️ WARNING: Missing MachineTension_Sensor`), ensuring the Quality Manager knows exactly what evidence is missing before making a CAPA decision.

### 2. Industrial Benchmark Validation (causRCA Dataset)
To prove the architecture is domain-agnostic, the engine was validated against the **causRCA dataset** (an expert-validated industrial CNC machining benchmark with 92 variables and 104 causal edges).
- **Graph Ingestion:** Loaded the 92-node expert causal graph into Neo4j.
- **Digital Twin Reconstruction:** Reconstructed a real-time machine state from event-based telemetry (54 active variables at the time of alarm).
- **Advanced RCA:** Engineered a Cypher-based traversal algorithm that filters out generic alarms (e.g., stack lights) and ranks root causes by **Explanatory Power** (the number of specific anomalies they explain).

**Sample causRCA Output (Coolant System Fault Scenario):**
*   `MPA_toInitPos` (Explains 9 cascading reset anomalies)
*   `HP_Pump_Ok` (Explains 4 pressure anomalies)
*   `CLT_Level_lt_Min` (Explains 2 direct coolant anomalies) ✅ *Correctly identifies the fault domain.*

### 3. Vision Model Performance
- **Model:** YOLOv8n (Edge-optimized)
- **Dataset:** BDFD9 (1,088 images from 4 real Bangladeshi factories)
- **Metrics:** 88.8% mAP@50 overall. Top classes: DropStitch (96.5%), Setup (96.4%). Target: Deploy via TensorRT FP16 (<30ms latency on edge devices).

## 🛠️ Tech Stack
- **Infrastructure:** Docker, Docker Compose, WSL2
- **Databases:** Neo4j (Graph), PostgreSQL (Relational - planned)
- **AI/ML:** YOLOv8 (Ultralytics), PyTorch, pgmpy (Bayesian Inference), LangGraph (Agentic Orchestration), scikit-learn, pandas
- **Backend/API:** FastAPI (planned), Streamlit (planned)

## 🚀 Quick Start
1. Clone the repo and ensure Docker Desktop is running.
2. Run `docker compose up -d` (Boots Neo4j and the Python app).
3. **Run the Textile MVP:**
   - `docker compose exec app python ingest_csv.py` (Ingests messy factory CSV)
   - `docker compose exec app python run_full_pipeline.py` (Executes diagnosis with missing-data audit)
4. **Run the causRCA Industrial Benchmark:**
   - `docker compose exec app python ingest_causrca.py` (Loads 92-node graph and telemetry)
   - `docker compose exec app python diagnose_cnc_fault.py` (Traces causal paths for CNC faults)

## 📂 Project Structure
- `agent_engine.py`: LangGraph state machine & Bayesian inference with missing-data audit.
- `edge_inference.py`: Modality bridge (YOLO to Causal Evidence).
- `ingest_csv.py` / `run_full_pipeline.py`: Textile MVP scripts.
- `ingest_causrca.py` / `diagnose_cnc_fault.py`: Industrial CNC benchmark scripts.
- `train_vision.py`: YOLOv8 training script (Colab-compatible).
- `docker-compose.yml`: Container orchestration.

## 📈 Future Roadmap
1. **Counterfactual Engine (L3):** Implement `do`-calculus in `pgmpy` to simulate interventions (e.g., "If we recalibrate tension, defect probability drops 42%").
2. **Streamlit Dashboard:** Build the Human-in-the-Loop UI for Quality Managers.
3. **Edge Optimization:** Quantize YOLOv8 to TensorRT for Jetson deployment.