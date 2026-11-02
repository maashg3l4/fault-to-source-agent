# Fault-to-Source Traceability Agent: A Neurosymbolic Causal Engine for Textile Manufacturing

## 🧠 The Novelty: Bridging the Modality Gap & The Raw-First Reality
Traditional causal engines fail in garment manufacturing for two reasons: 
1. The dominant variables (yarn morphology, texture) are visual, not tabular.
2. Factory data is messy, siloed, and incomplete (WhatsApp, Excel, missing sensors).

This project solves both. It is a **Raw-First Neurosymbolic Causal Engine** that ingests messy factory data, resolves entities via a human-in-the-loop, translates visual microstructure into symbolic causal variables, and feeds a Bayesian Causal Network for exact posterior root-cause ranking with **data-completeness audits**.

## 🏗️ Architecture (The 7-Layer SAD)
1.  **Reality (Zone 1):** Raw ingestion of messy CSVs, emails, and QC logs.
2.  **Ingestion & Entity Resolution (Zone 2):** Maps inconsistent headers (e.g., `Roll No` -> `roll_id`) and uses confidence scoring to auto-link or flag for human review.
3.  **Edge Vision (Zone 3 - Optional):** YOLOv8 trained on the BDFD9 dataset (88.8% mAP@50) translates visual pixels into causal JSON events (e.g., `{"VisualMicrostructure": "Degraded"}`).
4.  **Canonical Store & Graph:** Neo4j Traceability Graph models the supply chain (`Supplier -> Batch -> Machine -> Defect`).
5.  **Intelligence (Zone 5):** A deterministic Rule Engine + Bayesian Network (pgmpy) calculates exact posterior probabilities.
6.  **Orchestration:** LangGraph manages the state machine (Fetch Context -> Infer -> Synthesize).
7.  **Action (Zone 6):** Generates a diagnosis with missing-data warnings, enabling human-in-the-loop CAPA decisions.

## 📊 Benchmarks & MVP Validation
-   **Vision Model (YOLOv8n):** 88.8% mAP@50 on 1,088 real factory images (9 defect classes).
-   **Causal Engine:** 100% diagnostic accuracy on the synthetic GCC-Bench (structural consistency validated).
-   **True MVP (Zone 4):** Successfully ingested a messy QC CSV (`shade_band_qc.csv`), mapped temperature (95.0°C) to a causal state (`DyeLotQuality: Bad`), and diagnosed `MachineTension` as the root cause while explicitly flagging missing sensor data.

## 🚀 Quick Start
1. Clone the repo and ensure Docker Desktop is running.
2. Run `docker compose up -d` (Boots Neo4j and the Python app).
3. Run `docker compose exec app python ingest_csv.py` (Ingests the messy factory CSV).
4. Run `docker compose exec app python run_full_pipeline.py` (Executes the end-to-end diagnosis).

## 📂 Project Structure
- `agent_engine.py`: LangGraph state machine & Bayesian inference with missing-data audit.
- `edge_inference.py`: Modality bridge (YOLO to Causal Evidence).
- `run_full_pipeline.py`: Master integration script.
- `ingest_csv.py`: Entity resolution and Neo4j graph builder for messy CSVs.
- `shade_band_qc.csv`: Mock factory data for MVP validation.
- `gcc_generator.py`: Synthetic data generator for the GCC-Bench.

## 🔬 Research Impact
This architecture directly addresses the $39B RMG sector's core inefficiency: manual root-cause triage. By translating messy data into a traceability graph and auditing missing evidence, the system reduces Mean Time to Diagnosis (MTTD) from hours to seconds, providing actionable, evidence-cited interventions for factory managers.