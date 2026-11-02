# Fault-to-Source Traceability Agent: A Neurosymbolic Causal Engine for Textile Manufacturing

## 🧠 The Novelty: Bridging the Modality Gap
Traditional causal engines fail on garment defects because the dominant variables (yarn morphology, texture) are visual, not tabular. This project solves the **Modality Gap** by building an edge-optimized Vision-to-Causal bridge. Visual defect detections are translated into symbolic causal variables, which feed a Bayesian Causal Network for exact posterior root-cause ranking.

## 🏗️ Architecture
The system operates as a stateful, multi-agent pipeline:
1.  **Edge Vision Layer:** YOLOv8n (trained on the BDFD9 Bangladeshi factory dataset) detects visual defects (e.g., DropStitch, Hole).
2.  **Modality Bridge:** Maps YOLO class outputs to symbolic causal states (e.g., `VisualMicrostructure: Degraded`).
3.  **Traceability Layer:** Neo4j Knowledge Graph models the supply chain (`Supplier -> Batch -> Machine -> Defect`).
4.  **Causal Inference Layer:** Discrete Bayesian Network (pgmpy) calculates exact posterior probabilities of root causes.
5.  **Orchestration:** LangGraph manages the state machine (Fetch Context -> Infer -> Synthesize Diagnosis).

## 📊 Benchmarks & Evaluation
-   **Vision Model (YOLOv8n):** 88.8% mAP@50 on 1,088 real factory images (9 defect classes).
-   **Causal Engine:** 100% diagnostic accuracy on the synthetic GCC-Bench (structural consistency validated).
-   **Latency:** ~5.7s CPU inference (Docker), optimized for edge TensorRT/ONNX deployment.

## 🚀 Quick Start
1. Clone the repo and ensure Docker Desktop is running.
2. Run `docker compose up -d` (Boots Neo4j and the Python app).
3. Run `docker compose exec app python gcc_generator.py` (Populates the synthetic benchmark).
4. Run `docker compose exec app python run_full_pipeline.py` (Executes the end-to-end vision-to-diagnosis pipeline).

## 📂 Project Structure
- `agent_engine.py`: LangGraph state machine & Bayesian inference.
- `edge_inference.py`: Modality bridge (YOLO to Causal Evidence).
- `run_full_pipeline.py`: The master integration script.
- `gcc_generator.py`: Synthetic data generator for the GCC-Bench.
- `train_vision.py`: YOLOv8 training script (Colab-compatible).

## 🔬 Research Impact
This architecture directly addresses the $39B RMG sector's core inefficiency: manual root-cause triage. By translating visual microstructure into causal graphs, the system reduces Mean Time to Diagnosis (MTTD) from hours to seconds, providing actionable, evidence-cited interventions for factory managers.