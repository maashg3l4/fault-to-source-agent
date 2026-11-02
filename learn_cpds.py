import os
import glob
import pandas as pd
import networkx as nx
import traceback
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator

def learn_cpds():
    print("🚀 Starting CPD Learning (Robust BDeu Mode)...")

    # 1. Load the Expert Graph
    gml_path = "data/raw/expert_graph/expert_graph.gml"
    G = nx.read_gml(gml_path)
    edges = list(G.edges())
    model = DiscreteBayesianNetwork(edges)
    print(f"📂 Loaded graph: {len(model.nodes())} nodes, {len(model.edges())} edges.")

    # 2. Load Data
    csv_files = glob.glob("data/raw/real_op/*.csv")
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)[['time_s', 'node', 'value']]
            dfs.append(df)
        except:
            pass
    combined_df = pd.concat(dfs, ignore_index=True)

    # 3. Pivot and Fill
    print("🔄 Pivoting event logs to state matrix...")
    state_df = combined_df.pivot_table(index='time_s', columns='node', values='value', aggfunc='first')
    state_df = state_df.ffill().bfill()
    state_df = state_df.astype(str).replace('nan', 'Unknown')

    # 4. Align with Graph
    graph_nodes = set(model.nodes())
    valid_columns = list(graph_nodes.intersection(set(state_df.columns)))
    state_df = state_df[valid_columns]

    # 5. STRICT FILTERING: Remove constant AND high-cardinality columns
    print("🧹 Filtering columns (removing constant and >5 state variables)...")
    cols_to_drop = []
    for col in state_df.columns:
        unique_vals = state_df[col].nunique()
        if unique_vals <= 1 or unique_vals > 5:
            cols_to_drop.append(col)
            
    state_df = state_df.drop(columns=cols_to_drop)
    print(f"   Dropped {len(cols_to_drop)} variables. Remaining: {len(state_df.columns)}")

    # Rebuild model
    model = DiscreteBayesianNetwork(model.edges())
    model.remove_nodes_from([n for n in graph_nodes if n not in state_df.columns])
    print(f"   Final model: {len(model.nodes())} nodes, {len(model.edges())} edges.")

    # 6. Downsample to 10,000 rows (Statistically sufficient, prevents OOM)
    if len(state_df) > 10000:
        state_df = state_df.sample(n=10000, random_state=42).sort_index()
        print(f"✂️ Downsampled to 10,000 timesteps.")

    # 7. Define State Names
    state_names = {col: state_df[col].unique().tolist() for col in state_df.columns}

    # 8. Fit with BayesianEstimator (Robust against missing parent combinations)
    print(f"🧮 Fitting CPDs using BayesianEstimator (BDeu)...")
    try:
        # Use BayesianEstimator instead of MLE. 
        # equivalent_sample_size=5 adds smoothing (like Laplace smoothing) to prevent zero probabilities.
        estimator = BayesianEstimator(model, state_df, state_names=state_names)
        cpds = estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=5)
        
        # Attach to model
        model.add_cpds(*cpds)
        
        print("✅ CPDs successfully learned and attached to the model!")

        # Print a Sample
        sample_node = list(model.nodes())[0]
        print(f"\n--- 📄 Sample Learned CPD for '{sample_node}' ---")
        print(model.get_cpds(sample_node))

        # Save
        import pickle
        with open('learned_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("\n💾 Learned model saved to 'learned_model.pkl'.")

    except Exception as e:
        print("\n❌ CRITICAL ERROR:")
        traceback.print_exc()

if __name__ == "__main__":
    learn_cpds()