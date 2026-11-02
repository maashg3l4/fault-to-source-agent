import pandas as pd
import glob
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import networkx as nx

# 1. Load the expert graph structure
G = nx.read_gml("data/raw/causRCA/expert_graph/expert_graph.gml")
model = DiscreteBayesianNetwork(G.edges())

# 2. Load and concatenate real_op data
all_files = glob.glob("data/raw/causRCA/real_op/*.csv")
df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)

# 3. Discretize continuous columns (example: equal-frequency binning)
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = pd.qcut(df[col], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')

# 4. Fit the CPDs
model.fit(df, estimator=MaximumLikelihoodEstimator)

# 5. Save the learned model (optional)
# You can serialize the model or save the CPDs to JSON for reproducibility
print("✅ CPDs learned and fitted from real operational data.")