
import nltk, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from dataset import DatasetLoader
import utils as ut

# nltk.download('wordnet')
# nltk.download('stopwords')

data = DatasetLoader()

model_list = ['sentence-t5-large']
#model_list = ['all-distilroberta-v1']
#model_list = ['all-mpnet-base-v2']
#model_list = ['all-MiniLM-L6-v2']

# Joint scales_raw
data.scales_joint()
data.scramble_joint()

# Setting variables
num_refs = len(data.list_names)

# Summarize
ut.summarize(data, model_list, num_refs)

# Intra-Questionneire variance

#arr = ut.intra_similaraties(data, model_list, num_refs, num_scrambles=50)
#ut.plot_intra_barplot(arr, data)

# Clusters information and plot

# arr = ut.get_embedding(data, model_list, num_refs, num_scrambles=5)
# model = 0
# comps, variance = ut.PCA_embeddings(arr[:,model,:], n_comp=10)
# df = ut.clusters(arr[:, model,:], data, clusters=3)
# ut.plot_3D_PCA(comps, data.list_names, df['cluster'].values)



# Repetitions
#average_dist, std_dist = ut.similaraties_average(data, model_list, num_refs, num_scrambles=10)

# Create Raw DataFrame
#df_Distances_joint_raw = ut.convert_arr_to_pandas(average_dist, data)

# Min Max Scaler
#df_Distances_joint_raw = ut.min_max_norm(df_Distances_joint_raw)

# Plots
#ut.plot_dendogram(df_Distances_joint_raw)

#ut.plot_heatmap(df_Distances_joint_raw.replace(1.0 ,np.nan))

#ut.plot_node_degree(df_Distances_joint_raw)
