
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

# Setting variables
num_refs = len(data.list_names)

output, variance = ut.PCA_embeddings(data,model_list,10)

# output = ut.TSNE_embeddings(data, model_list, 3)

# ut.plot_PCA(output, data.list_names, clusters=5)
ut.plot_3D_PCA(output, data.list_names, clusters=5)
# ut.plot_3D_PCA_controls(output)

## Similarities loop
#distances_array_joint_raw = ut.similaraties(data, model_list, num_refs)

## Create Raw DataFrame
#df_Distances_joint_raw = ut.convert_arr_to_pandas(distances_array_joint_raw, data)

## Z-Score matrix
#df_Distances_joint_raw = ut.min_max_norm(df_Distances_joint_raw)

## Plots
#ut.plot_dendogram(df_Distances_joint_raw)
#ut.plot_dendrogram_and_heatmap(df_Distances_joint_raw)

# ut.plot_heatmap(df_Distances_joint_raw)

#ut.plot_node_degree(df_Distances_joint_raw)
#
# ut.plot_node_degree(df)

# graph = ut.create_network(data, df_Distances_joint_raw)
#
# graph_properties, global_metrics = ut.graph_properties(graph)
#
# inverted_graph = ut.invert_weights(graph)
#
# graph_lenght = ut.shortest_path(inverted_graph)
#
# ut.plot_chord(graph)
