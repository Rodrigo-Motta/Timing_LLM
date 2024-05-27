
import nltk, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from dataset import DatasetLoader
import utils as ut

# nltk.download('wordnet')
# nltk.download('stopwords')

data = DatasetLoader()

#model_list = ['sentence-t5-large']
#model_list = ['all-distilroberta-v1']
model_list = ['all-mpnet-base-v2']
#model_list = ['all-MiniLM-L6-v2']

# Joint scales_raw
data.scales_joint()
data.scramble_joint()

# Setting variables
num_refs = len(data.list_names)

N = 30
output = np.zeros((len(data.list_names), N))
for scrambles in range(10):
    data.scramble_joint()
    aux,variance = ut.PCA_embeddings(data,model_list,n_comp=N, scramble=True)
    output += aux

output = output/10

# ut.plot_PCA(output, data.list_names, clusters=5)
ut.plot_3D_PCA(output, data.list_names, clusters=5)
# ut.plot_3D_PCA_controls(output)

# output = ut.TSNE_embeddings(data, model_list, 3)

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
