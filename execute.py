import fit_model
import data_process
import pandas as pd
import numpy as np
from operator import itemgetter
from sklearn.preprocessing import StandardScaler

# parameters: 1) file 2) cluster number hyperparameter list 3) number of queries per cluster
raw_data = "./data/raw_data_10k_94_02092023.csv"
cluster_number_list = [10, 15, 20, 25, 30]
cluster_size = 5
n_components = 5

print("processing data from raw features into vectors")
print("******************************************************************")
processed_data = data_process.main(raw_data)

print("running kmeans main function...centroids will be returned before finding queries")
print("******************************************************************")
centroids, data_pca = fit_model.main(processed_data, cluster_number_list, n_components)

print("finding queries closes to the centroids...")
print("******************************************************************")
# for each centroid, calculate distance to each other query and select 5 closest to add to workload
workload = []
raw = pd.read_csv(raw_data)
data = pd.DataFrame(data_pca)
for node in centroids:
    distances = []
    for i, row in data.iterrows():
        d = np.linalg.norm(pd.to_numeric(row) - pd.to_numeric(node))        
        distances.append((i, d))
    distances = sorted(distances, key=itemgetter(1))
    cluster = []
    rank = 0
    while (len(cluster) < cluster_size):
        q = distances[rank]
        uuid = raw.iloc[q[0], 0]
        if (uuid not in workload):
            cluster.append(uuid)
        rank += 1
    workload += cluster

# save to file
print("found this many centroids: ", len(centroids))
print("and around those centroids found this many queries: ", len(workload))
res = pd.DataFrame(workload)
res.columns = ['UUID']
pd.DataFrame(res).to_csv("./results/res_" + raw_data[16:], index=False)