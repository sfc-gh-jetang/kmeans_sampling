import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans


def load_embeddings(data):
    """
    Loading the query dataset in pandas dataframe
    :return: scaled data
    """
    # loading wine dataset
    q_raw = pd.read_csv(data)

    # checking data shape
    row, col = q_raw.shape
    print(f'There are {row} rows and {col} columns') 
    print(q_raw.head(10))

    # to work on copy of the data
    q_raw_scaled = q_raw.copy()

    # Scaling the data to keep the different attributes in same range.
    q_raw_scaled[q_raw_scaled.columns] = StandardScaler().fit_transform(q_raw_scaled)
    print(q_raw_scaled.describe())

    return q_raw_scaled


def pca_embeddings(df_scaled, n_components):
    """To reduce the dimensions of the query dataset we use Principal Component Analysis (PCA).
    :param df_scaled: scaled data
    :return: pca result, pca for plotting graph
    """

    pca = PCA(n_components)
    pca_result = pca.fit_transform(df_scaled)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    print('Cumulative variance explained by ' + str(n_components) +' principal components: {:.2%}'.format(
        np.sum(pca.explained_variance_ratio_)))

    # Results from pca.components_
    index = []
    for i in range(n_components):
        index.append("PC_" + str(i + 1))
    dataset_pca = pd.DataFrame(abs(pca.components_), columns=df_scaled.columns, index=index)
    print('\n\n', dataset_pca)
    
    print('\n\n Most important features:')
    print(dataset_pca.idxmax(axis=1))
    
    return pca_result, pca

def kmean_hyper_param_tuning(data, parameter_list):
    """
    Hyper parameter tuning to select the best from all the parameters on the basis of silhouette_score.
    :param data: dimensionality reduced data after applying PCA
    :return: best number of clusters for the model (used for KMeans n_clusters)
    """
    # candidate values for our number of cluster
    parameters = parameter_list

    # instantiating ParameterGrid, pass number of clusters as input
    parameter_grid = ParameterGrid({'n_clusters': parameters})

    best_score = -1
    kmeans_model = KMeans(init='k-means++')     # instantiating KMeans model
    silhouette_scores = []

    # evaluation based on silhouette_score
    for p in parameter_grid:
        kmeans_model.set_params(**p)    # set current hyper parameter
        kmeans_model.fit(data)          # fit model on query dataset, this will find clusters based on parameter p

        ss = metrics.silhouette_score(data, kmeans_model.labels_)   # calculate silhouette_score
        silhouette_scores += [ss]       # store all the scores

        print('Parameter:', p, 'Score', ss)

        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p

    return best_grid['n_clusters']


def visualizing_results(pca_result, label, centroids_pca):
    """ Visualizing the clusters
    :param pca_result: PCA applied data
    :param label: K Means labels
    :param centroids_pca: PCA format K Means centroids
    """
    # ------------------ Using Matplotlib for plotting-----------------------
    x = pca_result[:, 0]
    y = pca_result[:, 1]

    plt.scatter(x, y, c=label, alpha=0.5, s=200)  # plot different colors per cluster
    plt.title('query clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, linewidths=1.5,
                color='red', edgecolors="black", lw=1.5)

    plt.show()


def get_results(centroids_pca):
    print(centroids_pca)


def main(data, parameter_list, n_components):
    print("1. Loading Query dataset\n")
    data_scaled = load_embeddings(data)
    
    # pd.DataFrame(data_scaled).to_csv("scaled.csv")
    
    print("\n\n2. Reducing via PCA\n")
    data_pca, pca = pca_embeddings(data_scaled, n_components)
    
    # pd.DataFrame(data_pca).to_csv("pca.csv")

    print("\n\n3. HyperTuning the Parameter for KMeans\n")
    optimum_num_clusters = kmean_hyper_param_tuning(data_pca, parameter_list)
    print("optimum num of clusters =", optimum_num_clusters)

    # fitting KMeans
    kmeans = KMeans(n_clusters=optimum_num_clusters, init='k-means++')
    kmeans.fit(data_pca)
    centroids = kmeans.cluster_centers_
    
    print("\n\n4. Printing centroids data")
    get_results(centroids)
    
    # return centroids and pca datasets
    return centroids, data_pca

if __name__ == "__main__":
    main()