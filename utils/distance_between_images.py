from scipy.sparse.csgraph import dijkstra
from sklearn.feature_extraction.image import img_to_graph




def compute_distances_array(img1, img2, flow12, desired_indices_img1):
    #compute graph for img 1
    final_connectivity_graph_1 = []
    final_connectivity_graph_2 = []
    for ch in range(img1.shape[2]):
        connectivity_graph_1 = img_to_graph(img1[ch])
        final_connectivity_graph_1 += connectivity_graph_1 ** 2 

    for ch in range(img2.shape[2]):
        connectivity_graph_2 = img_to_graph(img2[ch])
        final_connectivity_graph_2 += connectivity_graph_2 ** 2 
