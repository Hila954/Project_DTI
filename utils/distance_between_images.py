from scipy.sparse.csgraph import dijkstra
from sklearn.feature_extraction.image import img_to_graph
import numpy as np



def compute_distances_array(img1, img2, flow12, desired_indices_img1):
    ''' find the distance between images, kind of gromov hausdorff method.
        1. choose indexes and find the shortest path in the image (using dijkstra) 
         2.  use the flow output to find the corresponding chosen indexes and find the path in img2
          3. average the difference'''
    
    #Init
    final_connectivity_graph_1 = 0
    final_connectivity_graph_2 = 0
    lambda_value = 1
    #create vertices matrix + edges for easier access, first parameter in shape in channel
    vertices = np.arange(img1.shape[1]*img1.shape[2]*img1.shape[3]).reshape(img1[0].shape)
    #take subset of the image according to the desired indices
    subimage1 = img1[:, (desired_indices_img1[0]-3):(desired_indices_img1[0]+3), (desired_indices_img1[1]-3):(desired_indices_img1[1]+3), (desired_indices_img1[2]-3):(desired_indices_img1[2]+3)]
    edges = make_edges_3d(*subimage1[0].shape)
    
    #compute graph for subimage1 img 1 that statisfies dl^2 = dx^2+dy^2+dz^2+lamda*dI^2 
    for ch in range(img1.shape[0]):
        connectivity_graph_1 = img_to_graph(subimage1[ch])
        final_connectivity_graph_1 = lambda_value*np.add(final_connectivity_graph_1, np.square(connectivity_graph_1.toarray()))
    
    # as we have 6 neighbors for each node and only dx/dy/dz changes, we add +1 to express the location change between nodes 
    final_connectivity_graph_1[edges[0], edges[1]] +=1
    final_connectivity_graph_1[edges[1], edges[0]] +=1
    # img_to_graph sets the diag as the node value, so we cancel it out 
    final_connectivity_graph_1[edges[0], edges[0]] = 0
    shortest_path_subimage1 = dijkstra(final_connectivity_graph_1, indices=desired_indices_img1)


    #compute graph for img 2
    for ch in range(img2.shape[0]):
        connectivity_graph_2 = img_to_graph(img2[ch])
        final_connectivity_graph_2 = lambda_value*np.add(final_connectivity_graph_2, np.square(connectivity_graph_2), )
    final_connectivity_graph_2 = np.add(final_connectivity_graph_2, np.power(final_connectivity_graph_2, 0))



def from_index_to_graph_node(index, nodes_matrix):
    return nodes_matrix[index[0], index[1], index[2]]


def from_graph_node_to_matrix_index(node_number, nodes_matrix):
    return np.where(nodes_matrix == node_number)


#Taken from sklearn python package 
def make_edges_3d(n_x, n_y, n_z=1):
    """Returns a list of edges for a 3D image.

    Parameters
    ----------
    n_x : int
        The size of the grid in the x direction.
    n_y : int
        The size of the grid in the y direction.
    n_z : integer, default=1
        The size of the grid in the z direction, defaults to 1
    """
    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))
    edges_deep = np.vstack((vertices[:, :, :-1].ravel(), vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges






