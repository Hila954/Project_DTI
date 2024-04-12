from scipy.sparse.csgraph import dijkstra
from sklearn.feature_extraction.image import img_to_graph
import numpy as np
import torch

from utils.flow_utils import flow_warp



def compute_distances_array(img1, img2, flow12, desired_indices_img1):
    ''' find the distance between images, kind of gromov hausdorff method.
        1. choose indexes and find the shortest path in the image (using dijkstra) 
         2.  use the flow output to find the corresponding chosen indexes and find the path in img2
          3. average the difference'''
    
    #Init
    lambda_value = 1
    win_len = 3
    #~ PART ONE - dijikstra for img1

    #create vertices matrix for easier access, first parameter in shape is channel
    vertices = np.arange(img1.shape[1]*img1.shape[2]*img1.shape[3]).reshape(img1[0].shape)

    #take subset of the data according to the desired indices
    cropped_img1 = img1[:, (desired_indices_img1[0]-win_len):(desired_indices_img1[0]+win_len),
                         (desired_indices_img1[1]-win_len):(desired_indices_img1[1]+win_len),
                           (desired_indices_img1[2]-win_len):(desired_indices_img1[2]+win_len)]
    
    # crop the original vertices to fit the cropped img1
    original_vertices_of_cropped_img1 = vertices[(desired_indices_img1[0]-3):(desired_indices_img1[0]+3), (desired_indices_img1[1]-3):(desired_indices_img1[1]+3), (desired_indices_img1[2]-3):(desired_indices_img1[2]+3)]

    # the new vertices of the new cropped_img1
    vertices_cropped_img1 = np.arange(cropped_img1.shape[1]*cropped_img1.shape[2]*cropped_img1.shape[3]).reshape(cropped_img1[0].shape)

    final_connectivity_graph_1 = create_DTI_connectivity_graph(cropped_img1, lambda_value)

    shortest_path_subimage1 = dijkstra(final_connectivity_graph_1, indices=[3])
    
    #~ PART TWO - find corresponding indexes by wraping the vertices matrix  (how each node moved)
    # the warp func is expecting a tensor [B, CH, H, W, D] img 
    tensor_vertices = torch.from_numpy(np.expand_dims(vertices, axis=(0, 1))).detach().type(torch.float32)
    wrapped_vertices = flow_warp(tensor_vertices, flow12.unsqueeze(0).cpu(), mode='nearest')[0]

    # we need to find the corresponding 'desired_indices_img1' in img2 
    desired_indices_img2_node = wrapped_vertices[0, desired_indices_img1[0], desired_indices_img1[1], desired_indices_img1[2]]
    desired_indices_img2 = from_graph_node_to_matrix_index(int(desired_indices_img2_node), vertices)

    # open 2*win around the corresponding node to get the matching cropped image 2 
    cropped_img2 = img2[:, (desired_indices_img2[0][0]-2*win_len):(desired_indices_img2[0][0]+2*win_len),
                        (desired_indices_img2[0][1]-2*win_len):(desired_indices_img2[0][1]+2*win_len),
                        (desired_indices_img2[0][2]-2*win_len):(desired_indices_img2[0][2]+2*win_len)]
    

    #compute graph for img 2
    final_connectivity_graph_2 = create_DTI_connectivity_graph(cropped_img2, lambda_value)
    shortest_path_subimage2 = dijkstra(final_connectivity_graph_2, indices=[3])


    



def compute_dijkstra_validation(original_img, reconstructed_img, desired_indices, win_len=3):
    ''' check if the reconstruced image preserves the dijikstra distances '''
    lambda_value = 1


    #take subset of the data according to the desired indices
    cropped_original_img = original_img[:, (desired_indices[0]-win_len):(desired_indices[0]+win_len),
                        (desired_indices[1]-win_len):(desired_indices[1]+win_len),
                        (desired_indices[2]-win_len):(desired_indices[2]+win_len)]
    cropped_reconstructed_img = reconstructed_img[:, (desired_indices[0]-win_len):(desired_indices[0]+win_len),
                (desired_indices[1]-win_len):(desired_indices[1]+win_len),
                (desired_indices[2]-win_len):(desired_indices[2]+win_len)]

    final_connectivity_graph = create_DTI_connectivity_graph(cropped_original_img, lambda_value)
    reconstructed_final_connectivity_graph = create_DTI_connectivity_graph(cropped_reconstructed_img, lambda_value)


    shortest_path_subimage = dijkstra(final_connectivity_graph)
    shortest_path_subimage_reconstructed = dijkstra(reconstructed_final_connectivity_graph)



    return np.mean((shortest_path_subimage - shortest_path_subimage_reconstructed) ** 2) 


def create_DTI_connectivity_graph(image, lambda_value):
    '''create the weighted DTI adjacency matrix'''
    # to get the edges of the graph, regardless of the gradient (that may be 0)
    final_connectivity_graph = 0
    edges = make_edges_3d(*image[0].shape)

    #compute graph for the image, such that it statisfies dl^2 = dx^2+dy^2+dz^2+lamda*(dI1^2+....dI6^2) 
    for ch in range(image.shape[0]):
        connectivity_graph_ch = img_to_graph(image[ch]) # the weights are the gradients 
        final_connectivity_graph = lambda_value*np.add(final_connectivity_graph, np.square(connectivity_graph_ch.toarray()))
    
    # as we have 6 neighbors for each node and only dx/dy/dz changes, we add +1 to express the location change between nodes 
    final_connectivity_graph[edges[0], edges[1]] +=1
    final_connectivity_graph[edges[1], edges[0]] +=1

    # img_to_graph sets the diag as the node value, so we cancel it out 
    final_connectivity_graph[edges[0], edges[0]] = 0
    return final_connectivity_graph

def from_index_to_graph_node(index, nodes_matrix):
    return nodes_matrix[index[0], index[1], index[2]]


def from_graph_node_to_matrix_index(node_number, nodes_matrix):
    return np.argwhere(nodes_matrix == node_number)


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






