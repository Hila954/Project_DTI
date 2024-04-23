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
    original_vertices_of_cropped_img1 = vertices[(desired_indices_img1[0]-win_len):(desired_indices_img1[0]+win_len),
                                                  (desired_indices_img1[1]-win_len):(desired_indices_img1[1]+win_len),
                                                    (desired_indices_img1[2]-win_len):(desired_indices_img1[2]+win_len)]

    # the new vertices of the new cropped_img1
    vertices_cropped_img1 = np.arange(cropped_img1.shape[1]*cropped_img1.shape[2]*cropped_img1.shape[3]).reshape(cropped_img1[0].shape)

    final_connectivity_graph_1 = create_DTI_connectivity_graph(cropped_img1, lambda_value)

    shortest_path_subimage1 = dijkstra(final_connectivity_graph_1, indices=[3])
    del final_connectivity_graph_1, img1
    #~ PART TWO - find corresponding indexes by wraping the vertices matrix  (how each node moved)
    # the warp func is expecting a tensor [B, CH, H, W, D] img 
    tensor_vertices = torch.from_numpy(np.expand_dims(vertices, axis=(0, 1))).detach().type(torch.float32)
    wrapped_vertices = flow_warp(tensor_vertices, flow12.unsqueeze(0).cpu(), mode='nearest')[0][0]

    # we need to find the corresponding 'desired_indices_img1' in img2 
    desired_indices_img2_node = wrapped_vertices[desired_indices_img1[0], desired_indices_img1[1], desired_indices_img1[2]]
    desired_indices_img2 = from_graph_node_to_matrix_index(int(desired_indices_img2_node), vertices)[0]


    
    # get corresponding for the the original vertices in the cropped img1 
    matching_nodes_for_cropped = []
    for vertice in original_vertices_of_cropped_img1.flatten():
        matching_index = from_graph_node_to_matrix_index(vertice, vertices)[0]
        matching_node = wrapped_vertices[matching_index[0], matching_index[1], matching_index[2]]
        matching_nodes_for_cropped.append(matching_node.item())

    # order the vertices as it is used in the graph 
    ordered_matching_nodes_for_cropped = np.array(np.sort(matching_nodes_for_cropped), dtype='int')
    
    #creating mask because of memory issue
    image_mask = False*np.ones(img2[0].shape)
    for current_node in matching_nodes_for_cropped:
        matching_index = from_graph_node_to_matrix_index(current_node, vertices)[0]
        image_mask[matching_index[0], matching_index[1], matching_index[2]] = True
    
    #compute graph for img 2, use unique because there is not duplicated nodes in the graph
    matching_node_in_connectivity_graph_2 = np.where(np.unique(ordered_matching_nodes_for_cropped) == desired_indices_img2_node.item())[0]
    final_connectivity_graph_2 = create_DTI_connectivity_graph(img2, lambda_value, np.array(image_mask, dtype='bool'))
    shortest_path_subimage2 = dijkstra(final_connectivity_graph_2, indices=matching_node_in_connectivity_graph_2)
    
    # get the vertices that match the ordered_matching_nodes_for_cropped
    graph_2_vertices_full = []
    current_vertice = 0
    for index in range(len(ordered_matching_nodes_for_cropped)):
        if index == 0:
            graph_2_vertices_full.append(current_vertice)
            continue
        if ordered_matching_nodes_for_cropped[index] == ordered_matching_nodes_for_cropped[index - 1]:
            graph_2_vertices_full.append(current_vertice)
        else:
            current_vertice += 1
            graph_2_vertices_full.append(current_vertice)
    pass
    how_to_sort = np.argsort(matching_nodes_for_cropped) #need to sort image1 so it will fit graph2
    sorted_according_to_matching_dijikstra = shortest_path_subimage1[:, how_to_sort]
    matching_shortest_path = shortest_path_subimage2[:, graph_2_vertices_full] # duplicate values if needed 
        





    



def compute_dijkstra_validation(original_img, reconstructed_img, desired_indices, lambda_value, win_len=3):
    ''' check if the reconstruced image preserves the dijikstra distances '''


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


def create_DTI_connectivity_graph(image, lambda_value, optional_mask=None):
    '''create the weighted DTI adjacency matrix'''
    # to get the edges of the graph, regardless of the gradient (that may be 0)
    final_connectivity_graph = 0
    edges = make_edges_3d(*image[0].shape)
    #cover some of the edges if there is a mask 
    if optional_mask is not None:
        edges = mask_edges_weights(optional_mask, edges)

    #compute graph for the image, such that it statisfies dl^2 = dx^2+dy^2+dz^2+lamda*(dI1^2+....dI6^2) 
    for ch in range(image.shape[0]):
        connectivity_graph_ch = img_to_graph(image[ch], mask=optional_mask) # the weights are the gradients 
        final_connectivity_graph = lambda_value*np.add(final_connectivity_graph, np.square(connectivity_graph_ch.toarray()))
    
    # as we have 6 neighbors for each node and only dx/dy/dz changes, we add +1 to express the location change between nodes 
    final_connectivity_graph[edges[0], edges[1]] +=1
    final_connectivity_graph[edges[1], edges[0]] +=1

    # img_to_graph sets the diag as the node value, so we cancel it out 
    np.fill_diagonal(final_connectivity_graph, 0, wrap=False)

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

#Taken from sklearn python package 
def mask_edges_weights(mask, edges, weights=None):
    """Apply a mask to edges (weighted or not)"""
    inds = np.arange(mask.size)
    inds = inds[mask.ravel()]
    ind_mask = np.logical_and(np.in1d(edges[0], inds), np.in1d(edges[1], inds))
    edges = edges[:, ind_mask]
    if weights is not None:
        weights = weights[ind_mask]
    if len(edges.ravel()):
        maxval = edges.max()
    else:
        maxval = 0
    order = np.searchsorted(np.flatnonzero(mask), np.arange(maxval + 1))
    edges = order[edges]
    if weights is None:
        return edges
    else:
        return edges, weights






