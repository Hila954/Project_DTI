from scipy.sparse.csgraph import dijkstra
from sklearn.feature_extraction.image import img_to_graph
import numpy as np
import torch
from cupyx.scipy.sparse import csr_matrix as csr_gpu
from utils.flow_utils import flow_warp
import time
import random



def compute_distances_array(img1, img2, flow12, points_img1, vox_dim1, vox_dim2, lambda_val = 1):
    ''' find the distance between images, kind of gromov hausdorff method.
        1. choose indexes and find the shortest path in the image (using dijkstra) 
         2.  use the flow output to find the corresponding chosen indexes and find the path in img2
          3. average the difference'''
    
    #Init
    start_time = time.time()
    lambda_value = lambda_val
    win_len = 5
    sum_distance = 0
    #! we create connectivity graph for the whole img2 regardless of desired indices because we can't 
    #! know where the corresponding points from img1 will appear, so do it once and not in loop
    final_connectivity_graph_2 = create_DTI_connectivity_graph(img2, lambda_value, vox_dim=vox_dim2)
    
    #~ PART ONE - dijikstra for img1

    #create vertices matrix for easier access, first parameter in shape is channel
    vertices = np.arange(img1.shape[1]*img1.shape[2]*img1.shape[3]).reshape(img1[0].shape)
    for point in points_img1:
        #take subset of the data according to the desired indices
        cropped_img1 = img1[:, (point[0]-win_len//2):(point[0]+win_len//2+1),
                            (point[1]-win_len//2):(point[1]+win_len//2+1),
                            (point[2]-win_len//2):(point[2]+win_len//2+1)]
        
        # crop the original vertices to fit the cropped img1
        original_vertices_of_cropped_img1 = vertices[(point[0]-win_len//2):(point[0]+win_len//2 + 1),
                                                    (point[1]-win_len//2):(point[1]+win_len//2 + 1),
                                                        (point[2]-win_len//2):(point[2]+win_len//2 + 1)]

        nx, ny, nz = cropped_img1.shape[1], cropped_img1.shape[2], cropped_img1.shape[3] # first one is channel 
        cropped_img1_vertices = np.arange(nx * ny * nz).reshape((nx, ny, nz))
        
        final_connectivity_graph_1 = create_DTI_connectivity_graph(cropped_img1, lambda_value, vox_dim=vox_dim1)

        # compute distances according to the desired index, middle of cube in cropped img1
        shortest_path_subimage1 = dijkstra(final_connectivity_graph_1,
                                            indices=[from_index_to_graph_node((nx//2, ny//2, nz//2), cropped_img1_vertices)])
        del final_connectivity_graph_1
        #~ PART TWO - find corresponding indexes by wraping the vertices matrix  (how each node moved)
        # the warp func is expecting a tensor [B, CH, H, W, D] img 
        tensor_vertices = torch.from_numpy(np.expand_dims(vertices, axis=(0, 1))).detach().type(torch.float32)
        wrapped_vertices = flow_warp(tensor_vertices, flow12.unsqueeze(0).cpu(), mode='nearest')[0][0]

        # we need to find the corresponding 'desired_indices_img1' in img2 
        desired_indices_img2_node = wrapped_vertices[point[0], point[1], point[2]]


        
        # get corresponding for the the original vertices in the cropped img1 
        matching_nodes_for_cropped = []
        for vertice in original_vertices_of_cropped_img1.flatten():
            matching_index = from_graph_node_to_matrix_index(vertice, vertices)[0] # get the (i,j,k) index
            matching_node = wrapped_vertices[matching_index[0], matching_index[1], matching_index[2]] # get the corresponding node
            matching_nodes_for_cropped.append(int(matching_node.item()))

        #~ PART THREE - get graph for the whole img2 and calculate the dijikstra 

        shortest_path_subimage2 = dijkstra(final_connectivity_graph_2, indices=int(desired_indices_img2_node.item()))
        
        matching_shortest_path = shortest_path_subimage2[matching_nodes_for_cropped] # duplicate values if needed 
        sum_distance += np.mean((shortest_path_subimage1 - matching_shortest_path) ** 2) 
    print("--- %s seconds ---" % (time.time() - start_time))

    return sum_distance/len(points_img1)
        





    



def compute_dijkstra_validation(original_img, reconstructed_img, chosen_points, lambda_value, win_len=5):
    ''' check if the reconstruced image preserves the dijikstra distances '''
    
    sum_of_mean = 0
    for desired_indices in chosen_points:
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
        sum_of_mean += np.mean((shortest_path_subimage - shortest_path_subimage_reconstructed) ** 2)


    return sum_of_mean/len(chosen_points)


def create_DTI_connectivity_graph(image, lambda_value, optional_mask=None, vox_dim=(1, 1, 1)):
    '''create the weighted DTI adjacency matrix'''
    
    Intensity_connectivity_graph = 0
    edges = make_edges_3d(*image[0].shape) # to get the edges of the graph, regardless of the gradient (that may be 0)
    #cover some of the edges if there is a mask 
    if optional_mask is not None:
        edges = mask_edges_weights(optional_mask, edges)

    #compute graph for the image, such that it statisfies dl^2 = dx^2+dy^2+dz^2+lamda*(dI1^2+....dI6^2) 
    for ch in range(image.shape[0]):
        connectivity_graph_ch = img_to_graph(image[ch], mask=optional_mask) # the weights are the gradients 
        connectivity_graph_ch.data = connectivity_graph_ch.data ** 2
        Intensity_connectivity_graph = lambda_value*np.add(Intensity_connectivity_graph, connectivity_graph_ch)
    
    #! to consider the vox dim of the images we want that each step in some direction will preset dx/dy/dz accordingly 
    image_of_dxdydz = mesh_grid_scale_fix(image.shape[1], image.shape[2], image.shape[3], vox_dim)
    connectivity_graph_for_dxdydz = img_to_graph(image_of_dxdydz) #gradient so now each cell has the corresponding dx dy dz 

    final_connectivity_graph = connectivity_graph_for_dxdydz + Intensity_connectivity_graph
    #~ some backup for now ~~~~~~~~~~
    # # as we have 6 neighbors for each node and only dx/dy/dz changes, we add +1 to express the location change between nodes 
    # final_connectivity_graph_gpu = csr_gpu(final_connectivity_graph)
    
    # final_connectivity_graph_gpu[edges[0], edges[1]] +=1
    # final_connectivity_graph_gpu[edges[1], edges[0]] +=1

    
    # # img_to_graph sets the diag as the node value, so we cancel it out + change to cpu
    # final_connectivity_graph = final_connectivity_graph_gpu.get()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    final_connectivity_graph.setdiag(0)
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
    

#~ taken from flow_utils with slight modifications 
def mesh_grid_scale_fix(H, W, D, image_dxdydz):
    x = np.arange(H) * (image_dxdydz[0]) ** 2
    y = np.arange(W) * (image_dxdydz[1]) ** 2
    z = np.arange(D) * (image_dxdydz[2]) ** 2
    mesh = np.stack(np.meshgrid(x, y, z), 0)
    return np.sum(mesh, axis=0)


def pick_points_in_DTI(img, needed_points_amount):
    random.seed(10)
    points = []
    img1_x_range, img1_y_range, img1_z_range = np.where(np.sum(img, axis = 0) > np.sum(img, axis = 0)[0,0,0])
    for picked_point in range(needed_points_amount):
        picked_point = random.randint(0, len(img1_x_range))
        points.append((img1_x_range[picked_point], img1_y_range[picked_point], img1_z_range[picked_point]))
    return points






