import numpy as np
import sys
from timeit import default_timer as timer
np.set_printoptions(threshold=sys.maxsize, formatter={'float': lambda x: "{0:0.4f}".format(x)})
import tensorflow as tf

def construct_arrays_for_LBS(test_model):
    meshes_per_frame = np.zeros((test_model.num_of_vertices,test_model.num_of_frames,4))
    weights_per_bones = np.zeros((test_model.num_of_vertices,test_model.num_of_frames,6))
    #print("\n\n---------------\weights_per_bones: ", weights_per_bones[0].shape, "\n---------------\n\n")
    v_weights = test_model.v_weights
    #print("\n\n---------------\nv_weights: ", v_weights[0], "\n---------------\n\n")
    transformation_matrix = np.zeros((test_model.num_of_vertices,test_model.num_of_frames,6,4,4))
    v_weights_indices = test_model.v_weights_indices
    affine_transformations = test_model.affine_transformations
    #every frame has the rest pose mesh with coordinates xyz1
    for f in range(0, test_model.num_of_frames):
        for i in range(0, test_model.num_of_vertices):
            vi = np.zeros(4)
            vi[0:3] = test_model.mesh_v_in_rest_pose[i]
            vi[3] = 1
            meshes_per_frame[i][f] = vi
            for w in range(0, 6):
                #print("\ni: ", i, "| f: ", f,"| w: ", w)
                weights_per_bones[i][f][w] = v_weights[i][w]
                #Tb = np.zeros((4,4))
                transformation_matrix[i][f][w][0,:] = affine_transformations[ v_weights_indices[i][w]*12 + 0 : v_weights_indices[i][w]*12 + 4 , f]
                transformation_matrix[i][f][w][1,:] = affine_transformations[ v_weights_indices[i][w]*12 + 4 : v_weights_indices[i][w]*12 + 8 , f]
                transformation_matrix[i][f][w][2,:] = affine_transformations[ v_weights_indices[i][w]*12 + 8 : v_weights_indices[i][w]*12 + 12, f]
                transformation_matrix[i][f][w][3,:] = [0, 0, 0, 1]

    return meshes_per_frame, weights_per_bones, transformation_matrix

def compute_approx_v_with_LBS(test_model):
    start = timer()
    meshes_per_frame, weights_per_bones, transformation_matrix = construct_arrays_for_LBS(test_model)
    end = timer()
    time = end - start
    #print('meshes_per_frame = ' ,meshes_per_frame.shape)
    #print('weights_per_bones = ' ,weights_per_bones.shape)
    #print('transformation_matrix = ',transformation_matrix.shape)
    #WT_Array = weights_per_bones[:, :, :, None, None]*transformation_matrix
    meshes_per_frame = meshes_per_frame[:, :, :, None]
    weights_per_bones = weights_per_bones[:, :, :, None].transpose((0,1,3,2))
    transformation_matrix = np.reshape(transformation_matrix,(transformation_matrix.shape[0],transformation_matrix.shape[1],transformation_matrix.shape[2],-1))
    #print('meshes_per_frame = ' ,meshes_per_frame.shape)
    #print('weights_per_bones = ' ,weights_per_bones.shape)
    #print('transformation_matrix = ',transformation_matrix.shape)

    WT = np.matmul(weights_per_bones,transformation_matrix)
    #print('WT = ', WT.shape)
    WT = np.reshape(WT,(WT.shape[0],WT.shape[1],4,-1))
    #print('WT = ', WT.shape)

    VWT = np.matmul(WT,meshes_per_frame)
    #print('VWT = ', VWT.shape)
    VWT = np.squeeze(VWT)
    #print('VWT = ', VWT.shape)

    approximated_v = []
    for f in range(0, test_model.num_of_frames):
        v_prime = np.zeros( (test_model.num_of_vertices, 3) )
        for i in range(0, test_model.num_of_vertices):
            v_prime[i] = VWT[i][f][0:3]
        approximated_v.append(v_prime)

    '''
    approximated_v = []
    v_weights = test_model.v_weights
    v_weights_indices = test_model.v_weights_indices
    affine_transformations = test_model.affine_transformations
    for f in range(0, test_model.num_of_frames):
        v_prime = np.zeros( (test_model.num_of_vertices, 3) )
        for i in range(0, test_model.num_of_vertices):
            vi = np.zeros(4)
            vi[0:3] = test_model.mesh_v_in_rest_pose[i]
            vi[3] = 1
            vi_prime = np.zeros(4)
            for w in range(0, len(v_weights[i])):
                wib = v_weights[i][w]
                Tb = np.zeros((4,4))
                Tb[0,:] = affine_transformations[ v_weights_indices[i][w]*12 + 0 : v_weights_indices[i][w]*12 + 4 , f]
                Tb[1,:] = affine_transformations[ v_weights_indices[i][w]*12 + 4 : v_weights_indices[i][w]*12 + 8 , f]
                Tb[2,:] = affine_transformations[ v_weights_indices[i][w]*12 + 8 : v_weights_indices[i][w]*12 + 12, f]
                Tb[3,:] = [0, 0, 0, 1]
                vi_prime += wib*(np.matmul(Tb, vi))
            vi_prime = np.array(vi_prime[0:3])
            v_prime[i] = vi_prime
        approximated_v.append(v_prime)
    '''
    return approximated_v,time

def calculate_errors(test_model):
    A_orig = np.zeros( (3*test_model.num_of_vertices, test_model.num_of_frames) )
    A_appr = np.zeros( (3*test_model.num_of_vertices, test_model.num_of_frames) )
    A_avg  = np.zeros( (3*test_model.num_of_vertices, test_model.num_of_frames) )
    D_AorigAappr  = np.zeros( (test_model.num_of_vertices, test_model.num_of_frames) )

    num_of_v = test_model.num_of_vertices
    for f in range(0, test_model.num_of_frames):
        A_orig[:, f] = np.array([xyz for v in test_model.mesh_v_in_all_frames[f*num_of_v : f*num_of_v + num_of_v] for xyz in v])
        A_appr[:, f] = np.array([xyz for v in test_model.approximated_v[f] for xyz in v])
    avg_of_all_poses = np.mean(A_orig, axis=1)

    for f in range(0, test_model.num_of_frames):
        A_avg[:, f]  = avg_of_all_poses

    # AVERAGE MAX DISTANCE ERROR METRIC
    for f in range(0, test_model.num_of_frames):
        i = 0
        for v in range(0, num_of_v*3, 3):
            xyz_orig = np.array([ A_orig[v+0, f], A_orig[v+1, f], A_orig[v+2, f] ])
            xyz_approx = np.array([ A_appr[v+0, f], A_appr[v+1, f], A_appr[v+2, f] ])
            D_AorigAappr[i, f] = np.linalg.norm(xyz_orig - xyz_approx)
            i+=1

    Max_Distance = np.amax(D_AorigAappr, axis=0)
    Min_Distance = np.amin(D_AorigAappr, axis=0)
    #print("\n------------\nMax_Distance: ", Max_Distance.shape, " - Min_Distance: ", Min_Distance.shape)
    #print("vertices: ", num_of_v, " vertices x 3: ", num_of_v*3, " frames: ", test_model.num_of_frames)
    Avg_Max_Distance_Error = np.average(Max_Distance)
    Avg_Min_Distance_Error = np.average(Min_Distance)
    print("\n------------\nAvg_Max_Distance_Error: ", Avg_Max_Distance_Error)
    print("Avg_Min_Distance_Error: ", Avg_Min_Distance_Error, "\n------------\n")

    N = test_model.num_of_vertices
    P = test_model.num_of_frames
    dE = 100*( np.linalg.norm(A_orig - A_appr) / np.linalg.norm(A_orig - A_avg) )
    E_RMS = 100*( np.linalg.norm(A_orig - A_appr) / np.sqrt( 3 * N * P ) )
    return dE, E_RMS, Avg_Max_Distance_Error
