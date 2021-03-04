import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import nnls
import scipy
import tensorflow as tf

def fit_affine_transformations(test_model):
    A = __construct_A_matrix_for_T(test_model)
    transformations = np.zeros((12*test_model.num_of_bones, test_model.num_of_frames))
    num_of_test_v = test_model.num_of_vertices
    current_pose = np.zeros((test_model.num_of_vertices*3, test_model.num_of_frames))
    la = np.matmul(np.transpose(A),A)
    AT = np.transpose(A)
    for i in range(0, test_model.num_of_frames):
        current_pose[:, i] = np.array([xyz for v in test_model.mesh_v_in_all_frames[i*num_of_test_v : i*num_of_test_v + num_of_test_v] for xyz in v])
        # f = np.array([xyz for v in test_model.mesh_v_in_all_frames[i*num_of_test_v : i*num_of_test_v + num_of_test_v] for xyz in v])
        # x = scipy.sparse.linalg.cg(la, np.matmul(AT,f), x0=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None)
        # transformations[:,i] = x[0]

    transformations = np.linalg.lstsq(A, current_pose, rcond=None)[0]
    # transformations_tensor = tf.linalg.lstsq(tf.convert_to_tensor(A, dtype=tf.float32), tf.convert_to_tensor(current_pose, dtype=tf.float32), l2_regularizer=0.01, fast=True, name=None)
    # sess = tf.Session()
    # with sess.as_default():
    # transformations = tf.Session().run(transformations_tensor)
    # print("A: ", (np.matmul(np.transpose(A),A)).shape)
    # print("B: ", (np.matmul(np.transpose(A),current_pose)).shape)
    return transformations

# construct A matrix to solve A x = b in the least squares sense
def __construct_A_matrix_for_T(test_model):
    A = np.zeros((3*test_model.num_of_vertices, 12*test_model.num_of_bones))
    for v in range(0, test_model.num_of_vertices):
        for index in range(0, len(test_model.v_weights_indices[v])):
            m_vb = [test_model.v_weights[v][index] * test_model.mesh_v_in_rest_pose[v][0],
                    test_model.v_weights[v][index] * test_model.mesh_v_in_rest_pose[v][1],
                    test_model.v_weights[v][index] * test_model.mesh_v_in_rest_pose[v][2],
                    test_model.v_weights[v][index]]
            m_vb = block_diag(m_vb, m_vb, m_vb)
            A[v*3 : v*3 + 3 , test_model.v_weights_indices[v][index]*12 : test_model.v_weights_indices[v][index]*12 + 12] = m_vb
    return A

'''def fit_weights(test_model, initial_v_weights_indices):
    v_weights = []
    v_weights_indices = []
    print('v_weights_for_cur_v before \n',test_model.v_weights[0])
    print('v_weights_indices_for_cur_v before  \n',test_model.v_weights_indices[0])
    for v in range(0, test_model.num_of_vertices):
        v_weights_for_cur_v = []
        v_weights_indices_for_cur_v = []
        A = __construct_A_matrix_for_w(test_model, v, initial_v_weights_indices)
        #print("A shape: ", A.shape)
        cur_v_in_all_poses = np.zeros( (3*test_model.num_of_frames) )
        for f in range(0, test_model.num_of_frames):
            cur_v_in_all_poses[3*f : 3*f + 3] = [xyz for xyz in test_model.mesh_v_in_all_frames[f*test_model.num_of_vertices + v]]

        #x = np.linalg.lstsq(A, cur_v_in_all_poses, rcond=None)[0]
        x = nnls(A, cur_v_in_all_poses, maxiter=None)[0]
        #print("\nfit_weights: ", x)
        sum = 0
        for i in range(0, len(x)):
            if x[i] >= 0 :
                sum += x[i]
                v_weights_for_cur_v.append(x[i])
                v_weights_indices_for_cur_v.append(initial_v_weights_indices[v][i])

        v_weights.append(v_weights_for_cur_v)
        v_weights_indices.append(v_weights_indices_for_cur_v)
    print('sum of weights =  \n',sum)
    print('v_weights_for_cur_v \n',v_weights[0])
    print('v_weights_indices_for_cur_v \n',v_weights_indices[0])
    return v_weights, v_weights_indices'''

def fit_weights(test_model,initial_v_weights_indices):
    v_weights = []
    v_weights_indices = []
    print('v_weights_for_cur_v before \n',test_model.v_weights[0])
    print('v_weights_indices_for_cur_v before  \n',test_model.v_weights_indices[0])
    for v in range(0, test_model.num_of_vertices):
        v_weights_for_cur_v = []
        v_weights_indices_for_cur_v = []
        A = __construct_A_matrix_for_w(test_model, v,initial_v_weights_indices)

        cur_v_in_all_poses = np.zeros( (3*(test_model.num_of_frames+1)) )
        for f in range(0, test_model.num_of_frames):
            cur_v_in_all_poses[3*f : 3*f + 3] = [xyz for xyz in test_model.mesh_v_in_all_frames[f*test_model.num_of_vertices + v]]
            #print("cur_v_in_all_poses[3*f : 3*f + 3]: ", cur_v_in_all_poses[3*f : 3*f + 3].shape)

        last_raw = np.ones_like(cur_v_in_all_poses[3*(test_model.num_of_frames-1) : 3*(test_model.num_of_frames-1) + 3])
        #print("last_raw: ", last_raw.shape)
        cur_v_in_all_poses[3*(test_model.num_of_frames) : 3*(test_model.num_of_frames) + 3] = last_raw
        #print("cur_v_in_all_poses: ", cur_v_in_all_poses.shape)
        #print("A: ", A.shape)

        #x = np.linalg.lstsq(A, cur_v_in_all_poses, rcond=None)[0]
        x = nnls(A, cur_v_in_all_poses, maxiter=None)[0]
        #x = scipy.sparse.linalg.cg( np.matmul(np.transpose(A),A), np.matmul(np.transpose(A),cur_v_in_all_poses), x0=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None)
        #print("x: ", len(x[0]))
        #x  = scipy.sparse.linalg.qmr(A, cur_v_in_all_poses, x0=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None)

        sum = 0
        for i in range(0, len(x)):
            if x[i] >= 0 :
                sum += x[i]
                v_weights_for_cur_v.append(x[i])
                v_weights_indices_for_cur_v.append(initial_v_weights_indices[v][i])

        v_weights.append(v_weights_for_cur_v)
        v_weights_indices.append(v_weights_indices_for_cur_v)
    print('sum of weights =  \n',sum)
    print('v_weights_for_cur_v \n',v_weights[0])
    print('v_weights_indices_for_cur_v \n',v_weights_indices[0])
    return v_weights, v_weights_indices

# construct A matrix to solve A x = b in the least squares sense
def __construct_A_matrix_for_w(test_model, v, initial_v_weights_indices):
    A = np.zeros( (3*(test_model.num_of_frames+1), 6) )
    i = -1
    for f in range(0, test_model.num_of_frames+1):
        if f <= test_model.num_of_frames-1:
            i = -1
            for bone in test_model.v_weights_indices[v]:
                if bone in initial_v_weights_indices[v]:
                    i += 1
                    A[3*f + 0][i] = test_model.affine_transformations[ 12*bone + 0 , f] * test_model.mesh_v_in_rest_pose[v][0] + \
                                       test_model.affine_transformations[ 12*bone + 1 , f] * test_model.mesh_v_in_rest_pose[v][1] + \
                                       test_model.affine_transformations[ 12*bone + 2 , f] * test_model.mesh_v_in_rest_pose[v][2] + \
                                       test_model.affine_transformations[ 12*bone + 3 , f]
                    A[3*f + 1][i] = test_model.affine_transformations[ 12*bone + 4 , f] * test_model.mesh_v_in_rest_pose[v][0] + \
                                       test_model.affine_transformations[ 12*bone + 5 , f] * test_model.mesh_v_in_rest_pose[v][1] + \
                                       test_model.affine_transformations[ 12*bone + 6 , f] * test_model.mesh_v_in_rest_pose[v][2] + \
                                       test_model.affine_transformations[ 12*bone + 7 , f]
                    A[3*f + 2][i] = test_model.affine_transformations[ 12*bone + 8 , f] * test_model.mesh_v_in_rest_pose[v][0] + \
                                       test_model.affine_transformations[ 12*bone + 9 , f] * test_model.mesh_v_in_rest_pose[v][1] + \
                                       test_model.affine_transformations[ 12*bone + 10 , f] * test_model.mesh_v_in_rest_pose[v][2] + \
                                       test_model.affine_transformations[ 12*bone + 11 , f]
        else :
            for k in range(0, 6):
                A[3*f + 0][k] = 1
                A[3*f + 1][k] = 1
                A[3*f + 2][k] = 1

        #print("A[3*f + 0]: ", A[3*f + 0])
    return A
