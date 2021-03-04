import sys
import os
import bpy

import input_model
import weight_calculation
import fitting
import linear_blend_skinning
import approximated_animation_sequence

import imp

import random
import numpy as np
from timeit import default_timer as timer

imp.reload(input_model)
imp.reload(weight_calculation)
imp.reload(fitting)
imp.reload(linear_blend_skinning)
imp.reload(approximated_animation_sequence)

P_factor = 2.0
k = 17	# number of clusters for p-center (number of proxy-bones)
already_processed_imported_objects = []

def p_center_clustering(paths_for_Blender, test_model_name):
    if 'fbx' in paths_for_Blender.test_model_path:
        test_model_names_list = []
        bpy.ops.import_scene.fbx(filepath = paths_for_Blender.test_model_path)
        test_model_names_list = get_model_names()
        print('test model names list = ',  test_model_names_list)
        test_model = input_model.TestModel_fbx(test_model_names_list[0], test_model_names_list[1], test_model_names_list[2])
    else:
        test_model_names_list = import_obj_files(paths_for_Blender)
        print('test model names list = ',  test_model_names_list)
        test_model = input_model.TestModel_obj(test_model_names_list)

    test_model.apply_rotation()
    copy_mesh_from_model(test_model)
    test_model_copy_names_list = get_model_names()
    test_model.set_name_of_copied_mesh(test_model_copy_names_list[0], test_model_copy_names_list[1])

    # deselect
    deselect_all()

    test_model.set_rest_pose_v()
    test_model.set_mesh_v_in_all_frames()
    number_of_model_vertices = test_model.num_of_vertices


    # --------------- P-center ---------------- #
    heads = []

    B = []
    indices_for_vertices = []
    indices_for_heads = []
    indices1 = []

    B1 = []

    for i in range(0, number_of_model_vertices):
        B1.append(test_model.mesh_v_in_rest_pose[i])
        indices1.append(i)

    arbitrary_index_of_head1 = random.randint(0, len(B1))
    head1 = B1[arbitrary_index_of_head1].copy()

    heads.append(head1)
    indices_for_heads.append(arbitrary_index_of_head1)

    B.append(B1)
    indices_for_vertices.append(indices1)
    print("B: ", len(B), " B[0]: ", len(B[0]), "B[0][0]: ", len(B[0][0]))
    for l in range(1, k):
        j = random.randint(1, l) % l
        v_i = B[j][0].copy()

        index_of_max = 0
        the_max = np.linalg.norm(np.array(heads[j]) - np.array(B[j][0]))

        for i in range(1, len(B[j])):
            euclidian_distance = np.linalg.norm(np.array(heads[j]) - np.array(B[j][i]))
            if euclidian_distance > the_max:
                the_max = euclidian_distance
                index_of_max = i
                v_i = B[j][i].copy()

        head_ = v_i.copy()

        heads.append(head_)
        indices_for_heads.append(indices_for_vertices[j][index_of_max])

        B_ = []
        indices_ = []

        B_.append(v_i)

        indices_.append(indices_for_vertices[j][index_of_max])

        if len(B[j]) > 1:
            B[j].pop(index_of_max)
            indices_for_vertices[j].pop(index_of_max)

        B.append(B_)
        indices_for_vertices.append(indices_)

        for f in range(0, len(B)):
            i = 0
            Bf_size = len(B[f])

            while i < Bf_size:
                distance_from_head = np.linalg.norm(np.array(B[f][i]) - np.array(heads[f]))
                distance_from_vi = np.linalg.norm(np.array(B[f][i]) - np.array(v_i))

                if distance_from_head >= distance_from_vi:
                    if len(B[f]) > 1:
                        B[len(B) - 1].append(B[f][i])
                        indices_for_vertices[len(indices_for_vertices) - 1].append(indices_for_vertices[f][i])

                        B[f].pop(i)
                        indices_for_vertices[f].pop(i)
                        i -= 1
                        Bf_size -= 1
                i += 1

    # ------------- Weights ---------------- #
    print('\n')
    radius = []

    for j in range(0, len(B)):
        the_max = np.linalg.norm(np.array(heads[j]) - np.array(B[j][0]))
        for i in range(1, len(B[j])):
            euclidian_distance = np.linalg.norm(np.array(heads[j]) - np.array(B[j][i]))
            if euclidian_distance > the_max:
                the_max = euclidian_distance

        r = the_max * P_factor
        radius.append(r)

    number_of_max_probs = 6
    weights = []
    weights_indices = []
    for j in range(0, len(B)):
        for i in range(0, len(B[j])):
            vertex_weight = []
            sum_of_weights = 0
            for bone in range(0, len(B)):
                d_ib = np.linalg.norm(np.array(heads[bone]) - np.array(B[j][i]))
                w_ib = 1.0 - (d_ib / radius[bone])

                if w_ib < 0.0:
                    w_ib = 0.0

                #sum_of_weights += w_ib
                vertex_weight.append(w_ib)

            #vertex_weight = vertex_weight.sort(reverse=True)

            vertex_weight_new_indices = sorted(range(len(vertex_weight)), key = lambda sub: vertex_weight[sub])[-number_of_max_probs:]

            for a in range(0, len(vertex_weight_new_indices)):
                sum_of_weights += vertex_weight[vertex_weight_new_indices[a]]

            for m in range(0, len(B)):
                if (m in vertex_weight_new_indices)  :
                    if (sum_of_weights > 0) :
                        #print("vertex_weight_new_indices", vertex_weight_new_indices,' found ',m)
                        vertex_weight[m] = vertex_weight[m] / sum_of_weights
                    else:
                        vertex_weight[m] = 0
                else :
                    vertex_weight[m] = 0
            weights.append(vertex_weight)
            weights_indices.append(vertex_weight_new_indices)
            #print('weights = ', len(weights),' weights_indices = ',len(weights_indices))

    print('GENERAL weights = ', len(weights),' weights_indices = ',len(weights_indices))
    print('weights[0] = ', len(weights[0]),' weights_indices[0] = ',len(weights_indices[0]))
    v_weights = []
    v_weights_indices = []
    #Sprint('len(B) = ', len(B),' test_model.num_of_vertices = ',test_model.num_of_vertices)
    for j in range(0, test_model.num_of_vertices):

        #print('vertex = ',j)
        v_weights.append([])
        v_weights_indices.append([])
        for bone in range(0, k):
            if weights[j][bone] > 0.0 or ( weights[j][bone] == 0.0 and bone in weights_indices[j] ):
                #print('weight = ',weights[j][bone])
                v_weights[j].append(weights[j][bone])
        #print('v_weights[',j,'] = ', v_weights[j],' weights_indices[',j,'] = ', weights_indices[j])

    v_weights_indices = weights_indices

    print('v_weights = ',len(v_weights[0]),' v_weights_indices = ',len(v_weights_indices[0]))
    test_model.num_of_bones = k
    test_model.set_v_weights(v_weights, v_weights_indices)
    #test_model.set_v_weights(v_weights, v_weights_indices)
    #-----------------------------------------------------------------------------------------
    fitting_time = 0
    start = timer()
    print('----> Fitting starts. This will take some time...')
    transformations = fitting.fit_affine_transformations(test_model)
    test_model.set_affine_transformations(transformations)
    #
    approx_v, constr_time = linear_blend_skinning.compute_approx_v_with_LBS(test_model)
    test_model.set_approximated_v(approx_v)
    #test_model.set_approximated_v(linear_blend_skinning.compute_approx_v_with_LBS(test_model))

    DistortionPercentage = []
    RMS_error = []
    AVG_MAX_Distance = []
    initial_dE, initial_E_RMS, initial_Avg_Max_Distance_Error = linear_blend_skinning.calculate_errors(test_model)

    print('\tInitial Distortion Percentage, before W_f= ', initial_dE)
    print('\tInitial E_RMS, before W_f = ', initial_E_RMS)
    print('\tInitial Avg_Max_Distance, before W_f = ', initial_Avg_Max_Distance_Error)
    DistortionPercentage.append(initial_dE)
    RMS_error.append(initial_E_RMS)
    AVG_MAX_Distance.append(initial_Avg_Max_Distance_Error)
    #
    end = timer()
    fitting_time += end - start

    old_v_weights_indices = []
    old_w_t = []
    old_errors = []
    number_of_fitting_steps = 5
    for i in range(0, number_of_fitting_steps):
        print('\ni = ', i)
        # weight fitting and computation of errors
        start = timer()
        new_v_weights, new_v_weights_indices = fitting.fit_weights(test_model, v_weights_indices)
        old_w_t.append(new_v_weights)
        old_v_weights_indices.append(new_v_weights_indices)
        test_model.set_v_weights(new_v_weights, new_v_weights_indices)
        end = timer()
        print('-- Time for weight fitting : ' + str(end - start))
        fitting_time += end - start

        start = timer()
        approx_v,constr_time = linear_blend_skinning.compute_approx_v_with_LBS(test_model)
        test_model.set_approximated_v(approx_v)
        #test_model.set_approximated_v(linear_blend_skinning.compute_approx_v_with_LBS(test_model))
        end = timer()
        print('Time to compute approximated v: ' + str(end - start))
        fitting_time += end - start

        start = timer()
        new_dE, new_E_RMS, new_Avg_Max_Distance_Error = linear_blend_skinning.calculate_errors(test_model)
        old_errors.append(new_dE)
        end = timer()
        print('Time to compute errors: ' + str(end - start))
        print('\tDistortion Percentage after weight fitting = ', new_dE)
        print('\tE_RMS after weight fitting = ', new_E_RMS)
        print('\tAvg_Max_Distance_Error after weight fitting = ', new_Avg_Max_Distance_Error)
        DistortionPercentage.append(new_dE)
        RMS_error.append(new_E_RMS)
        AVG_MAX_Distance.append(new_Avg_Max_Distance_Error)
        fitting_time += end - start

        # transformation fitting and computation of errors
        start = timer()
        new_transformations = fitting.fit_affine_transformations(test_model)
        old_w_t.append(new_transformations)
        test_model.set_affine_transformations(new_transformations)
        end = timer()
        print('-- Time for transformation fitting : ' + str(end - start))
        fitting_time += end - start

        start = timer()
        approx_v,constr_time = linear_blend_skinning.compute_approx_v_with_LBS(test_model)
        test_model.set_approximated_v(approx_v)
        #test_model.set_approximated_v(linear_blend_skinning.compute_approx_v_with_LBS(test_model))
        end = timer()
        print('Time to compute approximated v: ' + str(end - start))
        fitting_time += end - start

        start = timer()
        new_dE, new_E_RMS, new_Avg_Max_Distance_Error = linear_blend_skinning.calculate_errors(test_model)
        old_errors.append(new_dE)
        end = timer()
        print('Time to compute errors: ' + str(end - start))
        print('\tDistortion Percentage after transformation fitting = ', new_dE)
        print('\tE_RMS after transformation fitting = ', new_E_RMS)
        print('\tAvg_Max_Distance_Error after transformation fitting = ', new_Avg_Max_Distance_Error)
        DistortionPercentage.append(new_dE)
        RMS_error.append(new_E_RMS)
        AVG_MAX_Distance.append(new_Avg_Max_Distance_Error)

        fitting_time += end - start

    print('\n----> Fitting and LBS finished at last. Time for fitting: ', fitting_time)
    index_of_min_error = old_errors.index(min(old_errors))

    if index_of_min_error % 2 == 1:
        test_model.set_affine_transformations(old_w_t[index_of_min_error])
        test_model.set_v_weights(old_w_t[index_of_min_error - 1], old_v_weights_indices[index_of_min_error // 2])
    else:
        test_model.set_v_weights(old_w_t[index_of_min_error], old_v_weights_indices[index_of_min_error // 2])
        test_model.set_affine_transformations(old_w_t[index_of_min_error - 1])

    # linear blend skinning
    approx_v,constr_time = linear_blend_skinning.compute_approx_v_with_LBS(test_model)
    test_model.set_approximated_v(approx_v)
    #test_model.set_approximated_v(linear_blend_skinning.compute_approx_v_with_LBS(test_model))

    # compute error produced by linear blend skinning
    dE, E_RMS, Avg_Max_Distance_Error = linear_blend_skinning.calculate_errors(test_model)
    print('\tFinal Distortion Percentage = ', dE)
    print('\tFinal E_RMS = ', E_RMS)
    print('\tFinal Avg_Max_Distance_Error = ', Avg_Max_Distance_Error)

    np.savetxt(os.path.dirname(__file__)+'\\'+'P-CENTER-'+k+test_model_name+'-Dis_Per.txt', np.asarray(DistortionPercentage), fmt="%.6f", delimiter='|')
    np.savetxt(os.path.dirname(__file__)+'\\'+'P-CENTER-'+k+test_model_name+'-RMS_E.txt', np.asarray(RMS_error), fmt="%.6f", delimiter='|')
    np.savetxt(os.path.dirname(__file__)+'\\'+'P-CENTER-'+k+test_model_name+'-Avg_Max_Dis.txt', np.asarray(AVG_MAX_Distance), fmt="%.6f", delimiter='|')

    # construct array with approximated poses and init variables for the animation
    start = timer()
    approximated_animation_sequence.construct_array_with_approx_poses(test_model)
    approximated_animation_sequence.init_approx_animation_variables()
    end = timer()

    for i in range(0, len(heads)):
        bpy.ops.mesh.primitive_uv_sphere_add(size=0.01, location=heads[i])
        sphere = bpy.context.active_object
        mat = bpy.data.materials.new(name='Cube Material')
        sphere.data.materials.append(mat)
        sphere.data.materials[0].diffuse_color = [1, 0, 0]


def copy_mesh_from_model(model):
    scn = bpy.context.scene
    source_obj = bpy.data.objects[model.name_of_obj_mesh]
    obj_copy = source_obj.copy()
    obj_copy.data = source_obj.data.copy()
    obj_copy.animation_data_clear()
    scn.objects.link(obj_copy)

def deselect_all():
    for obj in bpy.context.selected_objects:
        obj.select = False

def import_obj_files(paths_for_Blender):
    start = timer()
    num_of_obj_files = 0
    test_model_names_list = []
    for obj_file_path in paths_for_Blender.list_of_obj_files_in_folder:
        if '.obj' in obj_file_path:
            bpy.ops.import_scene.obj(filepath = paths_for_Blender.test_model_path + obj_file_path)
            name_of_obj_file = get_model_names()[0]
            test_model_names_list.append(name_of_obj_file)
            num_of_obj_files += 1
            bpy.data.objects[name_of_obj_file].hide = True
    end = timer()
    print('elapsed time to load files = ', end - start)
    return test_model_names_list

def get_model_names():
    name_of_obj_mesh, name_of_mesh, name_of_armature = get_model_useful_names()
    model_names_list = [name_of_obj_mesh, name_of_mesh, name_of_armature]
    return model_names_list

def get_model_useful_names():
    # get the useful names of the model for later use
    global already_processed_imported_objects
    model_mesh = __get_desired_object('MESH')
    name_of_obj_mesh = model_mesh.name
    name_of_mesh = model_mesh.data.name
    model_armature = __get_desired_object('ARMATURE')
    if model_armature != None:
        name_of_armature = model_armature.name
    else:
        name_of_armature = ''
    # list with already processed objects. We need it in order to get new (imported) mesh/armature with get_desired_object(...)
    already_processed_imported_objects.append(model_mesh)
    already_processed_imported_objects.append(model_armature)
    return name_of_obj_mesh, name_of_mesh, name_of_armature

def __get_desired_object(object_type):
    global already_processed_imported_objects
    for obj in bpy.data.objects:
        if obj.type == object_type:
            if obj not in already_processed_imported_objects:
                desired_object = obj
                return desired_object
