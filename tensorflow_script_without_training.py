import bpy
import json
import tensorflow as tf
from tensorflow import keras
import sys
import numpy as np
from timeit import default_timer as timer

#from main import num_of_proxy_bones
import input_model
import weight_calculation
import fitting
import linear_blend_skinning
import approximated_animation_sequence
import p_center
import imp
import functools
import random
import math
import os

# PRINTING OPTIONS
np.set_printoptions(threshold=sys.maxsize, formatter={'float': lambda x: "{0:0.8f}".format(x)})

imp.reload(input_model)
imp.reload(weight_calculation)
imp.reload(fitting)
imp.reload(linear_blend_skinning)
imp.reload(approximated_animation_sequence)
imp.reload(p_center)

already_processed_imported_objects = []
number_of_proxy_bones = 120

cluster_color_data = [[1,0,0],[1,0.078,0.576],[1,1,0],[0,1,0],[0,0,1],
[0,1,1],[1,0,1],[0.502,0.502,0],[0.627,0.322,0.176],[0.871,0.722,0.529],
[0.294,0,0.510],[0,0.980,0.604],[1,0.549,0],[0.502,0.502,0.502],[0.941,1,1],
[1,1,0.9417],[0.737,0.561,0.561],[0,0,0.545],[0.118,0.565,1], [0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],[0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],[0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],[0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],[0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],[0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],[0.27543,0.50115,0.96594]
]

turbo_colormap_data = [[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],[0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],[0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],[0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],[0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],[0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],[0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],[0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],[0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],[0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],[0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],[0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],[0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],[0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],[0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],[0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],[0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],[0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],[0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],[0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],[0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],[0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],[0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],[0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],[0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],[0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],[0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],[0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],[0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],[0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],[0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],[0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],[0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],[0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],[0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],[0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],[0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],[0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],[0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],[0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],[0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],[0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],[0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],[0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],[0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],[0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],[0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],[0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],[0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],[0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],[0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],[0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],[0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],[0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],[0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],[0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],[0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],[0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],[0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],[0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],[0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],[0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],[0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],[0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],[0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],[0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],[0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],[0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],[0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],[0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],[0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],[0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],[0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],[0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],[0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],[0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],[0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],[0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],[0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],[0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],[0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],[0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],[0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],[0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],[0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],[0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],[0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],[0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],[0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],[0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],[0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],[0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],[0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],[0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],[0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],[0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],[0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],[0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],[0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],[0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],[0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],[0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],[0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],[0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],[0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],[0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],[0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],[0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],[0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],[0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],[0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],[0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],[0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],[0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]]

def interpolate(colormap, x):
  x = max(0.0, min(1.0, x))
  a = int(x*255.0)
  b = min(255, a + 1)
  f = x*255.0 - a
  return [colormap[a][0] + (colormap[b][0] - colormap[a][0]) * f,
          colormap[a][1] + (colormap[b][1] - colormap[a][1]) * f,
          colormap[a][2] + (colormap[b][2] - colormap[a][2]) * f]

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    #if(float(value - leftMin) !=0):
    valueScaled = float(value - leftMin) / float(leftSpan)
    #else:
    #    valueScaled = 0

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def clusterColors(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        r = translate(r, 0, 255, 0, 1)
        g = translate(g, 0, 255, 0, 1)
        b = translate(b, 0, 255, 0, 1)
        ret.append([r,g,b])
    return ret

#---------------------------------- Face Normal Calculation --------------------------------------
# vector substration
def vecsub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
# vector crossproduct
def veccross(x, y):
    v = [0, 0, 0]
    v[0] = x[1]*y[2] - x[2]*y[1]
    v[1] = x[2]*y[0] - x[0]*y[2]
    v[2] = x[0]*y[1] - x[1]*y[0]
    return v
# calculate normal from 3 verts
def Normal(v0, v1, v2):
    return veccross(vecsub(v0, v1),vecsub(v0, v2))
# calculate normal from 4 verts found in the blender sources in arithb.c
# (bl-blender\blender\source\blender\blenlib\intern\arithb.c)
def Normal4(v0, v1, v2, v3):
    return veccross(vecsub(v0, v2),vecsub(v1, v3))
#---------------------------------- Face Normal Calculation --------------------------------------

def start_tensorflow_without_training(paths_for_Blender, networkPrefix, test_model_name):

    validation_models = []
    for i in range(0, len(paths_for_Blender.train_model_paths)):
        if (i >= len(paths_for_Blender.train_model_paths) - 2):
            print("\n--------------------- VALIDATION MODEL: ", i, "---------------------\n")
            bpy.ops.import_scene.fbx(filepath = paths_for_Blender.train_model_paths[i])
            train_model_names_list = get_model_names()
            train_model = input_model.TrainModel(train_model_names_list[0], train_model_names_list[1], train_model_names_list[2])
            train_model.change_mesh_to_World_coordinates()
            train_model.number_of_proxy_bones = number_of_proxy_bones
            train_model.set_v_groups_names()
            train_model.set_v_groups()
            train_model.set_mesh_v_in_all_frames()
            for action in bpy.data.actions:
                bpy.data.actions.remove(action, do_unlink=True, do_id_user=True, do_ui_user=True)
            train_model.set_train_labels()
            validation_models.append(train_model)

    if paths_for_Blender.file_type == 'fbx':
        bpy.ops.import_scene.fbx(filepath = paths_for_Blender.test_model_path)
        test_model_names_list = get_model_names()
        test_model = input_model.TestModel_fbx(test_model_names_list[0], test_model_names_list[1], test_model_names_list[2])
    else:
        test_model_names_list = import_obj_files(paths_for_Blender)
        test_model = input_model.TestModel_obj(test_model_names_list)
        #test_model = input_model.TestModel_obj(test_model_names_list)
    print('test model names list = ',  test_model_names_list)

    #TIME LIST
    time_for_each_step = []
    time_for_each_step.append('\nTime required :')

    #NETWORK LOADING
    model = keras.models.load_model(os.path.dirname(__file__)+'\\'+networkPrefix+'.h5')
    model.summary()
    batch_size = 4096

    test_model.apply_rotation()
    copy_mesh_from_model(test_model)
    test_model_copy_names_list = get_model_names()
    test_model.set_name_of_copied_mesh(test_model_copy_names_list[0], test_model_copy_names_list[1])

    # deselect
    deselect_all()

    start = timer()
    # set the rest pose of the test model (the pose in the first frame of the animation)
    test_model.set_rest_pose_v()

    # get test set vertices from all frames and set them to the object (test_model)
    test_model.set_mesh_v_in_all_frames()

    # obj files only
    if paths_for_Blender.file_type != 'fbx':
        starting_sequence = test_model.get_mesh_v_in_all_frames_Array()
        print('test model: ', starting_sequence.shape)
        starting_sequence3D = test_model.get_mesh_v_in_all_frames_Array3D()
        print('test model 3D: ', starting_sequence3D.shape)
        np.savetxt(os.path.dirname(__file__)+'\\outcome\\'+networkPrefix+'InputSequence.txt', starting_sequence, fmt="%.6f", delimiter=' ')

    histories_evaluation = []
    mesh_and_action_name = []
    for validation_model in validation_models:
        for i in range(0, len(validation_model.mesh_v_in_takes)):
            print("EVALUATION validation_model: ", validation_model.mesh_v_in_takes[i].shape)
            current_val_history = model.evaluate(validation_model.mesh_v_in_takes[i], validation_model.labels_for_rest_pose, batch_size=batch_size)
            print('\n')
            print("\n\n------- EVALUATE -------: ", current_val_history)
            histories_evaluation.append(current_val_history)
            mesh_and_action_name.append([validation_model.name_of_mesh, "Action"+str(i)])

    print("histories_evaluation", histories_evaluation)
    eval_losses = np.zeros( (len(histories_evaluation)))
    eval_accuracies = np.zeros( (len(histories_evaluation)))
    print("eval_losses shape: ", eval_losses.shape)
    print("eval_accuracies shape: ", eval_accuracies.shape)
    print("mesh_and_action_name", mesh_and_action_name)

    for i in range(0, len(histories_evaluation)):
        for j in range(0, len(histories_evaluation[i])):
            if (j == 0): #loss
                eval_losses[i] = histories_evaluation[i][j]
            else: #accuracy
                eval_accuracies[i] += histories_evaluation[i][j]
    print("EVALUATION LOSSES: ", eval_losses)
    print("EVALUATION ACCURACIES: ", eval_accuracies)

    #outputs of neural network
    predictions = model.predict(test_model.mesh_v_for_predictions, batch_size=batch_size)

    np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'predictions.out', predictions, fmt="%.6f")
    test_model_mesh_rest_pose = test_model.get_rest_pose_v()
    print("------------ TEST MODEL ------------ \n")
    print("SHAPE: ", test_model_mesh_rest_pose.shape)
    np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'test_model_rest_pose.out', test_model_mesh_rest_pose, fmt="%.6f")

    test_model_mesh_rest_pose_to_string = str(test_model_mesh_rest_pose);
    np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'predictions.csv', np.concatenate((test_model_mesh_rest_pose, predictions), axis=1), fmt="%.6f", delimiter=',')

    # Save the model
    '''model.save(os.path.dirname(__file__)+'\\'+networkPrefix+'.h5')
    # Recreate the exact same model purely from the file
    new_model = keras.models.load_model(os.path.dirname(__file__)+'\\'+networkPrefix+'.h5')
    new_model.summary()
    # Check that the state is preserved
    new_predictions = new_model.predict(test_model.mesh_v_for_predictions, batch_size=batch_size)
    np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'new-predictions.out', new_predictions, fmt="%.6f")

    # Note that the optimizer state is preserved as well:
    # you can resume training where you left off.
    '''
    #print("------------- PREDICTIONS --------------\n", predictions)
    # construct test set labels
    test_set_labels = np.copy(predictions)
    #test_set_labels[test_set_labels >= 0.5] = 1
    #test_set_labels[test_set_labels < 0.5] = 0
    #print("------------- TEST LABELS --------------\n", test_set_labels)

    number_of_max_probs = 6
    indices_test_set_labels = np.zeros((test_set_labels.shape[0], number_of_max_probs), dtype=np.int)
    for i in range(0, test_set_labels.shape[0]):
        indices_test_set_labels[i] = np.argsort(-test_set_labels[i])[:number_of_max_probs]
    #print("------------- INDICES MAX LABELS --------------\n", indices_test_set_labels)

    for i in range(0, test_set_labels.shape[0]):
        for j in range(0, test_set_labels.shape[1]):
            for a in range(0, number_of_max_probs):
                if(j == indices_test_set_labels[i][a]):
                    test_set_labels[i][j] = 1


    for i in range(0, test_set_labels.shape[0]):
        for j in range(0, test_set_labels.shape[1]):
            if(test_set_labels[i][j] != 1):
                test_set_labels[i][j] = 0


    print("------------- TEST LABELS --------------\n", test_set_labels[0])
    print("------------- INDICES MAX LABELS --------------\n", indices_test_set_labels[0])
    print("------------- PREDICTIONS --------------\n", predictions[0])
    #print("------------- NEW PREDICTIONS --------------\n", new_predictions[0])
    test_set_labels_original = test_set_labels
    # assign the number of clusters(bones) to the test model
    test_model.num_of_bones = number_of_proxy_bones

    # CALCULATING CLUSTERS FOR P-center
    #test_set_labels_ARRAY = np.concatenate([np.asarray(i) for i in test_set_labels])
    print("\n------------- TEST LABELS SHAPE--------------\n", len(test_set_labels), len(test_set_labels[0]))
    #print("\n------------- TEST LABELS ARRAY SHAPE--------------\n", test_set_labels_ARRAY.shape)
    clusters_for_pCenter = np.count_nonzero(test_set_labels, axis=0)
    print("\n------------- CLUSTERS FOR P-CENTER --------------\n", clusters_for_pCenter)
    print("\n------------- CLUSTERS FOR P-CENTER shape --------------\n", clusters_for_pCenter.shape)
    #find non zero indices for color obj
    colorIndices = np.nonzero(clusters_for_pCenter)
    print("\n------------- COLOR CLUSTER INDICES --------------\n", colorIndices)
    clusters_for_pCenter = np.count_nonzero(clusters_for_pCenter)
    print("\n------------- CLUSTERS FOR P-CENTER --------------\n", clusters_for_pCenter)
    clusters_for_pCenter_list = [clusters_for_pCenter]
    # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'-'+test_model_name+'CLUSTERS.txt', clusters_for_pCenter_list, fmt="%.2f")



    # the vertices that were predicted to belong to zero bones
    # are then assigned to the one with the highest probability
    test_set_labels = assign_unassigned_vertices_to_bone(test_set_labels, predictions)
    test_model.set_labels(test_set_labels)
    print("------------- PREDICTIONS (AFTER) --------------\n", predictions[0])
    print("------------- TEST LABELS (AFTER) --------------\n", test_set_labels[0])



    end = timer()

    time_for_each_step.append('\tto construct test labels : ' + str(end - start))

    # calculate initial vertex weights
    start = timer()
    v_weights, v_weights_indices = weight_calculation.calculate_weights_using_tensorflow_predictions(test_model, predictions)
    test_model.set_v_weights(v_weights, v_weights_indices)
    print("------------- PREDICTIONS - WEIGHTS (AFTER) --------------\n", v_weights[0])
    print("------------- PREDICTIONS - WEIGHTS TEST MODEL (AFTER) --------------\n", test_model.v_weights[0])
    print("------------- PREDICTIONS - WEIGHTS (AFTER) SUM:  --------------\n", np.sum(v_weights[0]))

    np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'weights.csv', v_weights, fmt="%.6f", delimiter='|')
    np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'weights-indices.csv', v_weights_indices, fmt="%.6f", delimiter='|')


    end = timer()
    time_for_each_step.append('\tto calculate weights : ' + str(end - start))

    # get transformations produced by affine transformation fitting
    fitting_time = 5
    start = timer()
    transformations = fitting.fit_affine_transformations(test_model)
    #transformations_tensor = fitting.fit_affine_transformations(test_model)

    '''sess = tf.compat.v1.Session()
    with sess.as_default():
        transformations = sess.run(tf.transpose(transformations_tensor))
    print("\n\nINITIAL transformations", transformations.shape)
    '''
    test_model.set_affine_transformations(transformations)
    #
    approx_v,constr_time = linear_blend_skinning.compute_approx_v_with_LBS(test_model)
    test_model.set_approximated_v(approx_v)
    print('Time to compute approximated v: ' + str(constr_time))

    DistortionPercentage = []
    RMS_error = []
    AVG_MAX_Distance = []
    AVG_NORMAL_ANGLE_DISTANCE = []

    # horsedata = bpy.data.objects['fox1'].data
    # #horsedata = bpy.data.objects['tree_forg'].data
    # #horsedata = bpy.data.meshes['horse-gallop-01']
    # horsedataFaces = np.zeros( (len(horsedata.polygons), 3) )
    # print("len(horsedata.polygons):" + str(len(horsedata.polygons)))
    # a = 0
    # j = 0
    # for poly in horsedata.polygons:
    #     #print("Polygon index: %d, length: %d" % (poly.index, poly.loop_total))
    #     j = 0
    #     for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
    #         #print("Vertex: %d" % horsedata.loops[loop_index].vertex_index)
    #         horsedataFaces[a,j] = horsedata.loops[loop_index].vertex_index+1
    #         j += 1
    #     a+=1
    #         #print("    UV: %r" % uv_layer[loop_index].uv)
    #    # for idx in f.vertices:
    # #print("Polygons:" + horsedataFaces)
    # A_orig = np.zeros( (3*test_model.num_of_vertices, test_model.num_of_frames) )
    # A_appr = np.zeros( (3*test_model.num_of_vertices, test_model.num_of_frames) )
    # num_of_v = test_model.num_of_vertices
    # for f in range(0, test_model.num_of_frames):
    #     A_orig[:, f] = np.array([xyz for v in test_model.mesh_v_in_all_frames[f*num_of_v : f*num_of_v + num_of_v] for xyz in v])
    #     A_appr[:, f] = np.array([xyz for v in test_model.approximated_v[f] for xyz in v])
    #
    # Original = np.reshape(A_orig, (test_model.num_of_vertices, 3, -1))
    # Approximated = np.reshape(A_appr, (test_model.num_of_vertices, 3, -1))

    # # Face Normals Calculation ------------------------------------------------
    # print(" Frames: ", test_model.num_of_frames)
    # print("len(horsedata.polygons): ", len(horsedata.polygons))
    # horsedataNormals_orig = np.zeros( (3*len(horsedata.polygons), test_model.num_of_frames) )
    # horsedataNormals_approx = np.zeros( (3*len(horsedata.polygons), test_model.num_of_frames) )
    # for f in range(0, test_model.num_of_frames):
    #     for face in range(0, len(horsedataFaces)):
    #         faceVertices_orig = np.zeros((3, 3))
    #         faceVertices_approx = np.zeros((3, 3))
    #         for vertice in range(0, len(horsedataFaces[0])):
    #             v = int(horsedataFaces[face, vertice] - 1)
    #             #print("v: ", v)
    #             xyz_orig = np.array([ A_orig[v+0, f], A_orig[v+1, f], A_orig[v+2, f] ])
    #             xyz_approx = np.array([ A_appr[v+0, f], A_appr[v+1, f], A_appr[v+2, f] ])
    #             faceVertices_orig[vertice] = xyz_orig
    #             faceVertices_approx[vertice] = xyz_approx
    #         normal_orig = np.asarray(Normal(faceVertices_orig[0], faceVertices_orig[1], faceVertices_orig[2]))
    #         horsedataNormals_orig[(face*3), f] = normal_orig[0]
    #         horsedataNormals_orig[(face*3)+1, f] = normal_orig[1]
    #         horsedataNormals_orig[(face*3)+2, f] = normal_orig[2]
    #         normal_approx = np.asarray(Normal(faceVertices_approx[0], faceVertices_approx[1], faceVertices_approx[2]))
    #         horsedataNormals_approx[(face*3), f] = normal_approx[0]
    #         horsedataNormals_approx[(face*3)+1, f] = normal_approx[1]
    #         horsedataNormals_approx[(face*3)+2, f] = normal_approx[2]
    #
    # print("------------- ORIGINAL ----------------\n", len(horsedataNormals_orig), ", ", len(horsedataNormals_orig[0]))
    # print("------------- APPROXIMATED ----------------\n", len(horsedataNormals_approx), ", ", len(horsedataNormals_approx[0]))
    # print("------------- ORIGINAL ----------------\n", horsedataNormals_orig[0,0], ", ", horsedataNormals_orig[1,0], ", ", horsedataNormals_orig[2,0])
    # print("------------- APPROXIMATED ----------------\n", horsedataNormals_approx[0,0], ", ", horsedataNormals_approx[1,0], ", ", horsedataNormals_approx[2,0])
    # horsedataNormals_orig = np.reshape(horsedataNormals_orig, (len(horsedata.polygons), 3, -1))
    # horsedataNormals_approx = np.reshape(horsedataNormals_approx, (len(horsedata.polygons), 3, -1))
    #
    # # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+test_model_name+'Normals-Original.txt', horsedataNormals_orig[:,:,0], fmt="%.6f", delimiter=' ')
    # # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+test_model_name+'Normals-Approximation.txt', horsedataNormals_approx[:,:,0], fmt="%.6f", delimiter=' ')
    #
    # Normals_Distance  = np.zeros( (horsedataNormals_orig.shape[0], test_model.num_of_frames) )
    # # AVERAGE MAX DISTANCE ERROR METRIC
    # for f in range(0, horsedataNormals_orig.shape[2]):
    #     a = 0
    #     for v in range(0, horsedataNormals_orig.shape[0]):
    #         normal_xyz_orig = np.array([ horsedataNormals_orig[v, 0, f], horsedataNormals_orig[v, 1, f], horsedataNormals_orig[v, 2, f] ])
    #         normal_xyz_approx = np.array([horsedataNormals_approx[v, 0, f], horsedataNormals_approx[v, 1, f], horsedataNormals_approx[v, 2, f] ])
    #         # Normals_Distance[i, f] = np.linalg.norm(normal_xyz_orig - normal_xyz_approx)
    #         # print("\n-----------------\nnp.cross(normal_xyz_orig, normal_xyz_approx): ", np.cross(normal_xyz_orig, normal_xyz_approx))
    #         # print("\nnp.linalg.norm(np.cross(normal_xyz_orig, normal_xyz_approx)): ",np.linalg.norm(np.cross(normal_xyz_orig, normal_xyz_approx)))
    #         # print("\nnp.linalg.norm(normal_xyz_orig): ",np.linalg.norm(normal_xyz_orig),"\nnp.linalg.norm(normal_xyz_approx): ", np.linalg.norm(normal_xyz_approx))
    #         # print("(np.linalg.norm(normal_xyz_orig)* np.linalg.norm(normal_xyz_approx)): ", (np.linalg.norm(normal_xyz_orig)*np.linalg.norm(normal_xyz_approx)))
    #         Normals_Distance[a, f] = np.arcsin( np.linalg.norm(np.cross(normal_xyz_orig, normal_xyz_approx)) /  (np.linalg.norm(normal_xyz_orig)* np.linalg.norm(normal_xyz_approx)) )
    #         # print("\nNormals_Distance[i, f]: ", Normals_Distance[i, f],"\n----------------\n")
    #         a+=1
    #
    # # print("\n-----------\n", Normals_Distance)
    # MeanNormalDistance_Error = np.nanmean(Normals_Distance, axis=0)
    # # MaxDistanceNormal = np.amax(Normals_Distance, axis=0)
    # # Avg_MaxDistanceNormal_Error = np.average(MaxDistanceNormal)
    # # MinDistanceNormal = np.amin(Normals_Distance, axis=0)
    # # Avg_MinDistanceNormal_Error = np.average(MinDistanceNormal)
    # print("\n------------\nMeanNormalDistance_Error per frame - iter:", i, "error: ", MeanNormalDistance_Error)
    # initial_MeanNormalDistance_Error = np.mean(MeanNormalDistance_Error)
    # print("\n------------\nINITIAL Overall MeanNormalDistance_Error per frame - iter:", i, "error: ", initial_MeanNormalDistance_Error)
    # # print("\n------------\nAvg_MaxDistanceNormal_Error: ", Avg_MaxDistanceNormal_Error)
    # # print("Avg_MinDistanceNormal_Error: ", Avg_MinDistanceNormal_Error, "\n------------\n")
    # # print("Normals_Distance contains NAN: ", np.isnan(Normals_Distance).any())
    # AVG_NORMAL_ANGLE_DISTANCE.append(initial_MeanNormalDistance_Error)
    #
    initial_dE, initial_E_RMS, initial_Avg_Max_Distance_Error = linear_blend_skinning.calculate_errors(test_model)
    print('\tInitial Distortion Percentage, before W_f= ', initial_dE)
    print('\tInitial E_RMS, before W_f = ', initial_E_RMS)
    print('\tInitial Avg_Max_Distance, before W_f = ', initial_Avg_Max_Distance_Error)
    DistortionPercentage.append(initial_dE)
    RMS_error.append(initial_E_RMS)
    AVG_MAX_Distance.append(initial_Avg_Max_Distance_Error)
    print('\n')
    #
    end = timer()

    fitting_time += (end - start) - constr_time
    print('Time for fitting: ' + str((end - start)))

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
        print("------------- FOR LOOP  WEIGHTS (AFTER) --------------\n", new_v_weights[0])
        print("------------- FOR LOOP WEIGHTS TEST MODEL (AFTER) --------------\n", test_model.v_weights[0])
        print("------------- FOR LOOP  WEIGHTS (AFTER) --------------\n", new_v_weights[0])
        print("------------- FOR LOOP WEIGHTS TEST MODEL (AFTER) --------------\n", test_model.v_weights[0])
        print("------------- PREDICTIONS - WEIGHTS (AFTER) SUM 0:  --------------\n", np.sum(new_v_weights[0]))
        print("------------- PREDICTIONS - WEIGHTS (AFTER) SUM 10:  --------------\n", np.sum(new_v_weights[0]))

        end = timer()
        print('---- Time for weight fitting : ' + str(end - start))
        fitting_time += end - start

        start = timer()
        approx_v,constr_time = linear_blend_skinning.compute_approx_v_with_LBS(test_model)
        test_model.set_approximated_v(approx_v)
        end = timer()
        print('Time to compute approximated v: ' + str((end - start)-constr_time))
        fitting_time += end - start

        start = timer()
        new_dE, new_E_RMS, new_Avg_Max_Distance_Error = linear_blend_skinning.calculate_errors(test_model)
        old_errors.append(new_dE)
        end = timer()
        print('Time to compute errors: ' + str(end - start))
        print('\tDistortion Percentage after weight fitting = ', new_dE)
        print('\tE_RMS after weight fitting = ', new_E_RMS)
        print('\tAvg_Max_Distance after weight fitting = ', new_Avg_Max_Distance_Error)
        DistortionPercentage.append(new_dE)
        RMS_error.append(new_E_RMS)
        AVG_MAX_Distance.append(new_Avg_Max_Distance_Error)
        # horsedata = bpy.data.objects['fox1'].data
        # #horsedata = bpy.data.objects['tree_forg'].data
        # #horsedata = bpy.data.meshes['horse-gallop-01']
        # horsedataFaces = np.zeros( (len(horsedata.polygons), 3) )
        # print("len(horsedata.polygons):" + str(len(horsedata.polygons)))
        # a = 0
        # j = 0
        # for poly in horsedata.polygons:
        #     #print("Polygon index: %d, length: %d" % (poly.index, poly.loop_total))
        #     j = 0
        #     for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
        #         #print("Vertex: %d" % horsedata.loops[loop_index].vertex_index)
        #         horsedataFaces[a,j] = horsedata.loops[loop_index].vertex_index+1
        #         j += 1
        #     a+=1
        #         #print("    UV: %r" % uv_layer[loop_index].uv)
        #    # for idx in f.vertices:
        # #print("Polygons:" + horsedataFaces)
        # A_orig = np.zeros( (3*test_model.num_of_vertices, test_model.num_of_frames) )
        # A_appr = np.zeros( (3*test_model.num_of_vertices, test_model.num_of_frames) )
        # num_of_v = test_model.num_of_vertices
        # for f in range(0, test_model.num_of_frames):
        #     A_orig[:, f] = np.array([xyz for v in test_model.mesh_v_in_all_frames[f*num_of_v : f*num_of_v + num_of_v] for xyz in v])
        #     A_appr[:, f] = np.array([xyz for v in test_model.approximated_v[f] for xyz in v])
        #
        # Original = np.reshape(A_orig, (test_model.num_of_vertices, 3, -1))
        # Approximated = np.reshape(A_appr, (test_model.num_of_vertices, 3, -1))
        #
        # # Face Normals Calculation ------------------------------------------------
        # print(" Frames: ", test_model.num_of_frames)
        # print("len(horsedata.polygons): ", len(horsedata.polygons))
        # horsedataNormals_orig = np.zeros( (3*len(horsedata.polygons), test_model.num_of_frames) )
        # horsedataNormals_approx = np.zeros( (3*len(horsedata.polygons), test_model.num_of_frames) )
        # for f in range(0, test_model.num_of_frames):
        #     for face in range(0, len(horsedataFaces)):
        #         faceVertices_orig = np.zeros((3, 3))
        #         faceVertices_approx = np.zeros((3, 3))
        #         for vertice in range(0, len(horsedataFaces[0])):
        #             v = int(horsedataFaces[face, vertice] - 1)
        #             #print("v: ", v)
        #             xyz_orig = np.array([ A_orig[v+0, f], A_orig[v+1, f], A_orig[v+2, f] ])
        #             xyz_approx = np.array([ A_appr[v+0, f], A_appr[v+1, f], A_appr[v+2, f] ])
        #             faceVertices_orig[vertice] = xyz_orig
        #             faceVertices_approx[vertice] = xyz_approx
        #         normal_orig = np.asarray(Normal(faceVertices_orig[0], faceVertices_orig[1], faceVertices_orig[2]))
        #         horsedataNormals_orig[(face*3), f] = normal_orig[0]
        #         horsedataNormals_orig[(face*3)+1, f] = normal_orig[1]
        #         horsedataNormals_orig[(face*3)+2, f] = normal_orig[2]
        #         normal_approx = np.asarray(Normal(faceVertices_approx[0], faceVertices_approx[1], faceVertices_approx[2]))
        #         horsedataNormals_approx[(face*3), f] = normal_approx[0]
        #         horsedataNormals_approx[(face*3)+1, f] = normal_approx[1]
        #         horsedataNormals_approx[(face*3)+2, f] = normal_approx[2]
        #
        # print("------------- ORIGINAL ----------------\n", len(horsedataNormals_orig), ", ", len(horsedataNormals_orig[0]))
        # print("------------- APPROXIMATED ----------------\n", len(horsedataNormals_approx), ", ", len(horsedataNormals_approx[0]))
        # print("------------- ORIGINAL ----------------\n", horsedataNormals_orig[0,0], ", ", horsedataNormals_orig[1,0], ", ", horsedataNormals_orig[2,0])
        # print("------------- APPROXIMATED ----------------\n", horsedataNormals_approx[0,0], ", ", horsedataNormals_approx[1,0], ", ", horsedataNormals_approx[2,0])
        # horsedataNormals_orig = np.reshape(horsedataNormals_orig, (len(horsedata.polygons), 3, -1))
        # horsedataNormals_approx = np.reshape(horsedataNormals_approx, (len(horsedata.polygons), 3, -1))
        #
        # # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+test_model_name+'Normals-Original.txt', horsedataNormals_orig[:,:,0], fmt="%.6f", delimiter=' ')
        # # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+test_model_name+'Normals-Approximation.txt', horsedataNormals_approx[:,:,0], fmt="%.6f", delimiter=' ')
        #
        # Normals_Distance  = np.zeros( (horsedataNormals_orig.shape[0], test_model.num_of_frames) )
        # # AVERAGE MAX DISTANCE ERROR METRIC
        # for f in range(0, horsedataNormals_orig.shape[2]):
        #     a = 0
        #     for v in range(0, horsedataNormals_orig.shape[0]):
        #         normal_xyz_orig = np.array([ horsedataNormals_orig[v, 0, f], horsedataNormals_orig[v, 1, f], horsedataNormals_orig[v, 2, f] ])
        #         normal_xyz_approx = np.array([horsedataNormals_approx[v, 0, f], horsedataNormals_approx[v, 1, f], horsedataNormals_approx[v, 2, f] ])
        #         # Normals_Distance[i, f] = np.linalg.norm(normal_xyz_orig - normal_xyz_approx)
        #         # print("\n-----------------\nnp.cross(normal_xyz_orig, normal_xyz_approx): ", np.cross(normal_xyz_orig, normal_xyz_approx))
        #         # print("\nnp.linalg.norm(np.cross(normal_xyz_orig, normal_xyz_approx)): ",np.linalg.norm(np.cross(normal_xyz_orig, normal_xyz_approx)))
        #         # print("\nnp.linalg.norm(normal_xyz_orig): ",np.linalg.norm(normal_xyz_orig),"\nnp.linalg.norm(normal_xyz_approx): ", np.linalg.norm(normal_xyz_approx))
        #         # print("(np.linalg.norm(normal_xyz_orig)* np.linalg.norm(normal_xyz_approx)): ", (np.linalg.norm(normal_xyz_orig)*np.linalg.norm(normal_xyz_approx)))
        #         Normals_Distance[a, f] = np.arcsin( np.linalg.norm(np.cross(normal_xyz_orig, normal_xyz_approx)) /  (np.linalg.norm(normal_xyz_orig)* np.linalg.norm(normal_xyz_approx)) )
        #         # print("\nNormals_Distance[i, f]: ", Normals_Distance[i, f],"\n----------------\n")
        #         a+=1
        #
        # # print("\n-----------\n", Normals_Distance)
        # MeanNormalDistance_Error = np.nanmean(Normals_Distance, axis=0)
        # # MaxDistanceNormal = np.amax(Normals_Distance, axis=0)
        # # Avg_MaxDistanceNormal_Error = np.average(MaxDistanceNormal)
        # # MinDistanceNormal = np.amin(Normals_Distance, axis=0)
        # # Avg_MinDistanceNormal_Error = np.average(MinDistanceNormal)
        # print("\n------------\nMeanNormalDistance_Error per frame - iter:", i, "error: ", MeanNormalDistance_Error)
        # MeanNormalDistance_Error = np.mean(MeanNormalDistance_Error)
        # print("\n------------\nOverall MeanNormalDistance_Error per frame - iter:", i, "error: ", MeanNormalDistance_Error)
        # # print("\n------------\nAvg_MaxDistanceNormal_Error: ", Avg_MaxDistanceNormal_Error)
        # # print("Avg_MinDistanceNormal_Error: ", Avg_MinDistanceNormal_Error, "\n------------\n")
        # # print("Normals_Distance contains NAN: ", np.isnan(Normals_Distance).any())
        # AVG_NORMAL_ANGLE_DISTANCE.append(MeanNormalDistance_Error)
        # fitting_time += end - start

        # transformation fitting and computation of errors
        start = timer()
        new_transformations = fitting.fit_affine_transformations(test_model)

        #new_transformations_tensor = fitting.fit_affine_transformations(test_model)
        #with sess.as_default():
        #    new_transformations = sess.run(tf.transpose(new_transformations_tensor))

        print("\n\nCURRENT transformations", new_transformations.shape)
        #print("\n\nINITIAL transformations TENSORFLOW", new_transformations_my.shape)

        old_w_t.append(new_transformations)
        test_model.set_affine_transformations(new_transformations)
        end = timer()
        print('---- Time for transformation fitting : ' + str(end - start))
        fitting_time += end - start

        start = timer()
        approx_v, constr_time = linear_blend_skinning.compute_approx_v_with_LBS(test_model)
        test_model.set_approximated_v(approx_v)
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
        print('\tAvg_Max_Distance after transformation fitting = ', new_Avg_Max_Distance_Error)
        DistortionPercentage.append(new_dE)
        RMS_error.append(new_E_RMS)
        AVG_MAX_Distance.append(new_Avg_Max_Distance_Error)

        if (i == (number_of_fitting_steps - 1)):
            np.savetxt(os.path.dirname(__file__)+'\\outcome\\'+networkPrefix+'FinalTransformations.txt', np.transpose(new_transformations), fmt="%.6f", delimiter=' ')
            np.savetxt(os.path.dirname(__file__)+'\\outcome\\'+networkPrefix+'FinalWeights.txt', new_v_weights, fmt="%.6f", delimiter=' ')
            np.savetxt(os.path.dirname(__file__)+'\\outcome\\'+networkPrefix+'FinalWeightIndices.txt', new_v_weights_indices, fmt="%.df", delimiter=' ')
            print("Approximated v: ", len(approx_v))
            for f in range(0, test_model.num_of_frames):
                print("frame: ", str(f))
                np.savetxt(os.path.dirname(__file__)+'\\outcome\\'+networkPrefix+"-"+str(f)+'FinalApproximatedSequence.txt', approx_v[f], fmt="%.6f", delimiter=' ')
            approximated_sequence = np.reshape(approx_v, (-1, 3))
            np.savetxt(os.path.dirname(__file__)+'\\outcome\\'+networkPrefix+"-"+'FinalApproximatedSequence.txt', approximated_sequence, fmt="%.6f", delimiter=' ')

        fitting_time += end - start


    #     horsedata = bpy.data.objects['fox1'].data
    #     #horsedata = bpy.data.objects['tree_forg'].data
    #     #horsedata = bpy.data.meshes['horse-gallop-01']
    #     horsedataFaces = np.zeros( (len(horsedata.polygons), 3) )
    #     print("len(horsedata.polygons):" + str(len(horsedata.polygons)))
    #     a = 0
    #     j = 0
    #     for poly in horsedata.polygons:
    #         #print("Polygon index: %d, length: %d" % (poly.index, poly.loop_total))
    #         j = 0
    #         for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
    #             #print("Vertex: %d" % horsedata.loops[loop_index].vertex_index)
    #             horsedataFaces[a,j] = horsedata.loops[loop_index].vertex_index+1
    #             j += 1
    #         a+=1
    #             #print("    UV: %r" % uv_layer[loop_index].uv)
    #        # for idx in f.vertices:
    #     #print("Polygons:" + horsedataFaces)
    #     A_orig = np.zeros( (3*test_model.num_of_vertices, test_model.num_of_frames) )
    #     A_appr = np.zeros( (3*test_model.num_of_vertices, test_model.num_of_frames) )
    #     num_of_v = test_model.num_of_vertices
    #     for f in range(0, test_model.num_of_frames):
    #         A_orig[:, f] = np.array([xyz for v in test_model.mesh_v_in_all_frames[f*num_of_v : f*num_of_v + num_of_v] for xyz in v])
    #         A_appr[:, f] = np.array([xyz for v in test_model.approximated_v[f] for xyz in v])
    #
    #     Original = np.reshape(A_orig, (test_model.num_of_vertices, 3, -1))
    #     Approximated = np.reshape(A_appr, (test_model.num_of_vertices, 3, -1))
    #
    #     # Face Normals Calculation ------------------------------------------------
    #     print(" Frames: ", test_model.num_of_frames)
    #     print("len(horsedata.polygons): ", len(horsedata.polygons))
    #     horsedataNormals_orig = np.zeros( (3*len(horsedata.polygons), test_model.num_of_frames) )
    #     horsedataNormals_approx = np.zeros( (3*len(horsedata.polygons), test_model.num_of_frames) )
    #     for f in range(0, test_model.num_of_frames):
    #         for face in range(0, len(horsedataFaces)):
    #             faceVertices_orig = np.zeros((3, 3))
    #             faceVertices_approx = np.zeros((3, 3))
    #             for vertice in range(0, len(horsedataFaces[0])):
    #                 v = int(horsedataFaces[face, vertice] - 1)
    #                 #print("v: ", v)
    #                 xyz_orig = np.array([ A_orig[v+0, f], A_orig[v+1, f], A_orig[v+2, f] ])
    #                 xyz_approx = np.array([ A_appr[v+0, f], A_appr[v+1, f], A_appr[v+2, f] ])
    #                 faceVertices_orig[vertice] = xyz_orig
    #                 faceVertices_approx[vertice] = xyz_approx
    #             normal_orig = np.asarray(Normal(faceVertices_orig[0], faceVertices_orig[1], faceVertices_orig[2]))
    #             horsedataNormals_orig[(face*3), f] = normal_orig[0]
    #             horsedataNormals_orig[(face*3)+1, f] = normal_orig[1]
    #             horsedataNormals_orig[(face*3)+2, f] = normal_orig[2]
    #             normal_approx = np.asarray(Normal(faceVertices_approx[0], faceVertices_approx[1], faceVertices_approx[2]))
    #             horsedataNormals_approx[(face*3), f] = normal_approx[0]
    #             horsedataNormals_approx[(face*3)+1, f] = normal_approx[1]
    #             horsedataNormals_approx[(face*3)+2, f] = normal_approx[2]
    #
    #     print("------------- ORIGINAL ----------------\n", len(horsedataNormals_orig), ", ", len(horsedataNormals_orig[0]))
    #     print("------------- APPROXIMATED ----------------\n", len(horsedataNormals_approx), ", ", len(horsedataNormals_approx[0]))
    #     print("------------- ORIGINAL ----------------\n", horsedataNormals_orig[0,0], ", ", horsedataNormals_orig[1,0], ", ", horsedataNormals_orig[2,0])
    #     print("------------- APPROXIMATED ----------------\n", horsedataNormals_approx[0,0], ", ", horsedataNormals_approx[1,0], ", ", horsedataNormals_approx[2,0])
    #     horsedataNormals_orig = np.reshape(horsedataNormals_orig, (len(horsedata.polygons), 3, -1))
    #     horsedataNormals_approx = np.reshape(horsedataNormals_approx, (len(horsedata.polygons), 3, -1))
    #
    #     # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+test_model_name+'Normals-Original.txt', horsedataNormals_orig[:,:,0], fmt="%.6f", delimiter=' ')
    #     # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+test_model_name+'Normals-Approximation.txt', horsedataNormals_approx[:,:,0], fmt="%.6f", delimiter=' ')
    #
    #     Normals_Distance  = np.zeros( (horsedataNormals_orig.shape[0], test_model.num_of_frames) )
    #     # AVERAGE MAX DISTANCE ERROR METRIC
    #     for f in range(0, horsedataNormals_orig.shape[2]):
    #         a = 0
    #         for v in range(0, horsedataNormals_orig.shape[0]):
    #             normal_xyz_orig = np.array([ horsedataNormals_orig[v, 0, f], horsedataNormals_orig[v, 1, f], horsedataNormals_orig[v, 2, f] ])
    #             normal_xyz_approx = np.array([horsedataNormals_approx[v, 0, f], horsedataNormals_approx[v, 1, f], horsedataNormals_approx[v, 2, f] ])
    #             # Normals_Distance[i, f] = np.linalg.norm(normal_xyz_orig - normal_xyz_approx)
    #             # print("\n-----------------\nnp.cross(normal_xyz_orig, normal_xyz_approx): ", np.cross(normal_xyz_orig, normal_xyz_approx))
    #             # print("\nnp.linalg.norm(np.cross(normal_xyz_orig, normal_xyz_approx)): ",np.linalg.norm(np.cross(normal_xyz_orig, normal_xyz_approx)))
    #             # print("\nnp.linalg.norm(normal_xyz_orig): ",np.linalg.norm(normal_xyz_orig),"\nnp.linalg.norm(normal_xyz_approx): ", np.linalg.norm(normal_xyz_approx))
    #             # print("(np.linalg.norm(normal_xyz_orig)* np.linalg.norm(normal_xyz_approx)): ", (np.linalg.norm(normal_xyz_orig)*np.linalg.norm(normal_xyz_approx)))
    #             Normals_Distance[a, f] = np.arcsin( np.linalg.norm(np.cross(normal_xyz_orig, normal_xyz_approx)) /  (np.linalg.norm(normal_xyz_orig)* np.linalg.norm(normal_xyz_approx)) )
    #             # print("\nNormals_Distance[i, f]: ", Normals_Distance[i, f],"\n----------------\n")
    #             a+=1
    #
    #     # print("\n-----------\n", Normals_Distance)
    #     MeanNormalDistance_Error = np.nanmean(Normals_Distance, axis=0)
    #     # MaxDistanceNormal = np.amax(Normals_Distance, axis=0)
    #     # Avg_MaxDistanceNormal_Error = np.average(MaxDistanceNormal)
    #     # MinDistanceNormal = np.amin(Normals_Distance, axis=0)
    #     # Avg_MinDistanceNormal_Error = np.average(MinDistanceNormal)
    #     print("\n------------\nMeanNormalDistance_Error per frame - iter:", i, "error: ", MeanNormalDistance_Error)
    #     MeanNormalDistance_Error = np.mean(MeanNormalDistance_Error)
    #     print("\n------------\nOverall MeanNormalDistance_Error per frame - iter:", i, "error: ", MeanNormalDistance_Error)
    #     # print("\n------------\nAvg_MaxDistanceNormal_Error: ", Avg_MaxDistanceNormal_Error)
    #     # print("Avg_MinDistanceNormal_Error: ", Avg_MinDistanceNormal_Error, "\n------------\n")
    #     # print("Normals_Distance contains NAN: ", np.isnan(Normals_Distance).any())
    #     AVG_NORMAL_ANGLE_DISTANCE.append(MeanNormalDistance_Error)
    #
    # time_for_each_step.append('\tfor fitting : ' + str(fitting_time))
    #
     # set the combination of weights and transformations that produced the best result during the fitting steps
    index_of_min_error = old_errors.index(min(old_errors))
    print("index_of_min_error: ", index_of_min_error)
    if index_of_min_error % 2 == 1:
        test_model.set_affine_transformations(old_w_t[index_of_min_error])
        test_model.set_v_weights(old_w_t[index_of_min_error - 1], old_v_weights_indices[index_of_min_error // 2])
        # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'FinalTransformations.csv', np.transpose(old_w_t[index_of_min_error]), fmt="%.6f", delimiter=' ')
        # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'FinalWeights.csv', old_w_t[index_of_min_error - 1], fmt="%.6f", delimiter=' ')
    else:
        test_model.set_v_weights(old_w_t[index_of_min_error], old_v_weights_indices[index_of_min_error // 2])
        test_model.set_affine_transformations(old_w_t[index_of_min_error - 1])
    #     np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'FinalTransformations.csv', np.transpose(old_w_t[index_of_min_error - 1]), fmt="%.6f", delimiter=' ')
    #     np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'FinalWeights.csv', old_w_t[index_of_min_error], fmt="%.6f", delimiter=' ')
    # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'FinalWeightIndices.csv', old_v_weights_indices[index_of_min_error // 2], fmt="%.df", delimiter=' ')

    # linear blend skinning
    start = timer()
    approx_v, constr_time = linear_blend_skinning.compute_approx_v_with_LBS(test_model)

    test_model.set_approximated_v(approx_v)
    end = timer()
    time_for_each_step.append('\tfor last linear blend skinning : ' + str(end - start))

    # compute error produced by linear blend skinning
    start = timer()
    dE, E_RMS, Avg_Max_Distance_Error = linear_blend_skinning.calculate_errors(test_model)
    end = timer()
    time_for_each_step.append('\tto calculate errors : ' + str(end - start))

    # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+test_model_name+'Dis_Per.txt', np.asarray(DistortionPercentage), fmt="%.6f", delimiter='|')
    # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+test_model_name+'RMS_E.txt', np.asarray(RMS_error), fmt="%.6f", delimiter='|')
    # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+test_model_name+'Avg_Max_Dis.txt', np.asarray(AVG_MAX_Distance), fmt="%.6f", delimiter='|')
    # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+test_model_name+'Avg_Normal_Angle_Dist.txt', np.asarray(AVG_NORMAL_ANGLE_DISTANCE), fmt="%.6f", delimiter='|')

    # horsedata = bpy.data.objects['fox1'].data
    # #horsedata = bpy.data.objects['tree_forg'].data
    # #horsedata = bpy.data.meshes['horse-gallop-01']
    # horsedataFaces = np.zeros( (len(horsedata.polygons), 3) )
    # print("len(horsedata.polygons):" + str(len(horsedata.polygons)))
    # i = 0
    # j = 0
    # for poly in horsedata.polygons:
    #     #print("Polygon index: %d, length: %d" % (poly.index, poly.loop_total))
    #     j = 0
    #     for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
    #         #print("Vertex: %d" % horsedata.loops[loop_index].vertex_index)
    #         horsedataFaces[i,j] = horsedata.loops[loop_index].vertex_index+1
    #         j += 1
    #     i+=1
    #         #print("    UV: %r" % uv_layer[loop_index].uv)
    #    # for idx in f.vertices:
    # #print("Polygons:" + horsedataFaces)
    # A_orig = np.zeros( (3*test_model.num_of_vertices, test_model.num_of_frames) )
    # A_appr = np.zeros( (3*test_model.num_of_vertices, test_model.num_of_frames) )
    # num_of_v = test_model.num_of_vertices
    # for f in range(0, test_model.num_of_frames):
    #     A_orig[:, f] = np.array([xyz for v in test_model.mesh_v_in_all_frames[f*num_of_v : f*num_of_v + num_of_v] for xyz in v])
    #     A_appr[:, f] = np.array([xyz for v in test_model.approximated_v[f] for xyz in v])
    #
    # Original = np.reshape(A_orig, (test_model.num_of_vertices, 3, -1))
    # Approximated = np.reshape(A_appr, (test_model.num_of_vertices, 3, -1))
    #
    # # Face Normals Calculation ------------------------------------------------
    # print(" Frames: ", test_model.num_of_frames)
    # print("len(horsedata.polygons): ", len(horsedata.polygons))
    # horsedataNormals_orig = np.zeros( (3*len(horsedata.polygons), test_model.num_of_frames) )
    # horsedataNormals_approx = np.zeros( (3*len(horsedata.polygons), test_model.num_of_frames) )
    # for f in range(0, test_model.num_of_frames):
    #     for face in range(0, len(horsedataFaces)):
    #         faceVertices_orig = np.zeros((3, 3))
    #         faceVertices_approx = np.zeros((3, 3))
    #         for vertice in range(0, len(horsedataFaces[0])):
    #             v = int(horsedataFaces[face, vertice] - 1)
    #             #print("v: ", v)
    #             xyz_orig = np.array([ A_orig[v+0, f], A_orig[v+1, f], A_orig[v+2, f] ])
    #             xyz_approx = np.array([ A_appr[v+0, f], A_appr[v+1, f], A_appr[v+2, f] ])
    #             faceVertices_orig[vertice] = xyz_orig
    #             faceVertices_approx[vertice] = xyz_approx
    #         normal_orig = np.asarray(Normal(faceVertices_orig[0], faceVertices_orig[1], faceVertices_orig[2]))
    #         horsedataNormals_orig[(face*3), f] = normal_orig[0]
    #         horsedataNormals_orig[(face*3)+1, f] = normal_orig[1]
    #         horsedataNormals_orig[(face*3)+2, f] = normal_orig[2]
    #         normal_approx = np.asarray(Normal(faceVertices_approx[0], faceVertices_approx[1], faceVertices_approx[2]))
    #         horsedataNormals_approx[(face*3), f] = normal_approx[0]
    #         horsedataNormals_approx[(face*3)+1, f] = normal_approx[1]
    #         horsedataNormals_approx[(face*3)+2, f] = normal_approx[2]
    #
    # print("------------- ORIGINAL ----------------\n", len(horsedataNormals_orig), ", ", len(horsedataNormals_orig[0]))
    # print("------------- APPROXIMATED ----------------\n", len(horsedataNormals_approx), ", ", len(horsedataNormals_approx[0]))
    # print("------------- ORIGINAL ----------------\n", horsedataNormals_orig[0,0], ", ", horsedataNormals_orig[1,0], ", ", horsedataNormals_orig[2,0])
    # print("------------- APPROXIMATED ----------------\n", horsedataNormals_approx[0,0], ", ", horsedataNormals_approx[1,0], ", ", horsedataNormals_approx[2,0])
    # horsedataNormals_orig = np.reshape(horsedataNormals_orig, (len(horsedata.polygons), 3, -1))
    # horsedataNormals_approx = np.reshape(horsedataNormals_approx, (len(horsedata.polygons), 3, -1))
    #
    # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+test_model_name+'Normals-Original.txt', horsedataNormals_orig[:,:,0], fmt="%.6f", delimiter=' ')
    # np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+test_model_name+'Normals-Approximation.txt', horsedataNormals_approx[:,:,0], fmt="%.6f", delimiter=' ')
    #
    # Normals_Distance  = np.zeros( (horsedataNormals_orig.shape[0], test_model.num_of_frames) )
    # # AVERAGE MAX DISTANCE ERROR METRIC
    # for f in range(0, horsedataNormals_orig.shape[2]):
    #     i = 0
    #     for v in range(0, horsedataNormals_orig.shape[0]):
    #         normal_xyz_orig = np.array([ horsedataNormals_orig[v, 0, f], horsedataNormals_orig[v, 1, f], horsedataNormals_orig[v, 2, f] ])
    #         normal_xyz_approx = np.array([horsedataNormals_approx[v, 0, f], horsedataNormals_approx[v, 1, f], horsedataNormals_approx[v, 2, f] ])
    #         # Normals_Distance[i, f] = np.linalg.norm(normal_xyz_orig - normal_xyz_approx)
    #         # print("\n-----------------\nnp.cross(normal_xyz_orig, normal_xyz_approx): ", np.cross(normal_xyz_orig, normal_xyz_approx))
    #         # print("\nnp.linalg.norm(np.cross(normal_xyz_orig, normal_xyz_approx)): ",np.linalg.norm(np.cross(normal_xyz_orig, normal_xyz_approx)))
    #         # print("\nnp.linalg.norm(normal_xyz_orig): ",np.linalg.norm(normal_xyz_orig),"\nnp.linalg.norm(normal_xyz_approx): ", np.linalg.norm(normal_xyz_approx))
    #         # print("(np.linalg.norm(normal_xyz_orig)* np.linalg.norm(normal_xyz_approx)): ", (np.linalg.norm(normal_xyz_orig)*np.linalg.norm(normal_xyz_approx)))
    #         Normals_Distance[i, f] = np.arcsin( np.linalg.norm(np.cross(normal_xyz_orig, normal_xyz_approx)) /  (np.linalg.norm(normal_xyz_orig)* np.linalg.norm(normal_xyz_approx)) )
    #         # print("\nNormals_Distance[i, f]: ", Normals_Distance[i, f],"\n----------------\n")
    #         i+=1
    #
    # # print("\n-----------\n", Normals_Distance)
    # MeanNormalDistance_Error = np.nanmean(Normals_Distance, axis=0)
    # # MaxDistanceNormal = np.amax(Normals_Distance, axis=0)
    # # Avg_MaxDistanceNormal_Error = np.average(MaxDistanceNormal)
    # # MinDistanceNormal = np.amin(Normals_Distance, axis=0)
    # # Avg_MinDistanceNormal_Error = np.average(MinDistanceNormal)
    # print("\n------------\nMeanNormalDistance_Error per frame: ", MeanNormalDistance_Error)
    # MeanNormalDistance_Error = np.mean(MeanNormalDistance_Error)
    # print("\n------------\nOverall MeanNormalDistance_Error: ", MeanNormalDistance_Error)
    # # print("\n------------\nAvg_MaxDistanceNormal_Error: ", Avg_MaxDistanceNormal_Error)
    # # print("Avg_MinDistanceNormal_Error: ", Avg_MinDistanceNormal_Error, "\n------------\n")
    # # print("Normals_Distance contains NAN: ", np.isnan(Normals_Distance).any())
    #
    #
    # Approximated = np.reshape(A_appr, (test_model.num_of_vertices, 3, -1))
    # DistanceFromOriginal = np.zeros( (test_model.num_of_vertices, test_model.num_of_frames) )
    # for f in range(0, test_model.num_of_frames):
    #     i = 0
    #     for v in range(0, num_of_v*3, 3):
    #         xyz_orig = np.array([ A_orig[v+0, f], A_orig[v+1, f], A_orig[v+2, f] ])
    #         xyz_approx = np.array([ A_appr[v+0, f], A_appr[v+1, f], A_appr[v+2, f] ])
    #         DistanceFromOriginal[i, f] = np.linalg.norm(xyz_orig - xyz_approx)
    #         i+=1
    # Colors_Medians = np.median(DistanceFromOriginal, axis=0)
    # #print("Colors_Medians ",Colors_Medians[0])
    #
    # ApproximatedWithColor = np.zeros((test_model.num_of_vertices, 6, test_model.num_of_frames))
    # for f in range(0, test_model.num_of_frames):
    #     for v in range(0, test_model.num_of_vertices):
    #         for xyzrgb in range(0, 3):
    #             ApproximatedWithColor[v,xyzrgb,f] = Approximated[v,xyzrgb,f]
    #         #colors in r g b
    #         rgb = interpolate(turbo_colormap_data,translate(DistanceFromOriginal[v, f], DistanceFromOriginal[:,f].min(), DistanceFromOriginal[:,f].max(), 0, 1))
    #         ApproximatedWithColor[v,3,f] = rgb[0]
    #         ApproximatedWithColor[v,4,f] = rgb[1]
    #         ApproximatedWithColor[v,5,f] = rgb[2]
    #
    # for f in range(0, test_model.num_of_frames):
    #     np.savetxt(os.path.dirname(__file__)+'\\spiderman\\'+networkPrefix+"-"+test_model_name+"_approx"+str(f)+".obj", ApproximatedWithColor[:,:,f], fmt="%.6f", newline="\nv ",header="#horse gallop model approximated!")
    #     with open(os.path.dirname(__file__)+'\\spiderman\\'+networkPrefix+"-"+test_model_name+"_approx"+str(f)+".obj", "ab") as file:
    #         np.savetxt(file, horsedataFaces,fmt="%d", newline="\nf ")
    #
    #
    # clusterColours = cluster_color_data
    # print("----------- CLUSTERS COLORS --------------\n", clusterColours)
    # # TEST LABELS size (vertices X number_of_proxy_bones)
    # print("----------- CLUSTERS Ind --------------\n", colorIndices)
    # colorIndices = np.array(colorIndices)
    # colorIndices = colorIndices[0]
    # weights_final = test_model.get_v_weights()
    # print("----------- weights_final --------------\n", weights_final[0])
    # print("------------- INDICES MAX LABELS ", str(v),"--------------\n", indices_test_set_labels[0])
    # print("----------- CLUSTERS ind after --------------\n", colorIndices)
    #
    # # for f in range(0, test_model.num_of_frames):
    # #     for v in range(0, test_model.num_of_vertices):
    # #         max = -1
    # #         maxPosition = 0
    # #         for w in range(0, len(weights_final[v])):
    # #             if(max >= weights_final[v][w]):
    # #                 max = weights_final[v][w]
    # #                 maxPosition = w
    # #         indexMax = indices_test_set_labels[v][maxPosition]
    # #         #print("----------- indexMax --------------\n", indexMax)
    # #         colorMaxPosition = 0
    # #         for c in range(0, colorIndices.size):
    # #             if(colorIndices[c] == indexMax):
    # #                 colorMaxPosition = c
    # #         #print("----------- colorMaxPosition --------------\n", colorMaxPosition)
    # #         ApproximatedWithColor[v,3,f] = clusterColours[colorMaxPosition][0]
    # #         ApproximatedWithColor[v,4,f] = clusterColours[colorMaxPosition][1]
    # #         ApproximatedWithColor[v,5,f] = clusterColours[colorMaxPosition][2]
    #
    # for f in range(0, test_model.num_of_frames):
    #     for v in range(0, test_model.num_of_vertices):
    #         counter=0
    #         for p in range(0, colorIndices.size):
    #             # print("----------- colorIndices[p] --------------\n", colorIndices[p])
    #             # print("----------- v  --------------\n", v)
    #             # print("----------- p  --------------\n", p)
    #             # print("----------- test_set_labels_original[v][colorIndices[p]] --------------\n", test_set_labels_original[v][colorIndices[p]])
    #             if(test_set_labels_original[v][colorIndices[p]] == 1):
    #                 if(p == 0):
    #                     ApproximatedWithColor[v,3,f] = clusterColours[p][0]
    #                     ApproximatedWithColor[v,4,f] = clusterColours[p][1]
    #                     ApproximatedWithColor[v,5,f] = clusterColours[p][2]
    #                 else:
    #                     ApproximatedWithColor[v,3,f] += clusterColours[p][0]
    #                     ApproximatedWithColor[v,4,f] += clusterColours[p][1]
    #                     ApproximatedWithColor[v,5,f] += clusterColours[p][2]
    #                 counter+=1
    #         ApproximatedWithColor[v,3,f] /= counter
    #         ApproximatedWithColor[v,4,f] /= counter
    #         ApproximatedWithColor[v,5,f] /= counter
    #
    # for f in range(0, test_model.num_of_frames):
    #     np.savetxt(os.path.dirname(__file__)+'\\spiderman\\clusters\\'+networkPrefix+test_model_name+"_approx"+str(f)+".obj", ApproximatedWithColor[:,:,f], fmt="%.6f", newline="\nv ",header="#horse gallop model approximated!")
    #     with open(os.path.dirname(__file__)+'\\spiderman\\clusters\\'+networkPrefix+test_model_name+"_approx"+str(f)+".obj", "ab") as file:
    #         np.savetxt(file, horsedataFaces,fmt="%d", newline="\nf ")

    # construct array with approximated poses and init variables for the animation
    start = timer()
    approximated_animation_sequence.construct_array_with_approx_poses(test_model)
    approximated_animation_sequence.init_approx_animation_variables()
    end = timer()
    time_for_each_step.append('\tto construct array with approx poses and init variables for animation: ' + str(end - start))

    print('Error metrics :')
    print('\tDistortion Percentage = ', dE)
    print('\tE_RMS = ', E_RMS)
    print('\tAvg_Max_Distance_Error = ', Avg_Max_Distance_Error)

    for step in time_for_each_step:
        print(step)


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
            bpy.data.objects[name_of_obj_file].hide = False
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

# add vertices to a vertex group. Those vertices were initially predicted not to belong to any vertex group
def assign_unassigned_vertices_to_bone(test_set_labels, predictions):
    test_v_in_which_v_groups = []
    for i in range(0, len(test_set_labels) ):
        in_which_v_groups = []
        for j in range(0, len(test_set_labels[i]) ):
            if test_set_labels[i][j] == 1:
                in_which_v_groups.append(j)
        if not in_which_v_groups:
            predictions_i_max = np.argmax(predictions[i])
            in_which_v_groups.append(predictions_i_max)
        test_v_in_which_v_groups.append(in_which_v_groups)
    return test_v_in_which_v_groups
