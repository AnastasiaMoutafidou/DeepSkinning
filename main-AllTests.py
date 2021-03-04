import sys
import os
import bpy
from timeit import default_timer as timer

class PathsForBlender(object):
    def __init__(self, train_model_paths, test_model_path, file_type, list_of_obj_files_in_folder = []):
        self.train_model_paths = train_model_paths
        self.test_model_path = test_model_path
        self.file_type = file_type
        self.list_of_obj_files_in_folder = list_of_obj_files_in_folder

# remove mesh Cube
if "Cube" in bpy.data.meshes:
    mesh = bpy.data.meshes["Cube"]
    print("removing mesh", mesh)
    bpy.data.meshes.remove(mesh)

human_array = ['Spiderman', 'Cross', 'RunToRolling', 'SWAT']
animal_array = ['lion', 'Sqriell Jump', 'Sailfish', 'turtle']
#human_array = ['Spiderman']
#animal_array = ['fox', 'Sailfish']

for iter in range(0, (len(human_array) + len(animal_array))):

    start = timer()
    #train_model_path_3 = 'C:\\Users\\User\\Desktop\\animations\\Alien-Animal.fbx'
    #train_model_path_4 = 'C:\\Users\\User\\Desktop\\animations\\Dragon.fbx'
    #train_model_path_5 = 'C:\\Users\\User\\Desktop\\animations\\Tree_frog.fbx'

    train_model_paths = []
    #fbx_path = 'C:\\Users\\User\\Desktop\\animations\\DataSet_Animals\\dataset\\animal_'
    #for i in range(1, 14):
    #    train_model_paths.append(fbx_path+str(i)+".fbx")

    #train_model_paths.append(train_model_path_1)
    #train_model_paths.append(train_model_path_2)
    #train_model_paths.append(train_model_path_3)
    #train_model_paths.append(train_model_path_4)
    '''train_model_paths.append(train_model_path_5)
    train_model_paths.append(train_model_path_6)
    train_model_paths.append(train_model_path_7)
    #train_model_paths.append(train_model_path_8)
    train_model_paths.append(train_model_path_9)
    train_model_paths.append(train_model_path_10)
    '''
    # ANIMALS DATASET
    '''for i in range(1, 25):
        if ( i != 8 and i!= 22):
            train_model_paths.append('C:\\Users\\User\\Desktop\\animations\\DataSet_Animals\\dataset\\animal_'+str(i)+'.fbx')

    # HUMANS DATASET
    for i in range(1, 31):
       if ( i != 11 and i != 15and i != 17 and i != 20 and i!= 21 and i != 25  and i != 28 ):
           train_model_paths.append('C:\\Users\\User\\Desktop\\animations\\DataSetHumans\\Human_'+str(i)+'.fbx')'''
    human = ""
    animal =""
    if (iter < len(human_array)):
        human = human_array[iter]
        test_model_path = 'C:\\Users\\User\\Desktop\\animations\\DataSetHumans\\'+human+'.fbx'
    else:
        animal = animal_array[iter-len(human_array)]
        test_model_path = 'C:\\Users\\User\\Desktop\\animations\\DataSet_Animals\\dataset\\'+animal+'.fbx'

    #test_model_path = 'C:\\Users\\User\\Desktop\\animations\\DataSetHumans\\human_'+str(3)+'.fbx'
    #test_model_path = 'C:\\Users\\User\\Desktop\\animations\\DataSet_Animals\\dataset\\lion.fbx'

    #test_model_path = 'C:\\Users\\User\\Desktop\\animations\\DataSet_Animals\\dataset\\horse-gallop\\'
    train_model_paths.append(test_model_path) # for evaluation

    file_path = 'C:\\Users\\User\\Desktop\\skeleton_training\\scripts'
    python_packages_path = 'C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python35\\Lib\\site-packages'
    sys.path.append(file_path)
    sys.path.insert(0, python_packages_path)

    import tensorflow_script
    import tensorflow_script_without_training
    import p_center
    import imp

    print('\n---------- NEW RUN ----------')
    print('\n')

    if 'fbx' in test_model_path:
        file_type = 'fbx'
        pathsForBlender = PathsForBlender(train_model_paths, test_model_path, file_type)
    else:
        list_of_obj_files_in_folder = os.listdir(test_model_path)
        file_type = 'obj'
        pathsForBlender = PathsForBlender(train_model_paths, test_model_path, file_type, list_of_obj_files_in_folder)

    imp.reload(tensorflow_script)
    imp.reload(tensorflow_script_without_training)

    # for p-center clustering comment tensorflow_scipt.start_tensorflow(...) line and uncomment p_center...

    if (iter < len(human_array)):
        #tensorflow_script.start_tensorflow(pathsForBlender, "General-1024-RNN-LSTM-")
        tensorflow_script_without_training.start_tensorflow_without_training(pathsForBlender, "General-4096-CNN2x8- ", human)
        #p_center.p_center_clustering(pathsForBlender, animal)
    else:
        #tensorflow_script.start_tensorflow(pathsForBlender, "General-1024-RNN-LSTM-")
        tensorflow_script_without_training.start_tensorflow_without_training(pathsForBlender, "General-4096-CNN2x8- ", animal)
        #p_center.p_center_clustering(pathsForBlender, animal)

    print('\n')
    print('!!!!! Script successfully finished running !!!!!')
    end = timer()
    print('time required to run the whole application : ', end - start)

    # Delect objects by type
    for o in bpy.context.scene.objects:
        o.select = True
    # Call the operator only once
    bpy.ops.object.delete()
