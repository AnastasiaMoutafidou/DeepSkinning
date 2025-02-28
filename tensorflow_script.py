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

def start_tensorflow(paths_for_Blender, networkPrefix, customNumbering):
	train_model_names_list = []
	test_model_names_list = []
	train_models = []
	validation_models = []

	# load and preprocess train models
	start = timer()
	for i in range(0, len(paths_for_Blender.train_model_paths)):
		print("\n------------------- TRAIN MODEL: ", (i+1), "---------------------\n")
		start = timer()
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
		if (customNumbering):
			train_model.set_train_labels(filepath = paths_for_Blender.train_model_paths[i])
		else:
			train_model.set_train_labels()

		if (i < len(paths_for_Blender.train_model_paths)-1):
			train_models.append(train_model)
		else:
			validation_models.append(train_model)
			print("\n--------------------- VALIDATION MODEL: ", i, "---------------------\n")

		end = timer()
		print('\ntime to preprocess ', i , ' model : ' + str(end-start))

	if paths_for_Blender.file_type == 'fbx':
		bpy.ops.import_scene.fbx(filepath = paths_for_Blender.test_model_path)
		test_model_names_list = get_model_names()
		test_model = input_model.TestModel_fbx(test_model_names_list[0], test_model_names_list[1], test_model_names_list[2])
	else:
		test_model_names_list = import_obj_files(paths_for_Blender)
		test_model = input_model.TestModel_obj(test_model_names_list)
		#test_model = input_model.TestModel_obj(test_model_names_list)


	print('test model names list = ',  test_model_names_list)
	end = timer()

	time_for_each_step = []
	time_for_each_step.append('\nTime required :')
	time_for_each_step.append('\tfor the preprocessing of train models : ' + str(end - start))

	batch_size = 4096
	epochs = 100
	network = 3
	#networkPrefix = None
	start = timer()
	model = keras.Sequential()
		#keras.layers.Conv1D(64, kernel_size=3, activation=tf.nn.relu, padding='same', input_shape=(None,3)),
		#keras.layers.Conv1D(32, kernel_size=3, activation=tf.nn.relu, padding='same'),
	#model.add( keras.layers.LSTM(number_of_proxy_bones, return_sequences=False, activation=tf.nn.sigmoid, input_shape=(None,3)) )
	if (network == 1):
		# RECURRENT NEURAL NETWORK RNN - LSTM for SEQUENCE CLASSIFICATION
		model.add( keras.layers.LSTM(number_of_proxy_bones, return_sequences=False, activation=tf.nn.sigmoid, input_shape=(None,3)) )
		#model.add( keras.layers.LSTM(number_of_proxy_bones, return_sequences=True, activation=tf.nn.sigmoid) )
		#model.add( keras.layers.LSTM(number_of_proxy_bones, return_sequences=True, activation=tf.nn.sigmoid) )
		#model.add( keras.layers.LSTM(number_of_proxy_bones, return_sequences=False, activation=tf.nn.sigmoid) )
		#model.add( keras.layers.Dense(number_of_proxy_bones, activation=tf.nn.sigmoid) )
		networkPrefix += " "
	elif (network == 2):
		model.add(keras.layers.Conv1D(16, 2, activation='relu', padding='same', input_shape=(None, 3)))
		model.add(keras.layers.Conv1D(16, 2, activation='relu', padding='same'))
		model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))
		model.add(keras.layers.Conv1D(32, 2, activation='relu', padding='same'))
		model.add(keras.layers.Conv1D(32, 2, activation='relu', padding='same'))
		model.add(keras.layers.GlobalMaxPooling1D())
		model.add(keras.layers.Dropout(0.5))
		model.add(keras.layers.Dense(number_of_proxy_bones, activation='sigmoid'))
		networkPrefix += " "
	elif (network == 3):
		# CONVOLUTIONAL NEURAL NETWORK (CNN)
		model.add(keras.layers.Conv1D(8, 2, activation='relu', padding='same', input_shape=(None, 3)))
		model.add(keras.layers.Conv1D(8, 2, activation='relu', padding='same'))
		#best 32 CoNV1D X 2
		model.add(keras.layers.GlobalMaxPooling1D())
		model.add(keras.layers.Dense(number_of_proxy_bones, activation='sigmoid'))
		#model.add( keras.layers.LSTM(number_of_proxy_bones, return_sequences=False, activation=tf.nn.sigmoid) )
		networkPrefix += " "
	elif (network == 4):
		model.add(keras.layers.Conv1D(8, 2, activation='relu', padding='same', input_shape=(None, 3)))
		model.add(keras.layers.Conv1D(8, 2, activation='relu', padding='same'))
		model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))
		model.add( keras.layers.LSTM(number_of_proxy_bones, return_sequences=False, activation=tf.nn.tanh) )
		model.add( keras.layers.Dense(number_of_proxy_bones, activation=tf.nn.sigmoid) )
		networkPrefix += " "

	#top4_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=4)
	#top4_acc.__name__ = 'top4_acc'
	#tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
				  loss='binary_crossentropy',
				  #loss='categorical_crossentropy',
				  #metrics=['accuracy'])
  				  #metrics=['categorical_accuracy'])
				  #metrics=['top_k_categorical_accuracy'])
				  metrics=['binary_accuracy'])
				  #metrics=[top4_acc])
	model.summary()
	# PLOT NETWORK STRUCTURE
	#keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')
	# checkpoint
	#filepath="C:\\Users\\User\\Desktop\\skeleton_training\\scripts\\checkpoints\\weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
	#checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
	#callbacks_list = [checkpoint]

	histories_train = []
	histories_val = []
	for train_model in train_models:
		for i in range(0, len(train_model.mesh_v_in_takes)):
			print("Training train_model: ", train_model.mesh_v_in_takes[i].shape)
			print("Training train_model: ",train_model.labels_for_rest_pose.shape)
			current_train_history = model.fit(train_model.mesh_v_in_takes[i], train_model.labels_for_rest_pose, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.2)
			print('\n')
			histories_train.append([current_train_history.history['loss'], current_train_history.history['binary_accuracy']])
			histories_val.append([current_train_history.history['val_loss'], current_train_history.history['val_binary_accuracy']])
	print("histories_train: \n", histories_train)
	print("histories_val: \n", histories_val)

	train_losses = np.zeros( (len(histories_train[0][0])))
	train_accuracies = np.zeros( (len(histories_train[0][1])))
	print("losses shape: ", train_losses.shape)
	print("accuracies shape: ", train_accuracies.shape)
	val_losses = np.zeros( (len(histories_val[0][0])))
	val_accuracies = np.zeros( (len(histories_val[0][1])))
	for i in range(0, len(histories_train)):
		for j in range(0, len(histories_train[i])):
			for k in range(0, len(histories_train[i][j])):
				if (j == 0): #loss
					train_losses[k] += histories_train[i][j][k]
					val_losses[k] += histories_val[i][j][k]
				else: #accuracy
					train_accuracies[k] += histories_train[i][j][k]
					val_accuracies[k] += histories_val[i][j][k]
	for k in range(0, len(train_losses)):
		train_losses[k] = train_losses[k]/len(histories_train)
		val_losses[k] = val_losses[k]/len(histories_val)
	for k in range(0, len(train_accuracies)):
		train_accuracies[k] = train_accuracies[k]/len(histories_train)
		val_accuracies[k] = val_accuracies[k]/len(histories_val)
	print("TRAIN LOSSES: \n", train_losses)
	print("TRAIN ACCURACIES: \n", train_accuracies)
	print("VALIDATION LOSSES: \n", val_losses)
	print("VALIDATION ACCURACIES: \n", val_accuracies)
	np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'train-losses.out', train_losses, fmt="%.6f")   # X is an array
	np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'train-accuracies.out', train_accuracies, fmt="%.6f")   # X is an array
	np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'val-losses.out', val_losses, fmt="%.6f")   # X is an array
	np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'val-accuracies.out', val_accuracies, fmt="%.6f")   # X is an array
	# copy mesh of models and apply rotation
	# this copied mesh is needed later during animation playback
	test_model.apply_rotation()
	copy_mesh_from_model(test_model)
	test_model_copy_names_list = get_model_names()
	test_model.set_name_of_copied_mesh(test_model_copy_names_list[0], test_model_copy_names_list[1])

	# deselect
	deselect_all()

	end = timer()
	time_for_each_step.append('\tfor training : ' + str(end - start))

	start = timer()
	# set the rest pose of the test model (the pose in the first frame of the animation)
	test_model.set_rest_pose_v()

	# get test set vertices from all frames and set them to the object (test_model)
	test_model.set_mesh_v_in_all_frames()

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
	model.save(os.path.dirname(__file__)+'\\'+networkPrefix+'.h5')
	# Recreate the exact same model purely from the file
	new_model = keras.models.load_model(os.path.dirname(__file__)+'\\'+networkPrefix+'.h5')
	new_model.summary()
	# Check that the state is preserved
	new_predictions = new_model.predict(test_model.mesh_v_for_predictions, batch_size=batch_size)
	np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'new-predictions.out', new_predictions, fmt="%.6f")

	# Note that the optimizer state is preserved as well:
	# you can resume training where you left off.

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
	print("------------- NEW PREDICTIONS --------------\n", new_predictions[0])

	# assign the number of clusters(bones) to the test model
	test_model.num_of_bones = number_of_proxy_bones

	# the vertices that were predicted to belong to zero bones
	# are then assigned to the one with the highest probability
	test_set_labels = assign_unassigned_vertices_to_bone(test_set_labels, predictions)
	test_model.set_labels(test_set_labels)
	print("------------- PREDICTIONS (AFTER) --------------\n", predictions[0])
	print("------------- TEST LABELS (AFTER) --------------\n", test_set_labels[0])

	#clusters_for_pCenter = np.count_nonzero(test_set_labels, axis=1)
	#print("\n------------- CLUSTERS FOR P-CENTER --------------\n", clusters_for_pCenter)
	clusters_for_pCenter = np.count_nonzero(test_set_labels, axis=0)
	print("\n------------- CLUSTERS FOR P-CENTER --------------\n", clusters_for_pCenter)
	print("\n------------- CLUSTERS FOR P-CENTER shape --------------\n", clusters_for_pCenter.shape)
	clusters_for_pCenter = np.count_nonzero(clusters_for_pCenter)
	print("\n------------- CLUSTERS FOR P-CENTER --------------\n", clusters_for_pCenter)

	end = timer()
	time_for_each_step.append('\tto construct test labels : ' + str(end - start))

	# calculate initial vertex weights
	start = timer()
	v_weights, v_weights_indices = weight_calculation.calculate_weights_using_tensorflow_predictions(test_model, predictions)
	test_model.set_v_weights(v_weights, v_weights_indices)
	print("------------- PREDICTIONS - WEIGHTS (AFTER) --------------\n", v_weights[0])
	print("------------- PREDICTIONS - WEIGHTS TEST MODEL (AFTER) --------------\n", test_model.v_weights[0])
	print("------------- PREDICTIONS - WEIGHTS (AFTER) SUM:  --------------\n", np.sum(v_weights[0]))

	#np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'weights.csv', v_weights, fmt="%.6f", delimiter='|')
	#np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'weights-indices.csv', v_weights_indices, fmt="%.6f", delimiter='|')

	end = timer()
	time_for_each_step.append('\tto calculate weights : ' + str(end - start))

	# get transformations produced by affine transformation fitting
	fitting_time = 0
	start = timer()
	transformations = fitting.fit_affine_transformations(test_model)
	#transformations_tensor = fitting.fit_affine_transformations(test_model)

	test_model.set_affine_transformations(transformations)
	#
	approx_v, constr_time = linear_blend_skinning.compute_approx_v_with_LBS(test_model)
	test_model.set_approximated_v(approx_v)
	initial_dE, initial_E_RMS, initial_Avg_Max_Distance_Error = linear_blend_skinning.calculate_errors(test_model)
	print('\tInitial Distortion Percentage, before W_f= ', initial_dE)
	print('\tInitial E_RMS, before W_f = ', initial_E_RMS)
	print('\tInitial Avg_Max_Distance, before W_f = ', initial_Avg_Max_Distance_Error)
	print('\n')
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
		print("------------- FOR LOOP  WEIGHTS (AFTER) --------------\n", new_v_weights[0])
		print("------------- FOR LOOP WEIGHTS TEST MODEL (AFTER) --------------\n", test_model.v_weights[0])
		print("------------- FOR LOOP  WEIGHTS (AFTER) --------------\n", new_v_weights[10])
		print("------------- FOR LOOP WEIGHTS TEST MODEL (AFTER) --------------\n", test_model.v_weights[10])
		print("------------- PREDICTIONS - WEIGHTS (AFTER) SUM 0:  --------------\n", np.sum(new_v_weights[0]))
		print("------------- PREDICTIONS - WEIGHTS (AFTER) SUM 10:  --------------\n", np.sum(new_v_weights[10]))

		end = timer()
		print('---- Time for weight fitting : ' + str(end - start))
		fitting_time += end - start

		start = timer()
		approx_v,constr_time = linear_blend_skinning.compute_approx_v_with_LBS(test_model)
		test_model.set_approximated_v(approx_v)
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
		print('\tAvg_Max_Distance after weight fitting = ', new_Avg_Max_Distance_Error)

		fitting_time += end - start

		# transformation fitting and computation of errors
		start = timer()
		new_transformations = fitting.fit_affine_transformations(test_model)
		#new_transformations_tensor = fitting.fit_affine_transformations(test_model)
		#with sess.as_default():
		#	new_transformations = sess.run(tf.transpose(new_transformations_tensor))

		#print("\n\nCURRENT transformations", new_transformations.shape)
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

		fitting_time += end - start

	time_for_each_step.append('\tfor fitting : ' + str(fitting_time))

	# set the combination of weights and transformations that produced the best result during the fitting steps
	index_of_min_error = old_errors.index(min(old_errors))
	if index_of_min_error % 2 == 1:
		test_model.set_affine_transformations(old_w_t[index_of_min_error])
		test_model.set_v_weights(old_w_t[index_of_min_error - 1], old_v_weights_indices[index_of_min_error // 2])
	else:
		test_model.set_v_weights(old_w_t[index_of_min_error], old_v_weights_indices[index_of_min_error // 2])
		test_model.set_affine_transformations(old_w_t[index_of_min_error - 1])

	# linear blend skinning
	start = timer()
	approx_v,constr_time = linear_blend_skinning.compute_approx_v_with_LBS(test_model)
	test_model.set_approximated_v(approx_v)
	end = timer()
	time_for_each_step.append('\tfor last linear blend skinning : ' + str(end - start))

	# compute error produced by linear blend skinning
	start = timer()
	dE, E_RMS, Avg_Max_Distance_Error = linear_blend_skinning.calculate_errors(test_model)
	end = timer()
	time_for_each_step.append('\tto calculate errors : ' + str(end - start))
	#np.savetxt(os.path.dirname(__file__)+'\\'+networkPrefix+'Time.out', time_for_each_step)
	for step in time_for_each_step:
		print(step)
	# construct array with approximated poses and init variables for the animation
	start = timer()
	approximated_animation_sequence.construct_array_with_approx_poses(test_model)
	approximated_animation_sequence.init_approx_animation_variables()
	end = timer()
	time_for_each_step.append('\tto construct array with approx poses and init variables for animation: ' + str(end - start))

	print('Error metrics :')
	print('\tDistortion Percentage = ', dE)
	print('\tE_RMS = ', E_RMS)
	print('\tAvg_Max_Distance_Error = ',  Avg_Max_Distance_Error)

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
