import bpy
import mathutils
import numpy as np
import helper

class Model(object):
	def __init__(self, name_of_obj_mesh, name_of_mesh, name_of_armature):
		self.name_of_obj_mesh = name_of_obj_mesh
		self.name_of_mesh = name_of_mesh
		self.name_of_armature = name_of_armature
		self.num_of_vertices = len(bpy.data.meshes[name_of_mesh].vertices)
		self.change_mesh_to_World_coordinates()

	def change_mesh_to_World_coordinates(self):
		obj = bpy.data.objects[self.name_of_obj_mesh]
		mat = obj.matrix_world
		mesh = bpy.data.meshes[self.name_of_mesh]
		mesh.transform(mat)
		obj.matrix_world = mathutils.Matrix()

	def get_mesh_v_in_all_frames(self):
		# get mesh vertices per frame
		scn = bpy.context.scene
		mesh_obj = bpy.data.objects[self.name_of_obj_mesh]
		start_frame, end_frame = self.__get_active_action_frame_end()

		vertices = np.array([v.co for v in bpy.data.meshes[self.name_of_mesh].vertices])
		for f in range(start_frame, end_frame + 1):
			bpy.context.scene.frame_set(f)
			temp_mesh = mesh_obj.to_mesh(scn, True, 'PREVIEW')

			if f == 1:
				vertices = np.array([v.co for v in temp_mesh.vertices])
			else:
				vertices = np.vstack( (vertices, [v.co for v in temp_mesh.vertices]) )

			bpy.data.meshes.remove(temp_mesh)
		return vertices

	def get_mesh_v_of_whole_animation_3D(self):
		# get mesh vertices per frame
		scn = bpy.context.scene
		mesh_obj = bpy.data.objects[self.name_of_obj_mesh]
		start_frame, end_frame = self.__get_active_action_frame_end()

		vertices = []
		for v in range(0, self.num_of_vertices):
			vertices.append([])

		for f in range(start_frame, end_frame + 1):
			bpy.context.scene.frame_set(f)
			temp_mesh = mesh_obj.to_mesh(scn, True, 'PREVIEW')

			for v in range(0, self.num_of_vertices):
				vertices[v].append(np.array(temp_mesh.vertices[v].co))

			bpy.data.meshes.remove(temp_mesh)
		return np.array(vertices)

	def __get_active_action_frame_end(self):
		action_name = self.__get_active_action_name()
		frame_end = bpy.data.actions[action_name].frame_range
		self.num_of_frames = int(frame_end[1])
		return int(frame_end[0]), int(frame_end[1])

	def __get_active_action_name(self):
		action_name = bpy.data.objects[self.name_of_armature].animation_data.action.name
		return action_name

class TrainModel(Model):
	def __init__(self, name_of_obj_mesh, name_of_mesh, name_of_armature):
		super().__init__(name_of_obj_mesh, name_of_mesh, name_of_armature)

	def set_v_groups_names(self):
		self.v_groups_names = self.__get_v_groups_names()

	# get vertex group names (meaning bone names)
	def __get_v_groups_names(self):
		mesh_obj = bpy.data.objects[self.name_of_obj_mesh]
		v_groups_lookup = {g.index: g.name for g in mesh_obj.vertex_groups}
		return v_groups_lookup

	def set_v_groups(self):
		self.v_groups, self.v_groups_weights = self.__get_v_groups()
		self.num_of_bones = len(self.v_groups)

	def __get_v_groups(self):
		mesh_obj = bpy.data.objects[self.name_of_obj_mesh]
		print('vertex groups lookup : ' , self.v_groups_names)
		# array which contains the indices of the vertices in every vertex group
		v_groups = [[] for Bone_name in self.v_groups_names.values()]
		v_groups_weights = [[] for Bone_name in self.v_groups_names.values()]
		for v in mesh_obj.data.vertices:
			for g in v.groups:
				v_groups[g.group].append(v.index)
				v_groups_weights[g.group].append(g.weight)
		return v_groups, v_groups_weights

	def set_mesh_v_in_all_frames(self):
		self.mesh_v_in_takes = []
		for take in range(0, len(bpy.data.actions)):
			bpy.data.objects[self.name_of_armature].animation_data.action = bpy.data.actions[take]

			self.mesh_v_in_takes.append(self.get_mesh_v_of_whole_animation_3D())

	def set_train_labels(self, filepath = None):
		self.labels_for_rest_pose = self.__construct_labels_for_all_frames(filepath = filepath)

	def __construct_labels_for_all_frames(self, filepath = None):
		# construct array which contains 1 if the vertex belongs to x vertex group
		labels = np.zeros( (self.num_of_vertices, self.number_of_proxy_bones) )
		indices = None
		newIndices = None
		Path = ""
		if filepath != None:
			splittedPath = filepath.split(".fbx")
			Path = splittedPath[0]
			Path += "_bones.txt"
			indices = np.loadtxt(Path)

			newIndices = np.zeros_like(indices, dtype=int)
			print("Indices Size: ", len(newIndices), "Vertex Groups Size: ", len(self.v_groups))
			for i in range(0, len(indices)):
				newIndices[i] = int(indices[i])
				newIndices[i] = helper.mapRange(newIndices[i], 1, 212, 0, 120)
				print("newIndices: ", newIndices[i], " i: ", i)

		for i in range(0, self.num_of_vertices):
			for j in range(0, self.num_of_bones ):
				if i in self.v_groups[j]:
					if (filepath == None):
						labels[i][j] = 1 #self.v_groups_weights[j][self.v_groups[j].index(i)]
					else:
						labels[i][int(newIndices[j])] = 1 #self.v_groups_weights[j][self.v_groups[j].index(i)]


		labels_for_RestPose_vertices = np.copy(labels)
		return labels_for_RestPose_vertices


class TestModel(Model):
	def __init__(self, name_of_obj_mesh, name_of_mesh, name_of_armature):
		super().__init__(name_of_obj_mesh, name_of_mesh, name_of_armature)

	def set_name_of_copied_mesh(self, name_of_copied_obj_mesh, name_of_copied_mesh):
		self.name_of_copied_obj_mesh = name_of_copied_obj_mesh
		self.name_of_copied_mesh = name_of_copied_mesh

	def clear_parent(self):
		bpy.context.scene.objects[self.name_of_copied_obj_mesh].select = True
		bpy.context.scene.objects.active = bpy.data.objects[self.name_of_copied_obj_mesh]
		bpy.ops.object.parent_clear(type='CLEAR')

	def set_edit_mode_rest_pose_v(self):
		edit_mode_rest_pose = bpy.data.objects[self.name_of_copied_obj_mesh]
		self.edit_mode_mesh_v_in_rest_pose = np.array([v.co for v in edit_mode_rest_pose.data.vertices])

	def set_labels(self, test_set_labels):
		self.labels = test_set_labels

	def set_v_weights(self, v_weights, v_weights_indices):
		self.v_weights = v_weights
		self.v_weights_indices = v_weights_indices

	def set_affine_transformations(self, affine_transformations):
		self.affine_transformations = affine_transformations

	def set_approximated_v(self, approximated_v):
		self.approximated_v = approximated_v

	def get_approximated_v(self):
		return self.approximated_v

	def get_v_weights(self):
		return self.v_weights

class TestModel_fbx(TestModel):
	def __init__(self, name_of_obj_mesh, name_of_mesh, name_of_armature):
		super().__init__(name_of_obj_mesh, name_of_mesh, name_of_armature)

	def apply_rotation(self):
		bpy.context.scene.objects[self.name_of_armature].select = True
		bpy.context.scene.objects.active = bpy.data.objects[self.name_of_armature]
		bpy.ops.object.transform_apply(rotation=True)

	def set_rest_pose_v(self):
		self.mesh_v_in_rest_pose = self.__get_rest_pose_v()

	def __get_rest_pose_v(self):
		scn = bpy.context.scene
		mesh_obj = bpy.data.objects[self.name_of_obj_mesh]
		bpy.context.scene.frame_set(1)
		temp_mesh = mesh_obj.to_mesh(scn, True, 'PREVIEW')
		rest_pose_mesh = np.array([v.co for v in temp_mesh.vertices])
		bpy.data.meshes.remove(temp_mesh)
		return rest_pose_mesh

	def get_rest_pose_v(self):
		scn = bpy.context.scene
		mesh_obj = bpy.data.objects[self.name_of_obj_mesh]
		print('name of obj mesh in __get_rest_pose_v = ', self.name_of_obj_mesh)
		bpy.context.scene.frame_set(1)
		temp_mesh = mesh_obj.to_mesh(scn, True, 'PREVIEW')
		rest_pose_mesh = np.array([v.co for v in temp_mesh.vertices])
		return rest_pose_mesh

	def get_pose_v(self, frame):
		scn = bpy.context.scene
		mesh_obj = bpy.data.objects[self.name_of_obj_mesh]
		print('name of obj mesh in __get_rest_pose_v = ', self.name_of_obj_mesh)
		bpy.context.scene.frame_set(frame)
		temp_mesh = mesh_obj.to_mesh(scn, True, 'PREVIEW')
		rest_pose_mesh = np.array([v.co for v in temp_mesh.vertices])
		return rest_pose_mesh

	def set_mesh_v_in_all_frames(self):
		self.mesh_v_in_all_frames = self.get_mesh_v_in_all_frames()
		self.mesh_v_for_predictions = self.get_mesh_v_of_whole_animation_3D()

class TestModel_obj(TestModel):
	def __init__(self, names_of_obj_meshes):
		self.names_of_obj_meshes = names_of_obj_meshes
		self.num_of_frames = len(names_of_obj_meshes)
		self.name_of_obj_mesh = self.names_of_obj_meshes[0]
		for name in self.names_of_obj_meshes:
			if 'reference' in name:
				self.name_of_obj_mesh = name
				self.name_of_mesh = name
				bpy.data.objects[self.name_of_obj_mesh].hide = False

	def apply_rotation(self):
		for name in self.names_of_obj_meshes:
			self.__change_mesh_to_World_coordinates(name)

	def __change_mesh_to_World_coordinates(self, name):
		obj = bpy.data.objects[name]
		mat = obj.matrix_world
		mesh = bpy.data.meshes[name]
		mesh.transform(mat)
		obj.matrix_world = mathutils.Matrix()

	def set_rest_pose_v(self):
		self.mesh_v_in_rest_pose = self.__get_rest_pose_v()
		self.num_of_vertices = len(self.mesh_v_in_rest_pose)

	def __get_rest_pose_v(self):
		scn = bpy.context.scene
		mesh_obj = bpy.data.objects[self.name_of_obj_mesh]
		print('name of obj mesh in __get_rest_pose_v = ', self.name_of_obj_mesh)
		bpy.context.scene.frame_set(1)
		temp_mesh = mesh_obj.to_mesh(scn, True, 'PREVIEW')
		rest_pose_mesh = np.array([v.co for v in temp_mesh.vertices])
		bpy.data.meshes.remove(temp_mesh)
		return rest_pose_mesh

	def get_rest_pose_v(self):
		scn = bpy.context.scene
		mesh_obj = bpy.data.objects[self.name_of_obj_mesh]
		print('name of obj mesh in __get_rest_pose_v = ', self.name_of_obj_mesh)
		bpy.context.scene.frame_set(1)
		temp_mesh = mesh_obj.to_mesh(scn, True, 'PREVIEW')
		rest_pose_mesh = np.array([v.co for v in temp_mesh.vertices])
		return rest_pose_mesh

	def set_mesh_v_in_all_frames(self):
		self.mesh_v_in_all_frames = self.__get_mesh_v_in_all_frames()
		self.mesh_v_for_predictions = self.__get_mesh_v_of_whole_animation_3D()

	def __get_mesh_v_in_all_frames(self):
		# get mesh vertices per frame
		scn = bpy.context.scene
		vertices = np.array([])
		for f in range(1, self.num_of_frames + 1):
			mesh_obj = bpy.data.objects[self.names_of_obj_meshes[f-1]]
			temp_mesh = mesh_obj.to_mesh(scn, True, 'PREVIEW')
			if f == 1:
				vertices = np.array([v.co for v in temp_mesh.vertices])
			else:
				vertices = np.vstack( (vertices, [v.co for v in temp_mesh.vertices]) )
			bpy.data.meshes.remove(temp_mesh)
		return vertices

	def __get_mesh_v_of_whole_animation_3D(self):
		# get mesh vertices per frame
		scn = bpy.context.scene

		vertices = []
		for v in range(0, self.num_of_vertices):
			vertices.append([])

		for f in range(0, self.num_of_frames):
			mesh_obj = bpy.data.objects[self.names_of_obj_meshes[f]]
			temp_mesh = mesh_obj.to_mesh(scn, True, 'PREVIEW')

			for v in range(0, self.num_of_vertices):
				vertices[v].append(np.array(temp_mesh.vertices[v].co))

			bpy.data.meshes.remove(temp_mesh)
		return np.array(vertices)
	
	def get_mesh_v_in_all_frames_Array(self):
		return self.mesh_v_in_all_frames

	def get_mesh_v_in_all_frames_Array3D(self):
		return self.mesh_v_for_predictions
