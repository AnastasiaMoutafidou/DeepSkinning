import bpy
from random import *

approximated_poses = []

def construct_array_with_approx_poses(test_model):
    for obj in bpy.context.selected_objects:
        obj.select = True
    test_model.clear_parent()
    bpy.data.scenes[0].frame_end = test_model.num_of_frames

    for i in range(0, test_model.num_of_frames):
        names_of_duplicates = []
        if i != 0:
            bpy.ops.object.duplicate()

        name_of_duplicate_obj_mesh = bpy.context.object.name
        name_of_duplicate_mesh = bpy.context.object.data.name
        names_of_duplicates.append(name_of_duplicate_obj_mesh)
        names_of_duplicates.append(name_of_duplicate_mesh)
        approximated_poses.append(names_of_duplicates)

        for j in range(0, test_model.num_of_vertices):
            bpy.data.meshes[name_of_duplicate_mesh].vertices[j].co = test_model.approximated_v[i][j]

            if bpy.data.objects[test_model.name_of_obj_mesh].parent:
                bpy.data.meshes[name_of_duplicate_mesh].vertices[j].co -= bpy.data.objects[test_model.name_of_obj_mesh].parent.location

        destfolder = "C:\\Users\\User\\Desktop\\model-animal\\"
        name = name_of_duplicate_obj_mesh
        bpy.ops.export_scene.obj(filepath = destfolder + name + ".obj", use_selection = True)
        bpy.data.objects[name_of_duplicate_obj_mesh].hide = True

def init_approx_animation_variables():
    bpy.app.handlers.frame_change_post.append(my_handler)

def my_handler(scene):
    current_pose = approximated_poses[scene.frame_current-1][0]
    if scene.frame_current == 1:
        bpy.data.objects[current_pose].hide = False

        previous_pose = approximated_poses[-1][0]
        bpy.data.objects[previous_pose].hide = True
    else:
        bpy.data.objects[current_pose].hide = False

        previous_pose = approximated_poses[scene.frame_current-2][0]
        bpy.data.objects[previous_pose].hide = True

        if scene.frame_current != scene.frame_end:
            next_pose = approximated_poses[scene.frame_current][0]
            bpy.data.objects[next_pose].hide = True
