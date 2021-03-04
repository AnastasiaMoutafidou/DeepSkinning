bl_info = {
    "name": "my custom buttons",
    "author": "Evangelos Tsakanikas",
    "location": "View3D > Tools > My Buttons",
    "description": "some of my custom buttons",
    "category": "My Buttons",
    }

import bpy

class ReloadStartupFile(bpy.types.Operator):
    bl_idname = 'my.reload_startup_file'
    bl_label = 'Reload Startup File'
    bl_options = {"REGISTER", "UNDO"}
    
    def execute(self, context):
        bpy.ops.wm.read_homefile()
        return {"FINISHED"}

class HideImportedModels(bpy.types.Operator):
    bl_idname = 'my.hide_imported_models'
    bl_label = 'Hide Imported Models'
    bl_options = {"REGISTER", "UNDO"}
    
    def execute(self, context):
        for obj in bpy.data.objects:
            if obj.parent:
                obj.hide = not obj.hide
                obj.parent.hide = not obj.parent.hide
        return {"FINISHED"}

class UnlinkApproxObjects(bpy.types.Operator):
    bl_idname = 'my.unlink_approx_objects'
    bl_label = 'Unlink Approx Objects'
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                if not obj.parent:
                    if obj.name in bpy.data.scenes[0].objects:
                        bpy.data.scenes[0].objects.unlink(obj)
        return {"FINISHED"}

class LinkApproxObjects(bpy.types.Operator):
    bl_idname = 'my.link_approx_objects'
    bl_label = 'Link Approx Objects'
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                if not obj.parent:
                    if obj.name not in bpy.data.scenes[0].objects:
                        bpy.data.scenes[0].objects.link(obj)
        return {"FINISHED"}

class MyButtonsPanel(bpy.types.Panel):
    """Creates a Panel in the Tools window"""
    bl_label = "My Buttons Panel"
    bl_idname = "My Buttons"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "My Buttons"
    
    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.operator("my.reload_startup_file", text='Reload Startup File', icon='WORLD_DATA')
        row = layout.row()
        row.operator("my.hide_imported_models", text='Hide/Show imp objs', icon='WORLD_DATA')
        row = layout.row()
        row.operator("my.unlink_approx_objects", text='Unlink approx objs', icon='WORLD_DATA')
        row = layout.row()
        row.operator("my.link_approx_objects", text='Link approx objs', icon='WORLD_DATA')
        row = layout.row()
        row.prop(bpy.data.scenes[0].render, 'fps', slider=True)


def register():
    bpy.utils.register_class(MyButtonsPanel)
    bpy.utils.register_class(ReloadStartupFile)
    bpy.utils.register_class(HideImportedModels)
    bpy.utils.register_class(UnlinkApproxObjects)
    bpy.utils.register_class(LinkApproxObjects)

def unregister():
    bpy.utils.unregister_class(MyButtonsPanel)
    bpy.utils.unregister_class(ReloadStartupFile)
    bpy.utils.unregister_class(HideImportedModels)
    bpy.utils.register_class(UnlinkApproxObjects)
    bpy.utils.register_class(LinkApproxObjects)

if __name__ == "__main__":
    register()
