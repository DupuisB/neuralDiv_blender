import bpy
import os
import sys
import json
import torch
import bmesh
import logging
import tempfile

from .include import *   # Make sure TestMeshes and any helper functions are available
from .models import *

NETPARAMS = 'netparams.dat'
# Adjust NETWORK_PATH relative to this file
NETWORK_PATH = os.path.join(os.path.dirname(__file__), "data", "jobs", "net_cartoon_elephant")
HYPERPARAMS_FILE = os.path.join(NETWORK_PATH, "hyperparameters.json")

bl_info = {
    "name": "Neural 000",
    "description": "Neural Network Subdivision",
    "author": "Authors name",
    "version": (0, 0, 1),
    "blender": (2, 9, 0),
    "location": "View3D",
    "warning": "This addon is still in development.",
    "wiki_url": "",
    "category": "Object"
}

addon_keymaps = []

def write_obj(filepath, mesh):
    """
    Minimal OBJ exporter for a Blender mesh.
    Assumes the mesh is triangulated.
    """
    with open(filepath, 'w') as f:
        # Write vertices
        for v in mesh.vertices:
            f.write("v {} {} {}\n".format(v.co.x, v.co.y, v.co.z))
        # Write faces (OBJ indices are 1-indexed)
        for poly in mesh.polygons:
            # poly.vertices is a tuple of vertex indices
            face_line = "f " + " ".join(str(idx + 1) for idx in poly.vertices) + "\n"
            f.write(face_line)

class OBJECT_OT_NN_Subdivide(bpy.types.Operator):
    bl_idname = "object.nn_subdivide"
    bl_label = "Neural Network Subdivide"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object.")
            return {'CANCELLED'}

        # --- Prepare the mesh: copy and triangulate ---
        mesh = obj.data.copy()
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.free()

        # --- Export the Blender mesh as a temporary OBJ file ---
        try:
            temp_dir = tempfile.gettempdir()
            temp_obj_path = os.path.join(temp_dir, obj.name + "_temp.obj")
            write_obj(temp_obj_path, mesh)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to write temporary OBJ: {e}")
            return {'CANCELLED'}

        # --- Load network hyperparameters ---
        try:
            with open(HYPERPARAMS_FILE, 'r') as f:
                params = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load hyperparameters: {e}")
            return {'CANCELLED'}
        params['numSubd'] = 2
        params['output_path'] = NETWORK_PATH

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        params['device'] = device

        # --- Create a TestMeshes instance from the temporary OBJ ---
        try:
            # TestMeshes expects a list of file paths.
            T = TestMeshes([temp_obj_path], params['numSubd'])
            T.computeParameters()
            T.toDevice(params["device"])
        except Exception as e:
            self.report({'ERROR'}, f"TestMeshes failed: {e}")
            return {'CANCELLED'}

        # Get network input data from TestMeshes
        mIdx = 0
        try:
            x_input = T.getInputData(mIdx)  # Should be of shape [1, N, features]
        except Exception as e:
            self.report({'ERROR'}, f"Failed to get input data from TestMeshes: {e}")
            return {'CANCELLED'}

        # Extract additional parameters computed by TestMeshes
        hfList = T.hfList       # half flap list
        poolMats = T.poolMats   # pooling matrices
        dofs = T.dofs           # degrees of freedom (or any extra parameters)

        # --- Initialize and load the network ---
        try:
            net = SubdNet(params).to(device)
            state_dict = torch.load(os.path.join(params['output_path'], NETPARAMS),
                                    map_location=torch.device(device))
            net.load_state_dict(state_dict)
            net.eval()
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load network: {e}")
            return {'CANCELLED'}

        # --- Run the network ---
        try:
            with torch.no_grad():
                outputs = net(x_input, mIdx, hfList, poolMats, dofs)
        except Exception as e:
            self.report({'ERROR'}, f"Network forward pass failed: {e}")
            return {'CANCELLED'}

        # --- Process network output ---
        # Assume outputs is a list (one for each subdivision level).
        # We choose the last level (highest resolution) as our result.
        try:
            level = len(outputs) - 1
            out = outputs[level].cpu().squeeze(0)  # shape: [num_vertices, 3]
        except Exception as e:
            self.report({'ERROR'}, f"Failed to process network output: {e}")
            return {'CANCELLED'}

        # Retrieve face connectivity from TestMeshes.
        try:
            # T.meshes is assumed to be a list per input mesh; each element is a list over levels.
            faces_tensor = T.meshes[mIdx][level].F.cpu()
            faces = faces_tensor.tolist()
        except Exception as e:
            self.report({'ERROR'}, f"Failed to retrieve face connectivity: {e}")
            return {'CANCELLED'}

        # --- Create a new Blender mesh from the network output ---
        try:
            new_mesh = bpy.data.meshes.new(obj.name + "_subdiv")
            vertices_out = out.tolist()  # each element is [x, y, z]
            new_mesh.from_pydata(vertices_out, [], faces)
            new_mesh.update()

            new_obj = bpy.data.objects.new(obj.name + "_subdiv", new_mesh)
            context.collection.objects.link(new_obj)
            context.view_layer.objects.active = new_obj
            new_obj.select_set(True)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create new mesh: {e}")
            return {'CANCELLED'}

        # --- Clean up temporary file ---
        try:
            os.remove(temp_obj_path)
        except Exception as e:
            self.report({'WARNING'}, f"Could not remove temporary file: {e}")

        self.report({'INFO'}, "Neural subdivision complete.")
        return {'FINISHED'}

def menu_func(self, context):
    self.layout.operator(OBJECT_OT_NN_Subdivide.bl_idname)

def register():
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading Neural 000")
    logging.info("Neural 000: Registering Neural Network Subdivide operator")
    bpy.utils.register_class(OBJECT_OT_NN_Subdivide)
    bpy.types.VIEW3D_MT_object.append(menu_func)
    wm = bpy.context.window_manager
    km = wm.keyconfigs.addon.keymaps.new(name='Object Mode', space_type='EMPTY')
    kmi = km.keymap_items.new(OBJECT_OT_NN_Subdivide.bl_idname, type='N', value='PRESS', shift=True)
    addon_keymaps.append(km)

def unregister():
    bpy.utils.unregister_class(OBJECT_OT_NN_Subdivide)
    bpy.types.VIEW3D_MT_object.remove(menu_func)
    wm = bpy.context.window_manager
    for km in addon_keymaps:
        wm.keyconfigs.addon.keymaps.remove(km)
    addon_keymaps.clear()

if __name__ == "__main__":
    register()
