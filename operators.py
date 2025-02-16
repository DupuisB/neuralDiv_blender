import bpy
import os
import json
import torch
import bmesh
import tempfile
import logging

from .include import *   # ...existing code...
from .models import *    # ...existing code...

NETPARAMS = 'netparams.dat'
NETWORK_PATH = os.path.join(os.path.dirname(__file__), "data", "jobs", "net_cartoon_elephant")
HYPERPARAMS_FILE = os.path.join(NETWORK_PATH, "hyperparameters.json")

addon_keymaps = []

def write_obj(filepath, mesh):
    # Minimal OBJ exporter for a Blender mesh.
    with open(filepath, 'w') as f:
        for v in mesh.vertices:
            f.write("v {} {} {}\n".format(v.co.x, v.co.y, v.co.z))
        for poly in mesh.polygons:
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

        # ...prepare the mesh...
        mesh = obj.data.copy()
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.free()

        try:
            temp_dir = tempfile.gettempdir()
            temp_obj_path = os.path.join(temp_dir, obj.name + "_temp.obj")
            write_obj(temp_obj_path, mesh)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to write temporary OBJ: {e}")
            return {'CANCELLED'}

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

        try:
            T = TestMeshes([temp_obj_path], params['numSubd'])
            T.computeParameters()
            T.toDevice(params["device"])
        except Exception as e:
            self.report({'ERROR'}, f"TestMeshes failed: {e}")
            return {'CANCELLED'}

        mIdx = 0
        try:
            x_input = T.getInputData(mIdx)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to get input data: {e}")
            return {'CANCELLED'}

        hfList = T.hfList
        poolMats = T.poolMats
        dofs = T.dofs

        try:
            net = SubdNet(params).to(device)
            state_dict = torch.load(os.path.join(params['output_path'], NETPARAMS),
                                    map_location=torch.device(device))
            net.load_state_dict(state_dict)
            net.eval()
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load network: {e}")
            return {'CANCELLED'}

        try:
            with torch.no_grad():
                outputs = net(x_input, mIdx, hfList, poolMats, dofs)
        except Exception as e:
            self.report({'ERROR'}, f"Network forward pass failed: {e}")
            return {'CANCELLED'}

        try:
            level = len(outputs) - 1
            out = outputs[level].cpu().squeeze(0)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to process network output: {e}")
            return {'CANCELLED'}

        try:
            faces_tensor = T.meshes[mIdx][level].F.cpu()
            faces = faces_tensor.tolist()
        except Exception as e:
            self.report({'ERROR'}, f"Failed to retrieve face connectivity: {e}")
            return {'CANCELLED'}

        try:
            obj.data.clear_geometry()
            vertices_out = out.tolist()
            obj.data.from_pydata(vertices_out, [], faces)
            obj.data.update()
        except Exception as e:
            self.report({'ERROR'}, f"Failed to update mesh: {e}")
            return {'CANCELLED'}

        try:
            os.remove(temp_obj_path)
        except Exception as e:
            self.report({'WARNING'}, f"Could not remove temporary file: {e}")

        self.report({'INFO'}, "Neural subdivision complete.")
        return {'FINISHED'}

def register_operator():
    bpy.utils.register_class(OBJECT_OT_NN_Subdivide)
    wm = bpy.context.window_manager
    km = wm.keyconfigs.addon.keymaps.new(name='Object Mode', space_type='EMPTY')
    kmi = km.keymap_items.new(OBJECT_OT_NN_Subdivide.bl_idname, type='N', value='PRESS', shift=True)
    addon_keymaps.append(km)

def unregister_operator():
    bpy.utils.unregister_class(OBJECT_OT_NN_Subdivide)
    wm = bpy.context.window_manager
    for km in addon_keymaps:
        wm.keyconfigs.addon.keymaps.remove(km)
    addon_keymaps.clear()
