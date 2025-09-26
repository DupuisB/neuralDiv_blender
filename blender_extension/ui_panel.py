import os
import bpy
from bpy.props import EnumProperty, FloatProperty

def get_network_options(self, context):
    jobs_path = os.path.join(os.path.dirname(__file__), "data", "jobs")
    try:
        dirs = sorted([d for d in os.listdir(jobs_path) if os.path.isdir(os.path.join(jobs_path, d))])
        if "net_cartoon_elephant" in dirs:
            dirs.remove("net_cartoon_elephant")
            dirs.insert(0, "net_cartoon_elephant")
        items = [(d, d, "") for d in dirs]
        return items if items else [("None", "None", "")]
    except Exception:
        return [("None", "None", "")]

class OBJECT_PT_NetworkTab(bpy.types.Panel):
    bl_label = "Neural Networks"
    bl_idname = "OBJECT_PT_network_tab"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Neural Networks"

    def draw(self, context):
        layout = self.layout
        sc = context.scene
        
        # Dropdown for selecting the network
        layout.prop(sc, "network")
        layout.prop(sc, "example_param")

        # Button to run the neural network subdivision
        layout.operator("object.nn_subdivide", text="Run Subdivision", icon="MOD_SUBSURF")
        # New button to add the default loop subdivision (subsurf) modifier
        op = layout.operator("object.modifier_add", text="Run Loop Subdivision", icon="MOD_SUBSURF")
        op.type = 'SUBSURF'

def register_panel():
    bpy.types.Scene.network = EnumProperty(
        name="Network",
        description="Select a neural network to use for subdivision",
        items=get_network_options,
        default=0  # Default index
    )
    bpy.utils.register_class(OBJECT_PT_NetworkTab)

def unregister_panel():
    bpy.utils.unregister_class(OBJECT_PT_NetworkTab)
    del bpy.types.Scene.network
    del bpy.types.Scene.example_param
