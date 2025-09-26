import bpy
import logging

from .operators import register_operator, unregister_operator
from .ui_panel import register_panel, unregister_panel

bl_info = {
    "name": "Neural 000",
    "description": "Neural Network Subdivision",
    "author": "Authors name",
    "version": (0, 0, 1),
    "blender": (4, 2, 0),
    "location": "View3D",
    "warning": "This addon is still in development.",
    "wiki_url": "",
    "category": "Object"
}

def register():
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading Neural 000")
    logging.info("Registering Network Subdivision features")
    register_operator()
    register_panel()

def unregister():
    try:
        unregister_operator()
        unregister_panel()
    except Exception as e:
        logging.warning("Error during unregister: %s", e) 

if __name__ == "__main__":
    register()
