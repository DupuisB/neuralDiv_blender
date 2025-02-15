import os
import glob
import toml

def update_manifest():
    manifest_path = "blender_manifest.toml"
    wheels_dir = "./wheels"  # Updated path to one folder higher

    # Find all wheel files in the wheels directory and ensure forward slashes
    wheel_files = [
        os.path.join(wheels_dir, os.path.basename(w)).replace(os.sep, "/")
        for w in glob.glob(os.path.join(wheels_dir, "*.whl"))
    ]

    # Load existing manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = toml.load(f)

    # Update wheels array
    manifest["wheels"] = wheel_files

    # Write back the updated manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        toml.dump(manifest, f)

    print("Manifest updated with wheels:")
    for wheel in wheel_files:
        print(" -", wheel)

if __name__ == "__main__":
    update_manifest()