#!/bin/bash
# Install dependencies and generate requirements file using pip freeze

poetry install

poetry export -f requirements.txt --output requirements.txt

rm -rf ./wheels
mkdir ./wheels
pip download -r requirements.txt -d ./wheels
python3 ./update_wheels.py # Update the wheels in blender manifest

blender --command extension build