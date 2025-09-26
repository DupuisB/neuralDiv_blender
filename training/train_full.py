import json
import os
import sys
import time
import pickle
import subprocess
import numpy as np
import torch

from include import *
from models import *

import time
import os
    
def main():
    # load configuration from JSON file
    config_file = 'train_hyperparameters.json'
    if not os.path.exists(config_file):
        print("Config file not found:", config_file)
        sys.exit(1)
    with open(config_file, 'r') as f:
        config = json.load(f)
        
    output_path = config['output_path']
    
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    # Create nbSubd folders for the output meshes
    if not os.path.exists(os.path.join(output_path, "dataset")):
        os.mkdir(os.path.join(output_path, "dataset"))
        
    def generate_dataset():
        """
        Runs the external executable to generate training data.
        Assumes random_subdiv_remesh_bin.exe is in the current directory.
        ./random_subdiv_remesh_bin.exe [mesh_path] [target_faces] [number_subdivision] [random_seed] [output_path]
        As an output, it generates 
        """
        exe_path = os.path.join('.', 'random_subdiv_remesh_bin.exe')
        
        if not os.path.exists(exe_path):
            print("Executable not found:", exe_path)
            sys.exit(1)
            
        # Call the exe with the meshes folder as argument
        # Loop over all objects in the folder
        numObjet = 0
        
        for obj in os.listdir(config["mesh_folder"]):
            obj_path = os.path.join(config["mesh_folder"], obj)
            if os.path.isfile(obj_path) and obj.endswith('.obj'):
                numObjet += 1
                # Create a folder for the object
                os.mkdir(os.path.join(output_path, "dataset", f"mesh{numObjet}"))
                for j in range(config["numSubd"]+1):
                    os.mkdir(os.path.join(output_path, "dataset", f"mesh{numObjet}", f"subd{j}"))
                
                print("Processing", obj_path)
                
                for k in range(config["numMeshes"]): # Generate multiple meshes for each input mesh
                    subprocess.run([exe_path, obj_path, str(config["target_faces"]), str(config["numSubd"]), f"{k}", f"{output_path}"])

                    # We now have numSubd+1 meshes in the same folder, with names like obj_subd0.obj, obj_subd1.obj, etc.
                    # Let's put each one in a subfolder named subd0, subd1, etc. Let's name them 00i.obj
                    for j in range(config["numSubd"]+1):
                        new_path = os.path.join(output_path, "dataset", f"mesh{numObjet}", f"subd{j}", f"{k+1:03d}.obj")
                        os.rename(os.path.join(output_path, f"output_s{j}.obj"), new_path)
                
        print("Done generating dataset")

    def convert_to_pkl(delete = False):
        """
        Converts the meshes in mesh_folder into a .pkl file using TrainMeshes.
        """
        # In our previous gendataPKL.py, mesh_folders was a list.
        # Identify all folders under dataset
        folders = [output_path + "dataset/" + f + '/' for f in os.listdir(os.path.join(output_path, "dataset"))]
        S = TrainMeshes(folders)
        with open(os.path.join(output_path, "dataset.pkl"), "wb") as f:
            pickle.dump(S, f)
        if delete:
            os.remove(os.path.join(output_path, "dataset"))
        print(f"Saved dataset.pkl with {S.nM} meshes")

    def train_model(params):
        """
        Loads training and validation data and trains the network.
        """
        NETPARAMS = 'netparams.dat'
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load training data
        with open(output_path + "dataset.pkl", "rb") as f:
            S = pickle.load(f)
        S.computeParameters()
        S.toDevice(device)
        
        # Split dataset indices: 80% training, 20% validation
        n_total = S.nM
        n_train = int(0.8 * n_total)
        train_indices = list(range(n_train))
        valid_indices = list(range(n_train, n_total))
        
        # Write hyperparameters to output folder
        with open(os.path.join(params['output_path'], 'hyperparameters.json'), 'w') as f:
            json.dump(params, f)
        
        # Initialize network and its weights
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
        net = SubdNet(params)
        net = net.to(device)
        net.apply(init_weights)

        lossFunc = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])

        trainLossHis = []
        validLossHis = []
        bestLoss = np.inf

        for epoch in range(params['epochs']):
            ts = time.time()

            # Training loop over training indices
            trainErr = 0.0
            for mIdx in train_indices:
                x = S.getInputData(mIdx)
                outputs = net(x, mIdx, S.hfList, S.poolMats, S.dofs)
                Vt = S.meshes[mIdx][params["numSubd"]].V.to(device)
                loss = 0.0
                for ii in range(params["numSubd"]+1):
                    nV = outputs[ii].size(0)
                    loss += lossFunc(outputs[ii], Vt[:nV, :])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                trainErr += loss.cpu().item()

            trainLoss = trainErr / len(train_indices)
            trainLossHis.append(trainLoss)
            
            # Validation loop over validation indices
            validErr = 0.0
            with torch.no_grad():
                for mIdx in valid_indices:
                    x = S.getInputData(mIdx)
                    outputs = net(x, mIdx, S.hfList, S.poolMats, S.dofs)
                    Vt = S.meshes[mIdx][params["numSubd"]].V.to(device)
                    loss = 0.0
                    for ii in range(params["numSubd"]+1):
                        nV = outputs[ii].size(0)
                        loss += lossFunc(outputs[ii], Vt[:nV, :])
                    validErr += loss.cpu().item()
            validLoss = validErr / len(valid_indices)
            validLossHis.append(validLoss)

            # Save best network weights
            if validLoss < bestLoss:
                torch.save(net.state_dict(), os.path.join(params['output_path'], NETPARAMS))

            remain = int(round((params['epochs'] - epoch) * (time.time() - ts)))
            print("epoch %d, train loss: %.6e, valid loss: %.6e, remain time: %s" %
                  (epoch, trainLoss, validLoss, remain))

        # Save loss history
        np.savetxt(os.path.join(params['output_path'], 'train_loss.txt'), np.array(trainLossHis), delimiter=',')
        np.savetxt(os.path.join(params['output_path'], 'valid_loss.txt'), np.array(validLossHis), delimiter=',')


    print("Generating dataset using random_subdiv_remesh_bin.exe on", config["mesh_folder"])
    generate_dataset()
    convert_to_pkl(delete = False)
    print("Training the network")
    train_model(config)

if __name__ == '__main__':
    main()