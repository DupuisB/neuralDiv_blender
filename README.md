# Neural Subdivision for Blender

---

**A full presentation of the project can be found in the [report](report.pdf).**

---

This repository contains a Blender extension that implements the research paper **["Neural Subdivision"](https://arxiv.org/abs/2005.01819)**, allowing for data-driven, stylized 3D mesh refinement directly within the viewport.

Instead of traditional, linear subdivision rules, this tool uses a neural network to learn non-linear rules from example meshes. This allows for the creation of unique, "stylized" geometry that can capture the essence of a training dataset.

**Note that this repository is not production-ready and is intended for educational purposes. Some features may be limited or missing.**

---

## 💡 Core Features

*   **Apply Neural Subdivision:** Refine any selected mesh using a pre-trained neural network.
*   **Stylized Results:** Choose from different models trained on distinct datasets (e.g., humanoid, organic, chaotic) to achieve unique stylistic effects.
*   **Simple UI:** Integrated directly into Blender's sidebar (`N` Panel) and accessible via a hotkey (`Shift + N`).

![Demo GIF](https://i.imgur.com/cyhog6l.gif")
> *A quick demonstration of the plugin inside Blender (side by side loop subdivision and neural subdivision).*

---

## 🔧 Installation & Usage

1.  **Download:** Download the repo as a `.zip` file or clone it.
    *   Note that this repository **does not** contain the pre-trained models due to their size.
2.  **Install in Blender:**
    *   Go to `Edit > Preferences > Add-ons > Install...`
    *   Select the downloaded `.zip` file.
    *   Enable the "Neural Subdivision" add-on.
3.  **Usage:**
    *   Select a mesh object in your scene.
    *   Open the sidebar with the `N` key and find the "Neural SubD" tab.
    *   Alternatively, use the `Shift + N` hotkey.
    *   Choose a pre-trained model from the dropdown.
    *   Click **Apply Neural Subdivision**.

---


## 🧠 Training Your Own Models

This repository also contains the Python scripts to train your own custom subdivision models.

The training pipeline is located in the `/training` directory.

### Quickstart

1.  **Prepare Dataset:** Edit the first two lines in `train_hyperparameters.json`:
    - Set `mesh_folder` to the folder containing all your `.obj` files.
    - Specify the name of your output folder. This folder must not already exist and should be located under `./data/jobs/{insert name here}/`.
2.  **Install Dependencies:** Install the required Python packages:
    ```bash
    pip install torch scipy matplotlib
    ```
3.  **Run Training:** Execute the training script:
    ```bash
    python train_full.py
    ```
    > **Note:** This setup is designed for Windows.

    If you are using Linux:
    - Recompile the code in `surface_multigrid_code/09_random_subdiv_remesh`.
      - Add `libigl` under `surface_multigrid_code/` (it was removed to reduce the zip file size; you can find it [here](https://github.com/libigl/libigl/tree/70d2abaa53616af35c7857bf8f52c8a234f150ff)).
    - Replace `random_subdiv_remesh_bin.exe` with your newly compiled file (update both the folder and `train_full.py` accordingly).


##  Acknowledgments

This project is an implementation and extension of the work presented in the paper **Neural Subdivision** by [Liu et al](https://arxiv.org/abs/2005.01819). The original author's code can be found [here](https://github.com/HTDerekLiu/neuralSubdiv). This work was completed as a project report for the IGR course at Télécom Paris.