# Final Project - Comparing Mammalian DTI Scans with Unsupervised Optical Flow

Hey! welcome!
This project explores the use of Diffusion Tensor Imaging (DTI) scans to estimate distances between different animals. It leverages the PWC-Net architecture for optical flow estimation, adapting it from a previous project focused on lung scans (by Yariv Levy and Doron Kopit).

**GDL lab, EE Engineering Faculty, Tel Aviv University 2024**

 ## Project Overview:

* **Goal:** Estimate distances between animals using DTI scans and optical flow.
* **Method:**
    * Utilizes the PWC-Net model for optical flow calculation.
    * continuation of a prior lung scan optical flow project.
* **Implementation:**
    * `main_train.py` script iterates through a list of animals and outputs distance calculations to a designated JSON file.
    * The `Animals_to_check` and `save_root_json` variables are defined within the `main_train.py` script.
    * Users can choose between training the model for a new animal pair or using a pre-trained model from their folder with the `--distance` argument.

## Running the Project:

1. **Install dependencies:** A requirements.txt is included. This project was developed on Ubuntu 18.04.6 LTS, but this os is not strictly necessary
2. **Now you have two options:**
   * *Train the model for a new animal pair:*
   ```bash
   python main_train.py -c=$USER/your_configuration_path -l=$USER/model_to_load_path
   ```
   This will loop over the possible pairs and train the model. In practice we do not train from scratch but rather use a basic model to start from (load it in the -l argument). 
   * *Use a trained model that exists in your folders for distance estimation:*
   ```bash
   python main_train.py --distance
   ```
This will prioritize distance calculation over model training. If a pre-trained model for a specific animal pair exists, it will be used. Otherwise, a new model will be trained.
## Further Notes:

* The distance is inspired from the gromov hausdorff distance.
* You can set your own parameters for the distance (number of anchor point for example) using the script arguments.




