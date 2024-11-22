This repository is tested on Ubuntu 22.04 with Python 3.10. 

# Installation
1. Clone this reposity
2. Run `python3 -m venv mmwave_venv` to create a new virtual environment
3. Run `source mmwave_venv/bin/activate` to source the virtual environment
4. Run `python3 setup.py --install` to run the install. Please note:
    1. This installs numpy 1.24.3, which is required for PSBody. If you are not planning to run the simulation, you can use a different version of numpy by changing requirements.txt.
    2. This setup will install requirements as needed, depending on the settings in param.json. It will only install GPU requirements if use_cuda is true, and it will only install simulation/segmentation requirements if use_simulation/use_segmentation are true. Please update params.json accordingly before running python3 setup.py

# Video Demonstration

A video demonstration of MITO is available in `MITO_Video.mov`. It can also be seen here: https://youtu.be/OUmoDZ4PEiE


# Accessing Data
In the camera ready, we will provide an AWS link for the dataset in its entirety. To maintain anonymity, we currently provide data for one sample object at: https://drive.google.com/drive/folders/102Fiumd-HWXV_wIBj8oZeB9sXy81qmLg?usp=sharing

Download the zip file from the anonymous google drive, and extract it within the `data/` folder. 

# Visualizing Data
Data can be visualized by running `cd src/utils && python3 visualization.py`. More details can be found in the documentation of that python file.

# Classifier
Run `train_classifier.py` to recreate the current classifier results. 

# Running the Simulation
The simulation can be run with the `run_simulation.sh` file. See the file for more details. 

# Tutorials 
NOTE: The tutorials will be provided with the camera ready release. 

We provide the following tutorials to introduce different features of this repository. All tutorials are in the `tutorials\` folder.
### 1. Loading and Visualizing the Dataset
This tutorial introduces the dataset (contents and structure) and shows how to download, access, and visualize the data. If your goal is to build new models using the previously processed images, this tutorial should be sufficient for your goals. The remainder of the tutorials show more advanced functionality (e.g., building models on this dataset or simulating/processing new images.)

### 2. Simulating new mmWave Images
This tutorial shows how to use our open-source simulation tool to produce synthetic images for any 3D mesh. This can be used to produce more synthetic data than we have released in our initial dataset release. 

### 3. Segmenting mmWave Images
This tutorial shows our approach for segmenting mmWave images using the SegmentAnything model (https://github.com/facebookresearch/segment-anything). This is a good example of using our data in downstream models.

### 4. Classifying mmWave Images
This tutorial shows our approach for classifying mmWave images, using a custom classification network. This is a good example of buidling custom models with our dataset. 

### 5. Understanding mmWave Imaging
This is an advanced tutorial explaining in-depth how mmWave imaging works. 
