

# MatDeepLearn

MatDeepLearn is a platform for testing and using graph neural networks (GNNs) and other machine learning (ML) models for materials chemistry applications. MatDeepLearn takes in data in the form of atomic structures and their target properties, processes the data into graphs, trains the ML model of choice (optionally with hyperparameter optimization), and provides predictions on unseen data. It allows for different GNNs to be benchmarked on diverse datasets drawn from materials repositories as well as conventional training/prediction tasks. This package makes use of the [Pytorch-Geometric](https://github.com/rusty1s/pytorch_geometric) library, which provides powerful tools for GNN development and many prebuilt models readily available for use.

MatDeepLearn is currently under active development with more features to be added soon. Please <a href="#roadmap">contact</a> the developer(s) for bug fixes and feature requests.

## Table of contents
<ol>
	<li><a href="#installation">Installation</a></li>
	<li><a href="#usage">Usage</a></li>
	<li><a href="#faq">FAQ</a></li>
	<li><a href="#roadmap">Roadmap</a></li>
	<li><a href="#license">License</a></li>
	<li><a href="#acknowledgements">Acknowledgements</a></li>
</ol>

## Installation


### Prerequisites

Prerequisites are listed in requirements.txt. You will need two key packages, 1. Pytorch and 2. Pytorch-Geometric. You may want to create a virtual environment first, using Conda for example.

1. **Pytorch**: The package has been tested on Pytorch 1.9. To install, for example:
	```bash
	pip install torch==1.9.0
	```
2. **Pytorch-Geometric:**  The package has been tested on Pytorch-Geometric. 2.0.1 To install, [follow their instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), for example:
	```bash
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-geometric
	```	
    where where ${CUDA} and ${TORCH} should be replaced by your specific CUDA version (cpu, cu92, cu101, cu102, cu110, cu111) and PyTorch version (1.7.0, 1.8.0, 1.9.0), respectively.

3. **Remaining requirements:** The remainder may be installed by:
	```bash
    git clone https://github.com/vxfung/MatDeepLearn
    cd MatDeepLearn    
	pip install -r requirements.txt
	```
    
## Usage

### Running your first calculation

This example provides instructions for a bare minimum calculation. We will run the example with a on a small dataset (the Pt subset dataset containing ~1000 entries). This is just a toy example to test if the package is installed and working. Procedure below:

1. Go to MatDeepLearn/data/ and type
	```bash
	tar -xvf test_data.tar.gz 
	```
	to unpack the a test dataset of Pt clusters.
	
2.	Go to MatDeepLearn, type
	```bash
	python main.py --data_path=data/test_data/test_data
	```
	where default settings will be used and configurations will be read from the provided config.yml.
	
3. The program will begin training; on a regular CPU this should take ~10-20s per epoch. It is recommended to use GPUs which can provide a roughly ~5-20 times speedup, which is needed for the larger datasets. As default, the program will provide two outputs: (1) "my_model.pth" which is a saved model which can be used for predictions on new structures, (2) "myjob_train_job_XXX_outputs.csv" where XXX are train, val and test; these contain structure ids, targets and the predicted values from the last epoch of training and validation, and for the test set.

### The configuration file

The configuration file is provided in .yml format and encodes all the settings used. By default it should be in the same directory as main.py or specified in a separate location by --config_path in the command line. 

There are four categories or sections: 1. Job, 2. Processing, 3. Training, 4. Models

1. **Job:** This section encodes the settings specific to the type of job to run. Current supported are: Training, Predict, Repeat, CV, Hyperparameter, Ensemble, Analysis. The program will only read the section for the current job, which is selected by --run_mode in the command line, e.g. --run_mode=Training. Some other settings which can be changed in the command line are: --job_name, --model, --seed, --parallel.

2. **Processing:** This section encodes the settings specific to the processing of structures to graphs or other features. Primary settings are the "graph_max_radius", "graph_max_neighbors" and "graph_edge_length" which controls radius cutoff for edges, maximum number of edges, and length of edges from a basis expansion, respectively. Prior to this, the directory path containing the structure files must be specified by "data_path" in the file or --data_path in the command line.

3. **Training:** This section encodes the settings specific to the training. Primary settings are the "loss", "train_ratio" and "val_ratio" and "test_ratio". This can also be specified in the command line by --train_ratio, --val_ratio, --test_ratio.

4. **Models:** This section encodes the settings specific to the model used, aka hyperparameters. Example hyperparameters are provided in the example config.yml. Only the settings for the model selected in the Job section will be used. Model settings which can be changed in the command line are: --epochs, --batch_size, and --lr.


### Training and prediction on an unseen dataset

This example provides instructions for a conventional ML task of training on an existing dataset, and using a trained model to provide predictions on an unseen dataset for screening. This assumes the model used is already sufficiently good at the task to be performed (with suitable model hyperparameters, etc.). The default hyperparameters can do a reasonably good job for testing purposes; for hyperparameter optimization refer to the next section.

1. To run, MatDeepLearn requires: 
	- A configuration file, config.yml, as described in the previous section. 
	- A dataset directory containing structure files, a csv file containing structure ids and target properties (default: targets.csv), and optionally a json file containing elemental properties (default: atom_dict.json). Five example datasets are provided with all requisite files needed. Structure files can take any format supported by the Atomic Simulation Environment [(ASE)](https://wiki.fysik.dtu.dk/ase/) such as .cif, .xyz, POSCAR, and ASE's own .json format.

2. It is then necessary to first train the ML model an on existing dataset with available target properties. A general example for training is:

	```bash
	python main.py --data_path='XXX' --job_name="my_training_job" --run_mode='Training' --model='CGCNN_demo' --save_model='True' --model_path='my_trained_model.pth'
	```		
	where "data_path" points to the path of the training dataset, "model" selects the model to use, and "run_mode" specifies training. Once finished, a "my_trained_model.pth" should be saved. 

3. Run the prediction on an unseen dataset by:

	```bash
	python main.py --data_path='YYY' --job_name="my_prediction_job" --run_mode='Predict' --model_path='my_trained_model.pth'
	```		
	where the "data_path" and "run_mode" are now updated, and the model path is specified. The predictions will then be saved to my_prediction_job_predicted_outputs.csv for analysis.
	
### Hyperparameter optimization

This example provides instructions for hyperparameter optimization. 

1. Similar to regular training, ensure the dataset is available with requisite files in the directory.

2. To run hyperparameter optimization, one must first define the hyperparameter search space. MatDeepLearn uses [RayTune](https://docs.ray.io/en/master/tune/index.html) for distributed optimization, and the search space is defined with their provided methods. The choice of search space will depend on many factors, including available computational resources and focus of the study; we provide some examples for the existing models in main.py.

3. Assuming the search space is defined, we run hyperparameter optimization with :
	```bash
	python main.py --data_path=data/test_data --model='CGCNN_demo' --job_name="my_hyperparameter_job" --run_mode='Hyperparameter'
	```		
	this sets the run mode to hyperparameter optimization, with a set number of trials and concurrency. Concurrently sets the number of trials to be performed in parallel; this number should be higher than the number of available devices to avoid bottlenecking. The program should automatically detect number of GPUs and run on each device accordingly. Finally, an output will be written called "optimized_hyperparameters.json" which contains the hyperparameters for the model with the lowest test error. Raw results are saved in a directory called "ray_results."

### Ensemble

Ensemble functionality is provided here in the form of stacked models, where prediction output is the average of individual models. "ensemble_list" in the config.yml controls the list of individual models represented as a comma separated string.

```bash
python main.py --data_path=data/test_data --run_mode=Ensemble
```		

### Repeat trials

Sometimes it is desirable to obtain performance averaged over many trials. Specify repeat_trials in the config.yml for how many trials to run.

```bash
python main.py --data_path=data/test_data --run_mode=Repeat
```		

### Cross validation

Specify cv_folds in the config.yml for how many folds in the CV.

```bash
python main.py --data_path=data/test_data --run_mode=CV
```		

### Analysis

This mode allows the visualization of graph-wide features with t-SNE.

```bash
python main.py --data_path=data/test_data --run_mode=Analysis --model_path=XXX
```		
    
## FAQ



## Roadmap

TBA


## License

Distributed under the MIT License. 


## Acknowledgements

Contributors: Victor Fung, Eric Juarez, [Jiaxin Zhang](https://github.com/jxzhangjhu), Bobby Sumpter

## Contact

Code is maintained by:

[Victor Fung](https://www.ornl.gov/staff-profile/victor-fung), fungv (at) ornl.gov

