

# MatDeepLearn

This software package provides a testing platform for graph neural networks (GNNs) and other machine learning (ML) models. MatDeepLearn takes in data in the form of atomic structures and their target properties, processes the data into graphs, trains the ML model of choice (optionally with hyperparameter optimization), and provides predictions on unseen data. It allows for different GNNs to be benchmarked on diverse datasets drawn from materials repositories as well as conventional training/prediction tasks. This package makes use of the [Pytorch-Geometric](https://github.com/rusty1s/pytorch_geometric) library, which provides powerful tools for GNN development and many prebuilt models readily available for use.

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

Ensure prerequisites are installed, then clone the repo. 

### Prerequisites

Prerequisites are listed in requirements.txt. List reproduced here:

pymatgen: 2020.9.14 \
ase: 3.20.1 \
scipy: 1.3.1 \
numpy: 1.17.2 \
joblib: 0.13.2 \
torch: 1.6.0 \
torch_scatter: 2.0.5 \
torch_geometric: 1.6.1 \
matplotlib: 3.1.1 \
ray: 1.0.1.post1 \
dscribe: 0.3.5 \
scikit_learn: 0.24.0 


## Usage

### Running your first calculation

This example provides instructions for a bare minimum calculation. We will run the example with a on a small dataset (the test dataset containing ~1000 entries):

1. Go to MatDeepLearn/data/ and type
	```bash
	tar -xvf test_data.tar.gz 
	```
	to unpack the test dataset.
	
2.	Go to MatDeepLearn, type
	```bash
	python main.py
	```
	where default settings will be used and hyperparameters will be read from the provided hyperparameters.json
	
3. The program will begin training; on a regular CPU this should take ~10-20s per epoch. It is recommended to use GPUs which can provide a roughly ~5-20 times speedup, which is needed for the larger datasets. As default, the program will provide two outputs: (1) "myjob_model.pth" which is a saved model which can be used for predictions on new structures, (2) "myjob_XXX_outputs.csv" where XXX are train, val and test; these contain structure ids, targets and the predicted values from the last epoch of training and validation, and for the test set.


### Training and prediction on an unseen dataset

This example provides instructions for a conventional ML task of training on an existing dataset, and using a trained model to provide predictions on an unseen dataset for screening. This assumes the model used is already sufficiently good at the task to be performed (with suitable hyperparameters, etc.). The default hyperparameters can do a reasonably good job for testing purposes; for hyperparameter optimization refer to the next section.

1. To run, MatDeepLearn requires: 
	- A dataset directory containing structure files, a csv file containing structure ids and target properties (default: targets.csv), and a json file containing elemental properties (default: atom_dict.json). Five example datasets are provided with all requisite files needed. Structure files can take any format supported by the Atomic Simulation Environment [(ASE)](https://wiki.fysik.dtu.dk/ase/) such as .cif, .xyz, POSCAR, and ASE's own .json format.
	- A hyperparameter.json file containing input and model hyperparameters. These will contain model specific hyperparameters such as the number and dimensions of the neural network layers, as well as parameters for training. They will also contain input parameters which determines how the graphs are constructed. An example hyperparameter.json file is provided with default hyperparameters.

2. It is then necessary to first train the ML model an on existing dataset with available target properties. A general example for training is:

	```bash
	python main.py --job_name='my_training_job' --data_path='XXX' --model='CGCNN' --run_mode='training' --save_model=True
	```		
	where "data_dir" points to the path of the training dataset, "model" selects the model to use (available models include CGCNN, MPNN, SchNet, MEGNet, GCN_net, SOAP, SM) , and "run_mode" specifies training. Once finished, a "my_training_job_model.pth" should be saved. 

3. Run the prediction on an unseen dataset by:

	```bash
	python main.py --job_name='my_prediction_job' --data_path='YYY' --run_mode='predict' --model_path="my_training_job_model.pth"
	```		
	where the "data_dir" and "run_mode" are now updated, and the model path is specified. The predictions will then be saved to my_prediction_job_test_outputs.csv for analysis.
	
### Hyperparameter optimization

This example provides instructions for hyperparameter optimization. 

1. Similar to regular training, ensure the dataset is available with requisite files in the directory.

2. To run hyperparameter optimization, one must first define the hyperparameter search space. MatDeepLearn uses [RayTune](https://docs.ray.io/en/master/tune/index.html) for distributed optimization, and the search space is defined with their provided methods. The choice of search space will depend on many factors, including available computational resources and focus of the study; we provide some examples for the existing models in main.py.

3. Assuming the search space is defined, we run hyperparameter optimization with :
	```bash
	python main.py --job_name='my_hyperparameter_job' --data_path='XXX' --model='CGCNN' --run_mode='hyperparameter' --hyper_concurrency=8 --hyper_samples=160
	```		
	this sets the run mode to hyperparameter optimization, with 160 trials and a concurrency of 8. Concurrently sets the number of trials to be performed in parallel; this number should be higher than the number of available devices to avoid bottlenecking. The program should automatically detect number of GPUs and run on each device accordingly. Finally, an output will be written called "hyperparameters_optimized.json" which contains the hyperparameters for the model with the lowest test error. Raw results are saved in a directory called "ray_results."

## FAQ



## Roadmap

TBA


## License

Distributed under the MIT License. 


## Acknowledgements

Contributors: Victor Fung, Eric Juarez, Jiaxin Zhang, Bobby Sumpter

## Contact

Code is maintained by:

[Victor Fung](https://www.ornl.gov/staff-profile/victor-fung), fungv (at) ornl.gov

