import os
import argparse
import time
import csv
import sys
import json
import random
import numpy as np
import ase
import pprint

from matdeeplearn import models, process, training

import torch
from torch_geometric.data import DataLoader, Dataset, Data
import torch_geometric.transforms as T

import ray
from ray import tune

################################################################################
#
################################################################################
"""
MatDeepLearn code 
"""
################################################################################
#
################################################################################

start_time = time.time()
print("Starting...")
print(
    "GPU is available:",
    torch.cuda.is_available(),
    ",Quantity: ",
    torch.cuda.device_count(),
)

parser = argparse.ArgumentParser(description="ML framework")
###Training arguments
parser.add_argument(
    "--job_name",
    default="myjob",
    type=str,
    help="name of your job and output files/folders (default:myjob)",
)
parser.add_argument(
    "--train_ratio", default=0.8, type=float, help="train ratio (default:0.8)"
)
parser.add_argument(
    "--val_ratio", default=0.05, type=float, help="validation ratio (default:0.05)"
)
parser.add_argument(
    "--test_ratio", default=0.15, type=float, help="test ratio (default:0.15)"
)
parser.add_argument(
    "--epochs",
    default=None,
    type=int,
    help="number of total epochs to run (default:None)",
)
parser.add_argument(
    "--batch_size", default=None, type=int, help="batch size (default:None)"
)
parser.add_argument(
    "--lr", default=None, type=float, help="initial learning rate (default:None)"
)
parser.add_argument(
    "--run_mode",
    default="training",
    type=str,
    help="run modes: training, predict, CV, hyperparameter, training_ensemble, training_repeat, analysis (default: training )",
)
parser.add_argument(
    "--verbosity", default=5, type=int, help="prints errors every x epochs (default: 5)"
)
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="seed for data split, 0=random (default:0)",
)
parser.add_argument(
    "--loss",
    default="l1_loss",
    type=str,
    help="l1_loss, mse_loss, binary_cross_entropy (default: l1_loss (MAE))",
)
parser.add_argument(
    "--write_output",
    default="True",
    type=str,
    help="Write outputs to csv (default: True)",
)
parser.add_argument(
    "--target_index",
    default=0,
    type=int,
    help="which column to use as target property in the target file (default:0)",
)
###Model arguments
parser.add_argument(
    "--model",
    default="CGCNN",
    type=str,
    help="CGCNN, MPNN, SchNet, MEGNet, GCN_net, SOAP, SM (default:CGCNN)",
)
parser.add_argument(
    "--model_path",
    default="myjob_model.pth",
    type=str,
    help="path of the model .pth file (default:myjob_model.pth)",
)
parser.add_argument(
    "--save_model",
    default="True",
    type=str,
    help="Write outputs to csv (default: True)",
)
parser.add_argument(
    "--load_model",
    default="False",
    type=str,
    help="Write outputs to csv (default: False )",
)
###Processing arguments
parser.add_argument(
    "--hyperparameter_path",
    default="hyperparameters.json",
    type=str,
    help="Location of ML hyperparameter file (default:hyperparameters.json)",
)
parser.add_argument(
    "--data_path",
    default="data/2D_data",
    type=str,
    help="Location of data containing structures (json or any other valid format) and accompanying files (default: data/2D_data)",
)
parser.add_argument(
    "--reprocess",
    default="False",
    type=str,
    help="Reprocess data since last run (default:False)",
)
parser.add_argument(
    "--format", default="json", type=str, help="format of input data (default:--)"
)
parser.add_argument(
    "--dictionary_path",
    default="atom_dict.json",
    type=str,
    help="path to atom features dictionary (default:atom_dict.json)",
)
parser.add_argument(
    "--target_path",
    default="targets.csv",
    type=str,
    help="path to file-target designation file (default: --)",
)
parser.add_argument(
    "--extra_features",
    default="False",
    type=str,
    help="Calculates extra SOAP and SM features (default: False)",
)
# Mode specific arguments
parser.add_argument(
    "--hyper_samples",
    default=160,
    type=int,
    help="number of trials for hyperparameter optimization (default:160)",
)
parser.add_argument(
    "--hyper_concurrency",
    default=8,
    type=int,
    help="hyperparameter optimization concurrency (default:8)",
)
parser.add_argument(
    "--hyper_resume",
    default="False",
    type=str,
    help="Resume hyperparameter training (default:False)",
)
parser.add_argument(
    "--ensemble_list",
    default="CGCNN,MPNN,SchNet,MEGNet",
    type=str,
    help="List of models for ensemble; dont insert spaces between model names (default:CGCNN,MPNN,SchNet,MEGNet)",
)
parser.add_argument(
    "--repeat_trials",
    default=5,
    type=int,
    help="Number of repeat trials for repeated training (default:5)",
)
parser.add_argument(
    "--cv_folds",
    default=5,
    type=int,
    help="Number of folds for cross validation (default:5)",
)

# Get arguments from command line
args = parser.parse_args(sys.argv[1:])

with open(args.hyperparameter_path) as f:
    hyperparameters = json.load(f)

model_params = hyperparameters["model_parameters"][args.model]
input_params = hyperparameters["input_parameters"]

args.min_radius = float(input_params["graph"]["min_radius"])
args.max_radius = float(input_params["graph"]["max_radius"])
args.max_neighbors = int(input_params["graph"]["max_neighbors"])
args.gaussians = int(input_params["graph"]["gaussians"])
args.voronoi = str(input_params["graph"]["voronoi"])
args.fullconn = str(input_params["graph"]["fullconn"])

args.SOAP_rcut = float(input_params["SOAP"]["SOAP_rcut"])
args.SOAP_nmax = int(input_params["SOAP"]["SOAP_nmax"])
args.SOAP_lmax = int(input_params["SOAP"]["SOAP_lmax"])
args.SOAP_sigma = float(input_params["SOAP"]["SOAP_sigma"])

if args.epochs == None:
    args.epochs = int(model_params["epochs"])
if args.lr == None:
    args.lr = float(model_params["lr"])
if args.batch_size == None:
    args.batch_size = int(model_params["batch_size"])

print("Settings: ")
pprint.pprint(vars(args))

################################################################################
#  Begin processing
################################################################################

if args.seed == 0:
    args.seed = np.random.randint(1, 1e6)

if args.run_mode != "hyperparameter":

    process_start_time = time.time()
    if args.reprocess == "True":
        os.system("rm -rf " + str(args.data_path) + "/processed")

    if args.voronoi == 'True':
        transform = T.Compose([process.Get_Voronoi(), process.Get_Y(index=args.target_index)])
        dataset = process.StructureDataset(
            data_path=args.data_path,
            save_dir=args.data_path,
            params=args,
            transform=transform,
        )
    elif args.fullconn == 'True':
        transform = T.Compose([process.Get_Full(), process.Get_Y(index=args.target_index)])
        dataset = process.StructureDataset(
            data_path=args.data_path,
            save_dir=args.data_path,
            params=args,
            transform=transform,
        )
    else:
        transform = T.Compose([process.Get_Y(index=args.target_index)])
        dataset = process.StructureDataset(
            data_path=args.data_path, save_dir=args.data_path, params=args, transform=transform,
        )
    print("--- %s seconds for processing ---" % (time.time() - process_start_time))

    # print(dataset, dataset[0], dataset[-1])

elif args.run_mode == "hyperparameter":
    if args.reprocess == "False":
        dataset = process.StructureDataset(
            data_path=args.data_path, save_dir=args.data_path, params=args
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
#  Training begins
################################################################################

# Regular training
if args.run_mode == "training":

    print("Starting regular training")
    print(
        "running with "
        + str(args.epochs)
        + " epochs"
        + " on "
        + str(args.model)
        + " model"
    )
    train_error, val_error, test_error = training.train_regular(
        model_params,
        dataset,
        device,
        args,
        load=args.load_model,
        write=args.write_output,
        save=args.save_model,
        print_model="True",
    )

    print("Test Error: {:.5f}".format(test_error))

# Predicting from a trained model
elif args.run_mode == "predict":

    print("Starting prediction from trained model")
    print(
        "running with "
        + str(args.epochs)
        + " epochs"
        + " on "
        + str(args.model)
        + " model"
    )
    train_error = training.predict(dataset, device, args, write=args.write_output)

    print("Test Error: {:.5f}".format(train_error))

# Running n fold cross validation
elif args.run_mode == "CV":

    print("Starting cross validation")
    print(
        "running with "
        + str(args.epochs)
        + " epochs"
        + " on "
        + str(args.model)
        + " model"
    )
    CV_error = training.train_CV(
        model_params, dataset, device, args, write=args.write_output
    )
    print("CV error", CV_error)

# Hyperparameter optimization
elif args.run_mode == "hyperparameter":

    print("Starting hyperparameter optimization")
    print(
        "running with "
        + str(args.epochs)
        + " epochs"
        + " on "
        + str(args.model)
        + " model"
    )

    # set up search space for each model; these can subject to change
    hyper_args = {}
    dim1 = [x * 10 for x in range(1, 20)]
    dim2 = [x * 10 for x in range(1, 20)]
    dim3 = [x * 10 for x in range(1, 20)]
    batch = [x * 10 for x in range(1, 20)]
    hyper_args["SchNet"] = {
        "dim1": tune.choice(dim1),
        "dim2": tune.choice(dim2),
        "dim3": tune.choice(dim3),
        "conv_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
        "pool": tune.choice(
            ["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]
        ),
        "lr": tune.loguniform(1e-4, 0.05),
        "batch_size": tune.choice(batch),
        "cutoff": args.max_radius,
    }
    hyper_args["CGCNN"] = {
        "dim1": tune.choice(dim1),
        "dim2": tune.choice(dim2),
        "conv_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
        "pool": tune.choice(
            ["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]
        ),
        "lr": tune.loguniform(1e-4, 0.05),
        "batch_size": tune.choice(batch),
    }
    hyper_args["MPNN"] = {
        "dim1": tune.choice(dim1),
        "dim2": tune.choice(dim2),
        "dim3": tune.choice(dim3),
        "conv_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
        "pool": tune.choice(
            ["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]
        ),
        "lr": tune.loguniform(1e-4, 0.05),
        "batch_size": tune.choice(batch),
    }
    hyper_args["MEGNet"] = {
        "dim1": tune.choice(dim1),
        "dim2": tune.choice(dim2),
        "dim3": tune.choice(dim3),
        "conv_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
        "pool": tune.choice(["mean", "sum", "max", "set2set"]),
        "lr": tune.loguniform(1e-4, 0.05),
        "batch_size": tune.choice(batch),
    }
    hyper_args["GCN_net"] = {
        "dim1": tune.choice(dim1),
        "dim2": tune.choice(dim2),
        "conv_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
        "pool": tune.choice(
            ["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]
        ),
        "lr": tune.loguniform(1e-4, 0.05),
        "batch_size": tune.choice(batch),
    }
    hyper_args["SOAP"] = {
        "dim1": tune.choice(dim1),
        "fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
        "lr": tune.loguniform(1e-4, 0.05),
        "batch_size": tune.choice(batch),
        "nmax": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "lmax": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "sigma": tune.uniform(0.1, 2.0),
        "rcut": tune.uniform(1.0, 10.0),
    }
    hyper_args["SM"] = {
        "dim1": tune.choice(dim1),
        "fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
        "lr": tune.loguniform(1e-4, 0.05),
        "batch_size": tune.choice(batch),
    }

    best_trial = training.tune_setup(hyper_args[args.model], args)
    global_parameters = best_trial.config["global"]
    del best_trial.config["global"]
    hyperparameters = best_trial.config
    hyperparameters = {
        k: round(v, 5) if isinstance(v, float) else v
        for k, v in hyperparameters.items()
    }
    with open("hyperparameters_optimized.json", "w", encoding="utf-8") as f:
        json.dump(hyperparameters, f, ensure_ascii=False, indent=4)

    print("Best trial hyper_args: {}".format(hyperparameters))
    print(
        "Best trial final validation error: {:.5f}".format(
            best_trial.last_result["loss"]
        )
    )

# ensemble
elif args.run_mode == "training_ensemble":

    print("Starting simple (average) ensemble training")
    args.ensemble_list = args.ensemble_list.split(",")

    ensemble_test_error = training.train_ensemble(
        args.ensemble_list,
        hyperparameters["model_parameters"],
        dataset,
        device,
        args,
        write=args.write_output,
        save=args.save_model,
        print_model="False",
    )
    print("Ensemble Test Error: {:.5f}".format(ensemble_test_error))

# analysis mode
elif args.run_mode == "analysis":
    print("Starting analysis of graph features")
    print(
        "running with "
        + str(args.epochs)
        + " epochs"
        + " on "
        + str(args.model)
        + " model"
    )
    # dict for the tsne settings; please refer to sklearn.manifold.TSNE for information on the function arguments
    tsne_args = {
        "perplexity": 50,
        "early_exaggeration": 12,
        "learning_rate": 300,
        "n_iter": 5000,
        "verbose": 1,
        "random_state": 42,
    }
    # this saves the tsne output as a csv file with: structure id, y, tsne 1, tsne 2 as the columns
    training.analysis(
        model_params,
        dataset,
        device,
        args,
        tsne_args,
        load=args.load_model,
        write=args.write_output,
        save=args.save_model,
    )

# Running repeated trials
elif args.run_mode == "training_repeat":
    args.run_trials = 5
    print("Starting training with repeats")
    print(
        "Using the "
        + str(args.model)
        + " model running for"
        + str(args.epochs)
        + " epochs"
    )
    trial_errors = training.train_repeat(
        model_params,
        dataset,
        device,
        args,
        write=args.write_output,
        save=args.save_model,
    )
    print(
        "Train Average Error: {:.5f}".format(np.average(np.array(trial_errors)[:, 0])),
        "Train Standard Deviation: {:.5f}".format(np.std(np.array(trial_errors)[:, 0])),
    )
    print(
        "Val Average Error: {:.5f}".format(np.average(np.array(trial_errors)[:, 1])),
        "Val Standard Deviation: {:.5f}".format(np.std(np.array(trial_errors)[:, 1])),
    )
    print(
        "Test Average Error: {:.5f}".format(np.average(np.array(trial_errors)[:, 2])),
        "Test Standard Deviation: {:.5f}".format(np.std(np.array(trial_errors)[:, 2])),
    )

else:
    print("No valid mode selected, try again")

print("--- %s total seconds elapsed ---" % (time.time() - start_time))
