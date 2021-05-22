import csv
import os
import time
import numpy as np
from functools import partial

import torch.nn.functional as F
import torch
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset

import matdeeplearn.models as models
import matdeeplearn.process as process
import matdeeplearn.training as training
from matdeeplearn.models import model_summary

################################################################################
#  Training functions
################################################################################

# Generic pytorch train
def train(epoch, model, optimizer, loader, loss_method, device):
    model.train()
    loss_all = 0
    predict = []
    target = []
    count = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = getattr(F, loss_method)(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
        output = model(data)
        if count == 0:
            ids = data.structure_id
            predict = output.data.cpu().numpy()
            target = data.y.cpu().numpy()
        else:
            ids = ids + data.structure_id
            predict = np.hstack((predict, output.data.cpu().numpy()))
            target = np.hstack((target, data.y.cpu().numpy()))
        count += 1

    train_out = np.column_stack((ids, target, predict))
    loss_all = loss_all / len(loader.dataset)
    return loss_all, train_out


# Generic pytorch validation
def validation(loader, model, loss_method, device):
    model.eval()
    loss_all = 0
    predict = []
    target = []
    count = 0
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            loss = getattr(F, loss_method)(model(data), data.y)
            loss_all += loss.item() * data.num_graphs
            output = model(data)
            if count == 0:
                ids = data.structure_id
                predict = output.data.cpu().numpy()
                target = data.y.cpu().numpy()
            else:
                ids = ids + data.structure_id
                predict = np.hstack((predict, output.data.cpu().numpy()))
                target = np.hstack((target, data.y.cpu().numpy()))
            count += 1

    val_out = np.column_stack((ids, target, predict))
    loss_all = loss_all / len(loader.dataset)
    return loss_all, val_out


# Generic pytorch test
def test(loader, model, loss_method, device):
    model.eval()
    loss_all = 0
    predict = []
    target = []
    count = 0
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            loss = getattr(F, loss_method)(model(data), data.y)
            loss_all += loss.item() * data.num_graphs
            output = model(data)
            if count == 0:
                ids = data.structure_id
                predict = output.data.cpu().numpy()
                target = data.y.cpu().numpy()
            else:
                ids = ids + data.structure_id
                predict = np.hstack((predict, output.data.cpu().numpy()))
                target = np.hstack((target, data.y.cpu().numpy()))
            count += 1

    test_out = np.column_stack((ids, target, predict))
    loss_all = loss_all / len(loader.dataset)
    return loss_all, test_out


###Regular training with train, val, test split
def train_regular(
    model_params,
    dataset,
    device,
    args,
    load="False",
    write="True",
    save="True",
    print_model="True",
):

    train_start = time.time()

    # Split datasets
    train_dataset, val_dataset, test_dataset = process.split_data(
        dataset, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if load == "False":
        model = getattr(models, args.model)(data=dataset, **model_params).to(device)
    elif load == "True":
        model = torch.load(str(args.job_name) + "_model.pth")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10, min_lr=0.00001, threshold=0.0002
    )
    if print_model == "True":
        model_summary(model)

    train_error = val_error = test_error = epoch_time = float("NaN")

    for epoch in range(1, args.epochs + 1):
        lr = scheduler.optimizer.param_groups[0]["lr"]
        train_error, train_out = train(
            epoch, model, optimizer, train_loader, args.loss, device
        )

        if len(val_dataset) != 0:
            val_error, val_out = validation(val_loader, model, args.loss, device)

        scheduler.step(train_error)
        if epoch % args.verbosity == 0:
            epoch_time = time.time() - train_start
            train_start = time.time()
            print(
                "Epoch: {:04d}, Learning Rate: {:.5f}, Training Error: {:.5f}, Val Error: {:.5f}, Time(s): {:.5f}".format(
                    epoch, lr, train_error, val_error, epoch_time
                )
            )

    if len(test_dataset) != 0:
        test_error, test_out = test(test_loader, model, args.loss, device)

    if save == "True":
        torch.save(model, str(args.job_name) + "_model.pth")

    if write == "True":
        with open(str(args.job_name) + "_train_outputs.csv", "w") as f:
            csvwriter = csv.writer(f)
            for i in range(0, len(train_out)):
                if i == 0:
                    csvwriter.writerow(
                        [
                            "ids",
                            "target",
                            "prediction",
                        ]
                    )
                elif i > 0:
                    csvwriter.writerow(train_out[i - 1, :])
        if len(val_dataset) != 0:
            with open(str(args.job_name) + "_val_outputs.csv", "w") as f:
                csvwriter = csv.writer(f)
                for i in range(0, len(val_out)):
                    if i == 0:
                        csvwriter.writerow(
                            [
                                "ids",
                                "target",
                                "prediction",
                            ]
                        )
                    elif i > 0:
                        csvwriter.writerow(val_out[i - 1, :])
        if len(test_dataset) != 0:
            with open(str(args.job_name) + "_test_outputs.csv", "w") as f:
                csvwriter = csv.writer(f)
                for i in range(0, len(test_out)):
                    if i == 0:
                        csvwriter.writerow(
                            [
                                "ids",
                                "target",
                                "prediction",
                            ]
                        )
                    elif i > 0:
                        csvwriter.writerow(test_out[i - 1, :])

    return train_error, val_error, test_error


### predict using a saved movel
def predict(dataset, device, args, write="True"):

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    assert os.path.exists(args.model_path), "saved model not found"
    model = torch.load(str(args.job_name) + "_model.pth")
    model_summary(model)

    test_error, test_out = test(loader, model, args.loss, device)

    if write == "True":
        with open(str(args.job_name) + "_predict_outputs.csv", "w") as f:
            csvwriter = csv.writer(f)
            for i in range(0, len(test_out)):
                csvwriter.writerow(test_out[i, :])

    return test_error


### n-fold cross validation
def train_CV(model_params, dataset, device, args, write="True"):

    # Split datasets
    cv_dataset = process.split_data_CV(dataset, num_folds=args.cv_folds, seed=args.seed)
    cv_error = 0

    for index in range(0, len(cv_dataset)):

        model_CV = getattr(models, args.model)(data=dataset, **model_params).to(device)
        optimizer_CV = torch.optim.Adam(model_CV.parameters(), lr=args.lr)
        scheduler_CV = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_CV,
            mode="min",
            factor=0.8,
            patience=10,
            min_lr=0.00001,
            threshold=0.0002,
        )
        if index == 0:
            model_summary(model_CV)

        train_dataset = [x for i, x in enumerate(cv_dataset) if i != index]
        train_dataset = torch.utils.data.ConcatDataset(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        test_loader = DataLoader(
            cv_dataset[index],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        train_start = time.time()

        for epoch in range(1, args.epochs + 1):
            lr = scheduler_CV.optimizer.param_groups[0]["lr"]
            train_error, train_out = train(
                epoch,
                model_CV,
                optimizer_CV,
                train_loader,
                args.loss,
                device,
            )
            scheduler_CV.step(train_error)

            if epoch % args.verbosity == 0:
                print(
                    "Epoch: {:04d}, Learning Rate: {:.5f}, Error: {:.5f}".format(
                        epoch, lr, train_error
                    )
                )

        test_error, test_out = test(
            test_loader,
            model_CV,
            args.loss,
            device,
        )
        epoch_time = time.time() - train_start
        train_start = time.time()
        print(
            "Training Error: {:.5f}, Test Error: {:.5f}, Time(s): {:.5f}".format(
                train_error, test_error, epoch_time
            )
        )
        cv_error = cv_error + test_error

        if index == 0:
            total_rows = test_out
        else:
            total_rows = np.vstack((total_rows, test_out))

    if write == "True":
        with open(str(args.job_name) + "_CV_outputs.csv", "w") as f:
            csvwriter = csv.writer(f)
            for i in range(0, len(total_rows)):
                if i == 0:
                    csvwriter.writerow(
                        [
                            "ids",
                            "target",
                            "prediction",
                        ]
                    )
                elif i > 0:
                    csvwriter.writerow(total_rows[i - 1, :])
    cv_error = cv_error / len(cv_dataset)
    return cv_error


###Hyperparameter optimization
# trainable function for ray tune
def tune_trainable(config, checkpoint_dir=None, data_path=None):

    #imports
    from ray import tune
    
    print("Hyperparameter trial start")
    args = config["global"]

    if args.model == "SOAP":
        args.SOAP_nmax = config["nmax"]
        args.SOAP_lmax = config["lmax"]
        args.SOAP_sigma = config["sigma"]
        args.SOAP_rcut = config["rcut"]

    thedir = os.path.dirname(os.path.realpath(__file__))
    print(thedir)
    thedir = os.path.dirname(thedir)
    data_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.dirname(data_path)
    data_path = data_path + "/" + args.data_path
    print(data_path)

    if args.reprocess == "True":
        os.system("rm -rf " + data_path + "/processed")
        dataset = process.StructureDataset(
            data_path=data_path, save_dir=data_path, params=args
        )
    else:
        dataset = process.StructureDataset(
            data_path=data_path, save_dir=data_path, params=args
        )

    train_dataset, val_dataset, test_dataset = process.split_data(
        dataset, args.train_ratio, args.val_ratio + args.test_ratio, 0, args.seed
    )
    train_loader = DataLoader(
        train_dataset, batch_size=int(config["batch_size"]), shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(config["batch_size"]), shuffle=False
    )
    model_hyper = getattr(models, args.model)(
        dataset,
        **config,
    )

    optimizer_hyper = torch.optim.Adam(model_hyper.parameters(), lr=config["lr"])
    scheduler_hyper = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_hyper,
        mode="min",
        factor=0.8,
        patience=10,
        min_lr=0.00001,
        threshold=0.0002,
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model_hyper = torch.nn.DataParallel(model_hyper)
    model_hyper.to(device)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model_hyper.load_state_dict(model_state)
        optimizer_hyper.load_state_dict(optimizer_state)

    for epoch in range(1, args.epochs + 1):
        model_hyper.train()
        for data in train_loader:
            data = data.to(device)
            optimizer_hyper.zero_grad()
            loss = getattr(F, args.loss)(model_hyper(data), data.y)
            loss.backward()
            optimizer_hyper.step()

        # Validation loss
        val_error_all = 0
        model_hyper.eval()
        for data in val_loader:
            with torch.no_grad():
                data = data.to(device)
                val_error = getattr(F, args.loss)(model_hyper(data), data.y)
                val_error_all += val_error.item() * data.num_graphs
        if epoch == args.epochs:
            print("Finished Training")
            if args.reprocess == "True":
                os.system("rm -rf " + data_path + "/processed")
        if epoch % 5 == 0:
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(
                    (model_hyper.state_dict(), optimizer_hyper.state_dict()), path
                )
            tune.report(loss=(val_error_all / len(val_loader.dataset)))


# function for running ray tune
def tune_setup(hyper_args, args):
    
    #imports
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.suggest.hyperopt import HyperOptSearch
    from ray.tune.suggest import ConcurrencyLimiter
    from ray.tune import CLIReporter

    hyper_args["global"] = args
    print(hyper_args["global"])
    ray.init()
    data_path = "_"
    local_dir = "ray_results"
    gpus_per_trial = 1

    search_algo = HyperOptSearch(metric="loss", mode="min", n_initial_points=5)

    max_num_epochs = args.epochs
    grace_period = int(args.epochs * 0.9)
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=grace_period,
        reduction_factor=2,
    )
    search_algo = ConcurrencyLimiter(search_algo, max_concurrent=args.hyper_concurrency)

    if os.path.exists(local_dir + "/" + args.job_name) and os.path.isdir(
        local_dir + "/" + args.job_name
    ):
        # if not os.listdir(local_dir + "/" + args.job_name):
        if args.hyper_resume == "False":
            resume = False
        elif args.hyper_resume == "True":
            resume = True
        else:
            resume = "PROMPT"
    else:
        resume = False

    parameter_columns = [
        element for element in hyper_args.keys() if element not in "global"
    ]
    print(parameter_columns)
    reporter = CLIReporter(max_progress_rows=10, parameter_columns=parameter_columns)

    tune_result = tune.run(
        partial(tune_trainable, data_path=data_path),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=hyper_args,
        num_samples=args.hyper_samples,
        scheduler=scheduler,
        search_alg=search_algo,
        local_dir=local_dir,
        progress_reporter=reporter,
        verbose=1,
        resume=resume,
        name=args.job_name,
        max_failures=3,
        raise_on_failed_trial=False,
        stop={"training_iteration": args.epochs // 5},
    )

    best_trial = tune_result.get_best_trial("loss", "min", "last")

    return best_trial


# simple ensemble using averages
def train_ensemble(
    model_list,
    model_params,
    dataset,
    device,
    args,
    write="True",
    save="True",
    print_model="True",
):

    # Split datasets.
    train_dataset, val_dataset, test_dataset = process.split_data(
        dataset, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train_error = np.zeros(len(model_list))
    val_error = np.zeros(len(model_list))
    test_error = np.zeros(len(model_list))

    for i in range(0, len(model_list)):
        train_start = time.time()
        print("running on ", model_list[i])
        model = getattr(models, model_list[i])(
            data=dataset, **model_params[model_list[i]]
        ).to(device)
        # model_summary(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.8,
            patience=10,
            min_lr=0.00001,
            threshold=0.0002,
        )

        if print_model == "True":
            model_summary(model)

        for epoch in range(1, args.epochs + 1):
            lr = scheduler.optimizer.param_groups[0]["lr"]
            train_error[i], train_out = train(
                epoch, model, optimizer, train_loader, args.loss, device
            )

            if len(val_dataset) != 0:
                val_error[i], val_out = validation(val_loader, model, args.loss, device)

            scheduler.step(train_error[i])
            if epoch % args.verbosity == 0:
                epoch_time = time.time() - train_start
                train_start = time.time()
                print(
                    "Epoch: {:04d}, Learning Rate: {:.5f}, Training Error: {:.5f}, Val Error: {:.5f}, Time(s): {:.5f}".format(
                        epoch, lr, train_error[i], val_error[i], epoch_time
                    )
                )

        if len(test_dataset) != 0:
            test_error[i], test_out = test(test_loader, model, args.loss, device)
        print("Test Error: {:.5f}".format(test_error[i]))
        if i == 0:
            test_total = test_out
        elif i > 0:
            test_total = np.column_stack((test_total, test_out[:, 2]))

    # average ensembling, takes the mean of the predictions
    ensemble_test = np.mean(np.array(test_total[:, 2:]).astype(np.float), axis=1)
    ensemble_test_error = getattr(F, args.loss)(
        torch.tensor(ensemble_test),
        torch.tensor(test_total[:, 1].astype(np.float)),
    )
    test_total = np.column_stack((test_total, ensemble_test))

    if write == "True":
        with open(str(args.job_name) + "_test_ensemble_outputs.csv", "w") as f:
            csvwriter = csv.writer(f)
            for i in range(0, len(test_total) + 1):
                if i == 0:
                    csvwriter.writerow(
                        [
                            "ids",
                            "target",
                        ]
                        + model_list
                        + ["ensemble"]
                    )
                elif i > 0:
                    csvwriter.writerow(test_total[i - 1, :])

    return ensemble_test_error


# repeat training for n times
def train_repeat(model_params, dataset, device, args, write="True", save="True"):
    trial_errors = []
    job_name = args.job_name
    for i in range(0, args.repeat_trials):
        print("Starting trial " + str(i))
        args.seed = np.random.randint(1, 1e6)
        args.job_name = job_name + "_" + str(i)
        if i == 0:
            training_errors = train_regular(
                model_params,
                dataset,
                device,
                args,
                load="False",
                print_model="True",
                save=save,
                write=write,
            )
        else:
            training_errors = train_regular(
                model_params,
                dataset,
                device,
                args,
                load="False",
                print_model="False",
                save=save,
                write=write,
            )
        print(
            "Train error:",
            training_errors[0],
            "Val error:",
            training_errors[1],
            "Test error:",
            training_errors[2],
        )
        trial_errors.append(list(training_errors))

    return trial_errors


# obtains features from graph in a trained model and analysis with tsne
def analysis(
    model_params,
    dataset,
    device,
    args,
    tsne_args,
    load="True",
    write="True",
    save="True",
):

    #imports
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    inputs = []

    def hook(module, input, output):
        inputs.append(input)

    if load == "False":
        args.train_ratio = 1
        args.val_ratio = 0
        args.test_ratio = 0
        training_errors = train_regular(
            model_params,
            dataset,
            device,
            args,
            write=write,
            save="True",
            load="False",
            print_model="False",
        )
        print("Train error:", training_errors[0])

    assert os.path.exists(str(args.job_name) + "_model.pth"), "saved model not found"
    model = torch.load(str(args.job_name) + "_model.pth")
    model_summary(model)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model.eval()
    model.lin1.register_forward_hook(hook)
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            output = model(data)

    print("Number of samples: ", len(inputs))
    inputs = [i for sub in inputs for i in sub]
    inputs = torch.cat(inputs)
    inputs = inputs.cpu().numpy()

    targets = dataset.data.y.numpy()

    # pca = PCA(n_components=2)
    # pca_out=pca.fit_transform(inputs)
    # print(pca_out.shape)
    # np.savetxt('pca.csv', pca_out, delimiter=',')
    # plt.scatter(pca_out[:,1],pca_out[:,0],c=targets,s=15)
    # plt.colorbar()
    # plt.show()
    # plt.clf()

    # perplexity=50,early_exaggeration=12, learning_rate=300,n_iter=5000,verbose=1,random_state=42
    tsne = TSNE(**tsne_args)
    tsne_out = tsne.fit_transform(inputs)
    rows = zip(
        dataset.data.structure_id,
        list(dataset.data.y.numpy()),
        list(tsne_out[:, 0]),
        list(tsne_out[:, 1]),
    )
    if save == "True":
        with open("tsne_" + args.job_name + ".csv", "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            for row in rows:
                writer.writerow(row)

    fig, ax = plt.subplots()
    main = plt.scatter(tsne_out[:, 1], tsne_out[:, 0], c=targets, s=3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(main, ax=ax)
    stdev = np.std(targets)
    cbar.mappable.set_clim(
        np.mean(targets) - 2 * np.std(targets), np.mean(targets) + 2 * np.std(targets)
    )
    # cbar.ax.tick_params(labelsize=50)
    # cbar.ax.tick_params(size=40)
    plt.savefig("tsne_" + args.job_name + ".png", format="png", dpi=600)
    plt.show()
