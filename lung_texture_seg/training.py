import pickle
import argparse
import os
import ast
import re
from random import Random
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import (
    roc_curve, 
    auc,
    confusion_matrix
)
from itertools import cycle
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
import monai.networks.nets as nets
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    EnsureType
)
import datetime
import math
import time
import metrics
import itertools

from patch_dataset import PatchDataset

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()
    
    def get_elapsed_time(self):
        """Return the elapsed time, without stopping the timer"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")       
        elapsed_time = time.perf_counter() - self._start_time
        return elapsed_time


    def stop(self):
        """Stop the timer, and return the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return elapsed_time

def train(  train_directory,
            test_directory,
            output_dir,
            expected_labels, 
            num_images_train,
            fraction_images_val,
            num_images_test, 
            kernel_x,
            kernel_y,
            kernel_z,
            fill_fraction_train,
            fill_fraction_test,
            spacing_train,
            spacing_test,
            dim_string,
            deterministic_training=True, 
            max_epochs=10000, 
            train_batch_size=1000,
            val_batch_size=1000,
            test_batch_size=1000,
            val_interval=20,
            num_workers=8, # was 32
            checkpoint=None,
            labels_to_change=None,
            texture_names=None,
            early_stopping=None,
            experiment_name=None,
            device_name="cuda:0",
            model_name="DenseNet121",
            model_named_hp = { 
                "spatial_dims":2, 
                "in_channels":1,
                "out_channels":6,
            }
            ):

    print(f"train running with the following args: \n \
    {train_directory} \n \
    {test_directory} \n \
    {output_dir} \n \
    {str(expected_labels)} \n \
    {str(num_images_train)} \n \
    {str(fraction_images_val)} \n \
    {str(num_images_test)} \n \
    {str(kernel_x)} \n \
    {str(kernel_y)} \n \
    {str(kernel_z)} \n \
    {str(fill_fraction_train)} \n \
    {str(fill_fraction_test)} \n \
    {str(spacing_train)} \n \
    {str(spacing_test)} \n \
    {dim_string} \n \
    {deterministic_training} \n \
    {str(max_epochs)} \n \
    {str(train_batch_size)} \n \
    {str(val_batch_size)} \n \
    {str(test_batch_size)} \n \
    {str(val_interval)} \n \
    {str(num_workers)} \n \
    {checkpoint} \n \
    {str(labels_to_change)} \n \
    {str(texture_names)} \n \
    {early_stopping} \n \
    {experiment_name} \n \
    {device_name} \n \
    {model_name} \n \
    {str(model_named_hp)}")

    overall_timer = Timer()
    overall_timer.start()
    if deterministic_training:
        rand= Random(12345)
    else:
        rand = Random()

    now = datetime.datetime.now().strftime('%d%m%Y_%H%M%S') # ddmmyyyy_HHMMSS

    if experiment_name == None:
        part = '-'.join([str(l) for l in expected_labels])
        experiment_name = f'Tr{str(num_images_train)}_Ts{str(fraction_images_val)}_Val{str(num_images_test)}_X{str(kernel_x)}_Y{str(kernel_y)}_Z{str(kernel_z)}_ff{str(fill_fraction_train)}_s{spacing_train}_labs{part}_{model_name}_{dim_string}'

    # create the output directories
    experiment_dir = os.path.join(output_dir, experiment_name)
    example_image_dir = os.path.join(experiment_dir, 'patch_examples')

    os.makedirs(experiment_dir,exist_ok=True)
    os.makedirs(example_image_dir, exist_ok=True)
    num_class = len(expected_labels)

    # 1. Read all of our pickles and assemble into one big list
    # data_list_train = read_pickles(train_directory, pickle_prefix_train, expected_labels, num_images_train, labels_to_change, rand)
    data_list_train = read_pickles2(train_directory, dim_string, kernel_x, kernel_y, kernel_z, fill_fraction_train, spacing_train, expected_labels, num_images_train, labels_to_change, rand)
    num_total = len(data_list_train)
    num_val = int(num_total * fraction_images_val)

    # data_list_val = read_pickles(validate_directory, pickle_prefix_val, expected_labels, num_images_val, labels_to_change, rand)
    data_list_val = data_list_train[-num_val:]
    data_list_train = data_list_train[:-num_val]
    # data_list_test = read_pickles(test_directory, pickle_prefix_test, expected_labels, num_images_test, labels_to_change, rand)
    data_list_test = read_pickles2(test_directory, dim_string, kernel_x, kernel_y, kernel_z, fill_fraction_test, spacing_test, expected_labels, num_images_test, labels_to_change, rand)
    data_list_test = data_list_test[:num_images_test*num_class]
    

    if labels_to_change:
        for (orig, new) in labels_to_change:
            for i, e in enumerate(expected_labels):
                if e == orig:
                    expected_labels[i]=new 

    # 2. Randomly pick images from the dataset to visualize and check
    for data_list, cohort in zip([data_list_train, data_list_val, data_list_test], ['train', 'val', 'test']):
        display_patch(data_list, cohort, texture_names, expected_labels, example_image_dir, now)

    print(
        f"Training count: {len(data_list_train)}, Validation count: "
        f"{len(data_list_val)}, Test count: {len(data_list_test)}")

    # 3. Create transforms
    train_transforms = Compose(
        [
            AddChannel(),
            ScaleIntensity(),   # not sure about this
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            EnsureType(),
        ]
    )

    val_transforms = Compose(
        [ AddChannel(), ScaleIntensity(), EnsureType()])

    y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
    y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=num_class)])

    # pre load data:
    train_ds = PatchDataset(data_list_train, train_transforms, train=True)
    train_loader = torch.utils.data.DataLoader( train_ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

    val_ds = PatchDataset(data_list_val, val_transforms, train=True)
    val_loader = torch.utils.data.DataLoader( val_ds, batch_size=val_batch_size, shuffle=True, num_workers=num_workers)

    test_ds = PatchDataset(data_list_test, val_transforms, train=True)
    test_loader = torch.utils.data.DataLoader( test_ds, batch_size=test_batch_size, shuffle=True, num_workers=num_workers)

    # define network and optimiser
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, device, model_named_hp)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    # load pre trained weights if they exist:
    if checkpoint:
        if os.path.exists(os.path.join(experiment_dir, checkpoint)):
            model.load_state_dict(torch.load(os.path.join(experiment_dir, checkpoint)))

    auc_metric = ROCAUCMetric(average="none")

    # model training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    epochs_without_improvement = 0
    loading_time = overall_timer.get_elapsed_time()
    training_timer = Timer()
    training_timer.start()
    for epoch in range(max_epochs):
        if early_stopping and epochs_without_improvement> early_stopping:
            print("*" * 10)
            print(f"Early stopping specified and there has been no improvement within {early_stopping} epochs. Terminating training early")
            break
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                y_onehot = [y_trans(i) for i in decollate_batch(y)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate()
                mean_auc =  np.mean(result)
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)

                if mean_auc > best_metric:

                    epochs_without_improvement = -1  # reset the counter
                    best_metric = mean_auc
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        experiment_dir, "best_metric_model.pth"))
                    print(f"saved new best metric model at epoch {epoch + 1}")
                print(
                    f"current epoch: {epoch + 1} current AUC: {mean_auc:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
                for c, e in enumerate(expected_labels):
                    print(
                        f"AUC_{e} ({texture_names[c]}) = {result[c]:.4f} "
                    )

        epochs_without_improvement +=1

    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}") 

    # Plot the loss and metric
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val AUC")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig(os.path.join(experiment_dir, f'training_{now}.jpg'))
    plt.close()
    # plt.show()
    training_time = training_timer.stop()

    test_timer = Timer()
    test_timer.start()

    # first do inferences on the validation data
    val_mean_auc, val_acc_metric, val_result, val_y_onehot_arr, val_y_pred_onehot_arr = inference(val_loader, y_trans, y_pred_trans, model, device, os.path.join(experiment_dir, "best_metric_model.pth"), texture_names, expected_labels, os.path.join(experiment_dir, f'ROCAUC_validation_{now}.jpg'), os.path.join(experiment_dir, f'CM_validation_{now}.jpg'))


    # now we need to do inferences on the test_data
    mean_auc, acc_metric, result, y_onehot_arr, y_pred_onehot_arr = inference(test_loader, y_trans, y_pred_trans, model, device, os.path.join(experiment_dir, "best_metric_model.pth"), texture_names, expected_labels, os.path.join(experiment_dir, f'ROCAUC_test_{now}.jpg'), os.path.join(experiment_dir, f'CM_test_{now}.jpg'))
    testing_time = test_timer.stop()
    total_time = overall_timer.stop()

    # I need to save out a record of what was run through the experiment, what epoch we stopped, which was the best metric epoch etc.
    pd = {}
    pd['best_metric_epoch'] = best_metric_epoch
    pd['best_metric'] = best_metric
    pd['metric_values'] = metric_values
    pd['epoch_loss_values'] = epoch_loss_values
    pd['train_directory'] = train_directory
    pd['test_directory'] = test_directory
    pd['output_dir'] = output_dir
    pd['experiment_dir'] = experiment_dir
    # pd['pickle_prefix_train'] = pickle_prefix_train
    # pd['pickle_prefix_test'] = pickle_prefix_test
    pd['expected_labels'] = expected_labels
    pd['num_images_train'] = num_images_train
    pd['fraction_images_val'] = fraction_images_val
    pd['num_images_test'] = num_images_test
    pd['deterministic_training'] = deterministic_training
    pd['max_epochs'] = max_epochs
    pd['train_batch_size'] = train_batch_size
    pd['val_batch_size'] = val_batch_size
    pd['test_batch_size'] = test_batch_size
    pd['val_interval'] = val_interval
    pd['num_workers'] = num_workers
    pd['checkpoint'] = checkpoint
    pd['labels_to_change'] = labels_to_change
    pd['texture_names'] = texture_names
    pd['early_stopping'] = early_stopping
    pd['experiment_name'] = experiment_name
    pd['device_name'] = device_name
    pd['val_mean_auc'] = val_mean_auc
    pd['val_acc_metric'] = val_acc_metric
    pd['val_result'] = val_result
    pd['val_y_onehot_arr'] = val_y_onehot_arr
    pd['val_y_pred_onehot_arr'] = val_y_pred_onehot_arr
    pd['test_mean_auc'] = mean_auc
    pd['test_acc_metric'] = acc_metric
    pd['test_result'] = result
    pd['y_onehot_arr'] = y_onehot_arr
    pd['y_pred_onehot_arr'] = y_pred_onehot_arr
    pd['loading_time'] = loading_time
    pd['training_time'] = training_time
    pd['testing_time'] = testing_time
    pd['total_time'] = total_time


    with open(os.path.join(experiment_dir, "experiment_config.pickle"), 'wb') as f:
        pickle.dump(pd, f, protocol=pickle.HIGHEST_PROTOCOL)

def train_independent_val(  
            train_directory,
            val_directory,
            test_directory,
            output_dir,
            # pickle_prefix_train,
            # pickle_prefix_test,
            expected_labels, 
            num_images_train,
            num_images_val,
            num_images_test, 
            kernel_x,
            kernel_y,
            kernel_z,
            fill_fraction_train,
            fill_fraction_test,
            spacing_train,
            spacing_test,
            dim_string,
            deterministic_training=True, 
            max_epochs=10000, 
            train_batch_size=1000,
            val_batch_size=1000,
            test_batch_size=1000,
            val_interval=20,
            num_workers=8, # was 32
            checkpoint=None,
            labels_to_change=None,
            texture_names=None,
            early_stopping=None,
            experiment_name=None,
            device_name="cuda:0",
            model_name="DenseNet121",
            model_named_hp = { 
                "spatial_dims":2, 
                "in_channels":1,
                "out_channels":6,
            }
            ):

    print(f"train_independent_val running with the following args: \n \
    {train_directory} \n \
    {val_directory} \n \
    {test_directory} \n \
    {output_dir} \n \
    {str(expected_labels)} \n \
    {str(num_images_train)} \n \
    {str(num_images_val)} \n \
    {str(num_images_test)} \n \
    {str(kernel_x)} \n \
    {str(kernel_y)} \n \
    {str(kernel_z)} \n \
    {str(fill_fraction_train)} \n \
    {str(fill_fraction_test)} \n \
    {str(spacing_train)} \n \
    {str(spacing_test)} \n \
    {dim_string} \n \
    {deterministic_training} \n \
    {str(max_epochs)} \n \
    {str(train_batch_size)} \n \
    {str(val_batch_size)} \n \
    {str(test_batch_size)} \n \
    {str(val_interval)} \n \
    {str(num_workers)} \n \
    {checkpoint} \n \
    {str(labels_to_change)} \n \
    {str(texture_names)} \n \
    {early_stopping} \n \
    {experiment_name} \n \
    {device_name} \n \
    {model_name} \n \
    {str(model_named_hp)}")

    overall_timer = Timer()
    overall_timer.start()
    if deterministic_training:
        rand= Random(12345)
    else:
        rand = Random()

    now = datetime.datetime.now().strftime('%d%m%Y_%H%M%S') # ddmmyyyy_HHMMSS

    if experiment_name == None:
        part = '-'.join([str(l) for l in expected_labels])
        experiment_name = f'Tr{str(num_images_train)}_Ts{str(num_images_test)}_Val{str(num_images_val)}_X{str(kernel_x)}_Y{str(kernel_y)}_Z{str(kernel_z)}_ff{str(fill_fraction_train)}_s{spacing_train}_labs{part}_{model_name}_{dim_string}'

    # create the output directories
    experiment_dir = os.path.join(output_dir, experiment_name)
    example_image_dir = os.path.join(experiment_dir, 'patch_examples')

    os.makedirs(experiment_dir,exist_ok=True)
    os.makedirs(example_image_dir, exist_ok=True)
    num_class = len(expected_labels)

    # 1. Read all of our pickles and assemble into one big list
    # data_list_train = read_pickles(train_directory, pickle_prefix_train, expected_labels, num_images_train, labels_to_change, rand)
    data_list_train = read_pickles2(train_directory, dim_string, kernel_x, kernel_y, kernel_z, fill_fraction_train, spacing_train, expected_labels, num_images_train, labels_to_change, rand)
    data_list_val = read_pickles2(val_directory, dim_string, kernel_x, kernel_y, kernel_z, fill_fraction_train, spacing_train, expected_labels, num_images_val, labels_to_change, rand)
    # num_total = len(data_list_train)
    # num_val = int(num_total * fraction_images_val)

    # # data_list_val = read_pickles(validate_directory, pickle_prefix_val, expected_labels, num_images_val, labels_to_change, rand)
    # data_list_val = data_list_train[-num_val:]
    # data_list_train = data_list_train[:-num_val]
    # data_list_test = read_pickles(test_directory, pickle_prefix_test, expected_labels, num_images_test, labels_to_change, rand)
    data_list_test = read_pickles2(test_directory, dim_string, kernel_x, kernel_y, kernel_z, fill_fraction_test, spacing_test, expected_labels, num_images_test, labels_to_change, rand)
    # data_list_test = data_list_test[:num_images_test*num_class]
    

    if labels_to_change:
        for (orig, new) in labels_to_change:
            for i, e in enumerate(expected_labels):
                if e == orig:
                    expected_labels[i]=new 

    # 2. Randomly pick images from the dataset to visualize and check
    for data_list, cohort in zip([data_list_train, data_list_val, data_list_test], ['train', 'val', 'test']):
        display_patch(data_list, cohort, texture_names, expected_labels, example_image_dir, now)
    #     num_patchs = len(data_list)
    #     plt.subplots(3, 3, figsize=(8, 8))
    #     for i in range(3):
    #         for c, k in enumerate(np.random.randint(num_patchs, size=9)):
    #             arr = data_list[k]['patch']
    #             plt.subplot(3, 3, c + 1)
    #             plt.xlabel(f"Class = {texture_names[expected_labels.index(data_list[k]['classification'])]} ({data_list[k]['classification']})")
    #             plt.imshow(arr, cmap="gray", vmin=-1024, vmax=300, interpolation='bilinear')
    #             plt.xticks([])
    #             plt.yticks([])
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(example_image_dir, f'{cohort}_sample_{now}.jpeg'))
    #         # plt.show()
    #         plt.cla()
    # plt.close()
    print(
        f"Training count: {len(data_list_train)}, Validation count: "
        f"{len(data_list_val)}, Test count: {len(data_list_test)}")

    # 3. Create transforms
    train_transforms = Compose(
        [
            AddChannel(),
            ScaleIntensity(),   # not sure about this
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            EnsureType(),
        ]
    )

    val_transforms = Compose(
        [ AddChannel(), ScaleIntensity(), EnsureType()])

    y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
    y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=num_class)])

    # pre load data:
    train_ds = PatchDataset(data_list_train, train_transforms, train=True)
    train_loader = torch.utils.data.DataLoader( train_ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

    val_ds = PatchDataset(data_list_val, val_transforms, train=True)
    val_loader = torch.utils.data.DataLoader( val_ds, batch_size=val_batch_size, shuffle=True, num_workers=num_workers)

    test_ds = PatchDataset(data_list_test, val_transforms, train=True)
    test_loader = torch.utils.data.DataLoader( test_ds, batch_size=test_batch_size, shuffle=True, num_workers=num_workers)

    # define network and optimiser
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, device, model_named_hp)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    # load pre trained weights if they exist:
    if checkpoint:
        if os.path.exists(os.path.join(experiment_dir, checkpoint)):
            model.load_state_dict(torch.load(os.path.join(experiment_dir, checkpoint)))

    auc_metric = ROCAUCMetric(average="none")

    # model training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    epochs_without_improvement = 0
    loading_time = overall_timer.get_elapsed_time()
    training_timer = Timer()
    training_timer.start()
    for epoch in range(max_epochs):
        if early_stopping and epochs_without_improvement> early_stopping:
            print("*" * 10)
            print(f"Early stopping specified and there has been no improvement within {early_stopping} epochs. Terminating training early")
            break
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print(
            #     f"{step}/{len(train_ds) // train_loader.batch_size}, "
            #     f"train_loss: {loss.item():.4f}")
            # epoch_len = len(train_ds) // train_loader.batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                y_onehot = [y_trans(i) for i in decollate_batch(y)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate()
                mean_auc =  np.mean(result)
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)

                if mean_auc > best_metric:

                    epochs_without_improvement = -1  # reset the counter
                    best_metric = mean_auc
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        experiment_dir, "best_metric_model.pth"))
                    print(f"saved new best metric model at epoch {epoch + 1}")
                print(
                    f"current epoch: {epoch + 1} current AUC: {mean_auc:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
                for c, e in enumerate(expected_labels):
                    print(
                        f"AUC_{e} ({texture_names[c]}) = {result[c]:.4f} "
                    )

        epochs_without_improvement +=1

    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}") 

    # Plot the loss and metric
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val AUC")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig(os.path.join(experiment_dir, f'training_{now}.jpg'))
    plt.close()
    # plt.show()
    training_time = training_timer.stop()

    test_timer = Timer()
    test_timer.start()
    # first do inferences on the validation data
    val_mean_auc, val_acc_metric, val_result, val_y_onehot_arr, val_y_pred_onehot_arr = inference(val_loader, y_trans, y_pred_trans, model, device, os.path.join(experiment_dir, "best_metric_model.pth"), texture_names, expected_labels, os.path.join(experiment_dir, f'ROCAUC_validation_{now}.jpg'), os.path.join(experiment_dir, f'CM_validation_{now}.jpg'))


    # now we need to do inferences on the test_data
    mean_auc, acc_metric, result, y_onehot_arr, y_pred_onehot_arr = inference(test_loader, y_trans, y_pred_trans, model, device, os.path.join(experiment_dir, "best_metric_model.pth"), texture_names, expected_labels, os.path.join(experiment_dir, f'ROCAUC_test_{now}.jpg'), os.path.join(experiment_dir, f'CM_test_{now}.jpg'))
    testing_time = test_timer.stop()
    total_time = overall_timer.stop()

    # I need to save out a record of what was run through the experiment, what epoch we stopped, which was the best metric epoch etc.
    pd = {}
    pd['best_metric_epoch'] = best_metric_epoch
    pd['best_metric'] = best_metric
    pd['metric_values'] = metric_values
    pd['epoch_loss_values'] = epoch_loss_values
    pd['train_directory'] = train_directory
    pd['val_directory'] = val_directory
    pd['test_directory'] = test_directory
    pd['output_dir'] = output_dir
    pd['experiment_dir'] = experiment_dir
    # pd['pickle_prefix_train'] = pickle_prefix_train
    # pd['pickle_prefix_test'] = pickle_prefix_test
    pd['expected_labels'] = expected_labels
    pd['num_images_train'] = num_images_train
    pd['num_images_val'] = num_images_val
    pd['num_images_test'] = num_images_test
    pd['deterministic_training'] = deterministic_training
    pd['max_epochs'] = max_epochs
    pd['train_batch_size'] = train_batch_size
    pd['val_batch_size'] = val_batch_size
    pd['test_batch_size'] = test_batch_size
    pd['val_interval'] = val_interval
    pd['num_workers'] = num_workers
    pd['checkpoint'] = checkpoint
    pd['labels_to_change'] = labels_to_change
    pd['texture_names'] = texture_names
    pd['early_stopping'] = early_stopping
    pd['experiment_name'] = experiment_name
    pd['device_name'] = device_name
    pd['val_mean_auc'] = val_mean_auc
    pd['val_acc_metric'] = val_acc_metric
    pd['val_result'] = val_result
    pd['val_y_onehot_arr'] = val_y_onehot_arr
    pd['val_y_pred_onehot_arr'] = val_y_pred_onehot_arr
    pd['test_mean_auc'] = mean_auc
    pd['test_acc_metric'] = acc_metric
    pd['test_result'] = result
    pd['y_onehot_arr'] = y_onehot_arr
    pd['y_pred_onehot_arr'] = y_pred_onehot_arr
    pd['loading_time'] = loading_time
    pd['training_time'] = training_time
    pd['testing_time'] = testing_time
    pd['total_time'] = total_time


    with open(os.path.join(experiment_dir, "experiment_config.pickle"), 'wb') as f:
        pickle.dump(pd, f, protocol=pickle.HIGHEST_PROTOCOL)


def display_patch(data_list, cohort, texture_names, expected_labels, example_image_dir, current_time):
    num_patchs = len(data_list)
    plt.subplots(3, 3, figsize=(8, 8))
    for i in range(3):
        for c, k in enumerate(np.random.randint(num_patchs, size=9)):
            arr = data_list[k]['patch']
            if len(arr.shape) ==3 and arr.shape[0] !=3:
                arr = create_composite(arr)
            elif len(arr.shape) ==3 and arr.shape[0] ==3:
                arr = create_composite_bodge(arr)
            plt.subplot(3, 3, c + 1)
            plt.xlabel(f"Class = {texture_names[expected_labels.index(data_list[k]['classification'])]} ({data_list[k]['classification']})")
            plt.imshow(arr, cmap="gray", vmin=-1024, vmax=300, interpolation='bilinear')
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(example_image_dir, f'{cohort}_sample_{current_time}.jpeg'))
        # plt.show()
        plt.cla()
    plt.close()

def create_composite(arr : np.ndarray):
    x_dim, y_dim, num_panels = arr.shape
    x_pan, y_pan = calc_panels(num_panels)
    composite = np.zeros((x_dim * x_pan, y_dim * y_pan))
    current_pannel = 0
    for x in range(x_pan):
        for y in range(y_pan):
            if current_pannel >= num_panels:
                break
            x_loc = x_dim * x
            y_loc = y_dim * y
            composite[x_loc:x_loc+x_dim, y_loc:y_loc+y_dim] = arr[:,:,current_pannel]
            current_pannel +=1
    return composite

def create_composite_bodge(arr : np.ndarray):
    # this is because I made a mistake with the 2.5D patches they are z x, y instead of x,y,z
    arr = np.moveaxis(arr, 0, 2)
    x_dim, y_dim, num_panels = arr.shape
    x_pan, y_pan = calc_panels(num_panels)
    composite = np.zeros((x_dim * x_pan, y_dim * y_pan))
    current_pannel = 0
    for x in range(x_pan):
        for y in range(y_pan):
            if current_pannel >= num_panels:
                break
            x_loc = x_dim * x
            y_loc = y_dim * y
            composite[x_loc:x_loc+x_dim, y_loc:y_loc+y_dim] = arr[:,:,current_pannel]
            current_pannel +=1
    return composite

def calc_panels(x):
    root = math.sqrt(x)
    round_up = math.ceil(root)
    round_down = math.floor(root)
    if round_up * round_down > x:
        return round_down, round_up
    return round_up, round_up


def test_only(  train_directory,
            test_directory,
            output_dir,
            # pickle_prefix_train,
            # pickle_prefix_test,
            expected_labels, 
            num_images_train,
            fraction_images_val,
            num_images_test, 
            kernel_x,
            kernel_y,
            kernel_z,
            fill_fraction_train,
            fill_fraction_test,
            spacing_train,
            spacing_test,
            dim_string,
            deterministic_training=True, 
            max_epochs=10000, 
            train_batch_size=1000,
            val_batch_size=1000,
            test_batch_size=1000,
            val_interval=20,
            num_workers=8, # was 32
            checkpoint=None,
            labels_to_change=None,
            texture_names=None,
            early_stopping=None,
            experiment_name=None,
            device_name="cuda:0",
            model_name="DenseNet121",
            model_named_hp = { 
                "spatial_dims":2, 
                "in_channels":1,
                "out_channels":6,
            }
            ):

    print(f"train running with the following args: \n \
    {train_directory} \n \
    {test_directory} \n \
    {output_dir} \n \
    {str(expected_labels)} \n \
    {str(num_images_train)} \n \
    {str(fraction_images_val)} \n \
    {str(num_images_test)} \n \
    {str(kernel_x)} \n \
    {str(kernel_y)} \n \
    {str(kernel_z)} \n \
    {str(fill_fraction_train)} \n \
    {str(fill_fraction_test)} \n \
    {str(spacing_train)} \n \
    {str(spacing_test)} \n \
    {dim_string} \n \
    {deterministic_training} \n \
    {str(max_epochs)} \n \
    {str(train_batch_size)} \n \
    {str(val_batch_size)} \n \
    {str(test_batch_size)} \n \
    {str(val_interval)} \n \
    {str(num_workers)} \n \
    {checkpoint} \n \
    {str(labels_to_change)} \n \
    {str(texture_names)} \n \
    {early_stopping} \n \
    {experiment_name} \n \
    {device_name} \n \
    {model_name} \n \
    {str(model_named_hp)}")

    overall_timer = Timer()
    overall_timer.start()
    if deterministic_training:
        rand= Random(12345)
    else:
        rand = Random()

    now = datetime.datetime.now().strftime('%d%m%Y_%H%M%S') # ddmmyyyy_HHMMSS

    if experiment_name == None:
        # part = '-'.join([str(l) for l in expected_labels])
        # experiment_name = f'Tr{str(num_images_train)}_Ts{str(fraction_images_val)}_Val{str(num_images_test)}_X{str(kernel_x)}_Y{str(kernel_y)}_Z{str(kernel_z)}_ff{str(fill_fraction_train)}_labs{part}_{model_name}_{dim_string}'
        part = '-'.join([str(l) for l in expected_labels])
        experiment_name = f'Tr{str(num_images_train)}_Ts{str(fraction_images_val)}_Val{str(num_images_test)}_X{str(kernel_x)}_Y{str(kernel_y)}_Z{str(kernel_z)}_ff{str(fill_fraction_train)}_s{spacing_train}_labs{part}_{model_name}_{dim_string}'

    # create the output directories
    experiment_dir = os.path.join(output_dir, 'Exp1_fill_factor', experiment_name)
    example_image_dir = os.path.join(experiment_dir, 'patch_examples')

    os.makedirs(experiment_dir,exist_ok=True)
    os.makedirs(example_image_dir, exist_ok=True)
    num_class = len(expected_labels)

    # 1. Read all of our pickles and assemble into one big list
    # data_list_train = read_pickles(train_directory, pickle_prefix_train, expected_labels, num_images_train, labels_to_change, rand)
    # data_list_train = read_pickles2(train_directory, dim_string, kernel_x, kernel_y, kernel_z, fill_fraction_train, spacing_train, expected_labels, num_images_train, labels_to_change, rand)
    # num_total = len(data_list_train)
    # num_val = int(num_total * fraction_images_val)

    # # data_list_val = read_pickles(validate_directory, pickle_prefix_val, expected_labels, num_images_val, labels_to_change, rand)
    # data_list_val = data_list_train[-num_val:]
    # data_list_train = data_list_train[:-num_val]
    # data_list_test = read_pickles(test_directory, pickle_prefix_test, expected_labels, num_images_test, labels_to_change, rand)
    data_list_test = read_pickles2(test_directory, dim_string, kernel_x, kernel_y, kernel_z, fill_fraction_test, spacing_test, expected_labels, num_images_test, labels_to_change, rand)
    data_list_test = data_list_test[:num_images_test*num_class]
    

    if labels_to_change:
        for (orig, new) in labels_to_change:
            for i, e in enumerate(expected_labels):
                if e == orig:
                    expected_labels[i]=new 

    # 2. Randomly pick images from the dataset to visualize and check
    # for data_list, cohort in zip([data_list_train, data_list_val, data_list_test], ['train', 'val', 'test']):
    #     num_patchs = len(data_list)
    #     plt.subplots(3, 3, figsize=(8, 8))
    #     for i in range(3):
    #         for c, k in enumerate(np.random.randint(num_patchs, size=9)):
    #             arr = data_list[k]['patch']
    #             plt.subplot(3, 3, c + 1)
    #             plt.xlabel(f"Class = {texture_names[expected_labels.index(data_list[k]['classification'])]} ({data_list[k]['classification']})")
    #             plt.imshow(arr, cmap="gray", vmin=-1024, vmax=300, interpolation='bilinear')
    #             plt.xticks([])
    #             plt.yticks([])
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(example_image_dir, f'{cohort}_sample_{now}.jpeg'))
    #         # plt.show()
    #         plt.cla()
    # plt.close()
    # print(
    #     f"Training count: {len(data_list_train)}, Validation count: "
    #     f"{len(data_list_val)}, Test count: {len(data_list_test)}")
    print(
        f" Test count: {len(data_list_test)}")
    # 3. Create transforms
    train_transforms = Compose(
        [
            AddChannel(),
            ScaleIntensity(),   # not sure about this
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            EnsureType(),
        ]
    )

    val_transforms = Compose(
        [ AddChannel(), ScaleIntensity(), EnsureType()])

    y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
    y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=num_class)])

    # pre load data:
    # train_ds = PatchDataset(data_list_train, train_transforms, train=True)
    # train_loader = torch.utils.data.DataLoader( train_ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

    # val_ds = PatchDataset(data_list_val, val_transforms, train=True)
    # val_loader = torch.utils.data.DataLoader( val_ds, batch_size=val_batch_size, shuffle=True, num_workers=num_workers)

    test_ds = PatchDataset(data_list_test, val_transforms, train=True)
    test_loader = torch.utils.data.DataLoader( test_ds, batch_size=test_batch_size, shuffle=True, num_workers=num_workers)

    # define network and optimiser
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, device, model_named_hp)
    # loss_function = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    # load pre trained weights if they exist:
    # if checkpoint:
    #     if os.path.exists(os.path.join(experiment_dir, checkpoint)):
    #         model.load_state_dict(torch.load(os.path.join(experiment_dir, checkpoint)))

    # auc_metric = ROCAUCMetric(average="none")

    # # model training
    # best_metric = -1
    # best_metric_epoch = -1
    # epoch_loss_values = []
    # metric_values = []

    # epochs_without_improvement = 0
    # loading_time = overall_timer.get_elapsed_time()
    # training_timer = Timer()
    # training_timer.start()
    # for epoch in range(max_epochs):
    #     if early_stopping and epochs_without_improvement> early_stopping:
    #         print("*" * 10)
    #         print(f"Early stopping specified and there has been no improvement within {early_stopping} epochs. Terminating training early")
    #         break
    #     print("-" * 10)
    #     print(f"epoch {epoch + 1}/{max_epochs}")
    #     model.train()
    #     epoch_loss = 0
    #     step = 0
    #     for batch_data in train_loader:
    #         step += 1
    #         inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = loss_function(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item()
    #         # print(
    #         #     f"{step}/{len(train_ds) // train_loader.batch_size}, "
    #         #     f"train_loss: {loss.item():.4f}")
    #         # epoch_len = len(train_ds) // train_loader.batch_size
    #     epoch_loss /= step
    #     epoch_loss_values.append(epoch_loss)
    #     print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    #     if (epoch + 1) % val_interval == 0:
    #         model.eval()
    #         with torch.no_grad():
    #             y_pred = torch.tensor([], dtype=torch.float32, device=device)
    #             y = torch.tensor([], dtype=torch.long, device=device)
    #             for val_data in val_loader:
    #                 val_images, val_labels = (
    #                     val_data[0].to(device),
    #                     val_data[1].to(device),
    #                 )
    #                 y_pred = torch.cat([y_pred, model(val_images)], dim=0)
    #                 y = torch.cat([y, val_labels], dim=0)
    #             y_onehot = [y_trans(i) for i in decollate_batch(y)]
    #             y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
    #             auc_metric(y_pred_act, y_onehot)
    #             result = auc_metric.aggregate()
    #             mean_auc =  np.mean(result)
    #             auc_metric.reset()
    #             del y_pred_act, y_onehot
    #             metric_values.append(result)
    #             acc_value = torch.eq(y_pred.argmax(dim=1), y)
    #             acc_metric = acc_value.sum().item() / len(acc_value)

    #             if mean_auc > best_metric:

    #                 epochs_without_improvement = -1  # reset the counter
    #                 best_metric = mean_auc
    #                 best_metric_epoch = epoch + 1
    #                 torch.save(model.state_dict(), os.path.join(
    #                     experiment_dir, "best_metric_model.pth"))
    #                 print(f"saved new best metric model at epoch {epoch + 1}")
    #             print(
    #                 f"current epoch: {epoch + 1} current AUC: {mean_auc:.4f}"
    #                 f" current accuracy: {acc_metric:.4f}"
    #                 f" best AUC: {best_metric:.4f}"
    #                 f" at epoch: {best_metric_epoch}"
    #             )
    #             for c, e in enumerate(expected_labels):
    #                 print(
    #                     f"AUC_{e} ({texture_names[c]}) = {result[c]:.4f} "
    #                 )

    #     epochs_without_improvement +=1

    # print(
    #     f"train completed, best_metric: {best_metric:.4f} "
    #     f"at epoch: {best_metric_epoch}") 

    # # Plot the loss and metric
    # plt.figure("train", (12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("Epoch Average Loss")
    # x = [i + 1 for i in range(len(epoch_loss_values))]
    # y = epoch_loss_values
    # plt.xlabel("epoch")
    # plt.plot(x, y)
    # plt.subplot(1, 2, 2)
    # plt.title("Val AUC")
    # x = [val_interval * (i + 1) for i in range(len(metric_values))]
    # y = metric_values
    # plt.xlabel("epoch")
    # plt.plot(x, y)
    # plt.savefig(os.path.join(experiment_dir, f'training_{now}.jpg'))
    # plt.show()
    # training_time = training_timer.stop()

    test_timer = Timer()
    test_timer.start()
    # now we need to do inferences on the test_data
    mean_auc, acc_metric, result, y_onehot_arr, y_pred_onehot_arr = inference(test_loader, y_trans, y_pred_trans, model, device, os.path.join(experiment_dir, "best_metric_model.pth"), texture_names, expected_labels, os.path.join(experiment_dir, f'ROCAUC_repeat_{now}.png'), os.path.join(experiment_dir, f'CM_{now}.png'))
    testing_time = test_timer.stop()
    total_time = overall_timer.stop()

    # I need to save out a record of what was run through the experiment, what epoch we stopped, which was the best metric epoch etc.
    pd = {}
    # pd['best_metric_epoch'] = best_metric_epoch
    # pd['best_metric'] = best_metric
    # pd['metric_values'] = metric_values
    # pd['epoch_loss_values'] = epoch_loss_values
    pd['train_directory'] = train_directory
    pd['test_directory'] = test_directory
    pd['output_dir'] = output_dir
    pd['experiment_dir'] = experiment_dir
    # pd['pickle_prefix_train'] = pickle_prefix_train
    # pd['pickle_prefix_test'] = pickle_prefix_test
    pd['expected_labels'] = expected_labels
    pd['num_images_train'] = num_images_train
    pd['fraction_images_val'] = fraction_images_val
    pd['num_images_test'] = num_images_test
    pd['deterministic_training'] = deterministic_training
    pd['max_epochs'] = max_epochs
    pd['train_batch_size'] = train_batch_size
    pd['val_batch_size'] = val_batch_size
    pd['test_batch_size'] = test_batch_size
    pd['val_interval'] = val_interval
    pd['num_workers'] = num_workers
    pd['checkpoint'] = checkpoint
    pd['labels_to_change'] = labels_to_change
    pd['texture_names'] = texture_names
    pd['early_stopping'] = early_stopping
    pd['experiment_name'] = experiment_name
    pd['device_name'] = device_name
    pd['test_mean_auc'] = mean_auc
    pd['test_acc_metric'] = acc_metric
    pd['test_result'] = result
    pd['y_onehot_arr'] = y_onehot_arr
    pd['y_pred_onehot_arr'] = y_pred_onehot_arr
    # pd['loading_time'] = loading_time
    # pd['training_time'] = training_time
    pd['testing_time'] = testing_time
    pd['total_time'] = total_time


    with open(os.path.join(experiment_dir, "experiment_config_repeat.pickle"), 'wb') as f:
        pickle.dump(pd, f, protocol=pickle.HIGHEST_PROTOCOL)

def test_only_with_validate_change(  
            train_directory,
            val_directory,
            test_directory,
            output_dir,
            # pickle_prefix_train,
            # pickle_prefix_test,
            expected_labels, 
            num_images_train,
            num_images_val,
            num_images_test, 
            kernel_x,
            kernel_y,
            kernel_z,
            fill_fraction_train,
            fill_fraction_test,
            spacing_train,
            spacing_test,
            dim_string,
            deterministic_training=True, 
            max_epochs=10000, 
            train_batch_size=1000,
            val_batch_size=1000,
            test_batch_size=1000,
            val_interval=20,
            num_workers=8, # was 32
            checkpoint=None,
            labels_to_change=None,
            texture_names=None,
            early_stopping=None,
            experiment_name=None,
            device_name="cuda:0",
            model_name="DenseNet121",
            model_named_hp = { 
                "spatial_dims":2, 
                "in_channels":1,
                "out_channels":6,
            }
            ):

    print(f"train running with the following args: \n \
    {train_directory} \n \
    {test_directory} \n \
    {output_dir} \n \
    {str(expected_labels)} \n \
    {str(num_images_train)} \n \
    {str(num_images_val)} \n \
    {str(num_images_test)} \n \
    {str(kernel_x)} \n \
    {str(kernel_y)} \n \
    {str(kernel_z)} \n \
    {str(fill_fraction_train)} \n \
    {str(fill_fraction_test)} \n \
    {str(spacing_train)} \n \
    {str(spacing_test)} \n \
    {dim_string} \n \
    {deterministic_training} \n \
    {str(max_epochs)} \n \
    {str(train_batch_size)} \n \
    {str(val_batch_size)} \n \
    {str(test_batch_size)} \n \
    {str(val_interval)} \n \
    {str(num_workers)} \n \
    {checkpoint} \n \
    {str(labels_to_change)} \n \
    {str(texture_names)} \n \
    {early_stopping} \n \
    {experiment_name} \n \
    {device_name} \n \
    {model_name} \n \
    {str(model_named_hp)}")

    overall_timer = Timer()
    overall_timer.start()
    if deterministic_training:
        rand= Random(12345)
    else:
        rand = Random()

    now = datetime.datetime.now().strftime('%d%m%Y_%H%M%S') # ddmmyyyy_HHMMSS

    if experiment_name == None:
        # part = '-'.join([str(l) for l in expected_labels])
        # experiment_name = f'Tr{str(num_images_train)}_Ts{str(fraction_images_val)}_Val{str(num_images_test)}_X{str(kernel_x)}_Y{str(kernel_y)}_Z{str(kernel_z)}_ff{str(fill_fraction_train)}_labs{part}_{model_name}_{dim_string}'
        part = '-'.join([str(l) for l in expected_labels])
        experiment_name = f'Tr{str(num_images_train)}_Ts{str(num_images_test)}_Val{str(num_images_val)}_X{str(kernel_x)}_Y{str(kernel_y)}_Z{str(kernel_z)}_ff{str(fill_fraction_train)}_s{spacing_train}_labs{part}_{model_name}_{dim_string}'

    # create the output directories
    experiment_dir = os.path.join(output_dir, experiment_name)
    example_image_dir = os.path.join(experiment_dir, 'patch_examples')

    os.makedirs(experiment_dir,exist_ok=True)
    os.makedirs(example_image_dir, exist_ok=True)
    num_class = len(expected_labels)

    # 1. Read all of our pickles and assemble into one big list
    # data_list_train = read_pickles(train_directory, pickle_prefix_train, expected_labels, num_images_train, labels_to_change, rand)
    # data_list_train = read_pickles2(train_directory, dim_string, kernel_x, kernel_y, kernel_z, fill_fraction_train, spacing_train, expected_labels, num_images_train, labels_to_change, rand)
    # num_total = len(data_list_train)
    # num_val = int(num_total * fraction_images_val)

    # # data_list_val = read_pickles(validate_directory, pickle_prefix_val, expected_labels, num_images_val, labels_to_change, rand)
    # data_list_val = data_list_train[-num_val:]
    # data_list_train = data_list_train[:-num_val]
    # data_list_test = read_pickles(test_directory, pickle_prefix_test, expected_labels, num_images_test, labels_to_change, rand)
    data_list_test = read_pickles2(test_directory, dim_string, kernel_x, kernel_y, kernel_z, fill_fraction_test, spacing_test, expected_labels, num_images_test, labels_to_change, rand)
    data_list_test = data_list_test[:num_images_test*num_class]
    

    if labels_to_change:
        for (orig, new) in labels_to_change:
            for i, e in enumerate(expected_labels):
                if e == orig:
                    expected_labels[i]=new 

    # 2. Randomly pick images from the dataset to visualize and check
    # for data_list, cohort in zip([data_list_train, data_list_val, data_list_test], ['train', 'val', 'test']):
    #     num_patchs = len(data_list)
    #     plt.subplots(3, 3, figsize=(8, 8))
    #     for i in range(3):
    #         for c, k in enumerate(np.random.randint(num_patchs, size=9)):
    #             arr = data_list[k]['patch']
    #             plt.subplot(3, 3, c + 1)
    #             plt.xlabel(f"Class = {texture_names[expected_labels.index(data_list[k]['classification'])]} ({data_list[k]['classification']})")
    #             plt.imshow(arr, cmap="gray", vmin=-1024, vmax=300, interpolation='bilinear')
    #             plt.xticks([])
    #             plt.yticks([])
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(example_image_dir, f'{cohort}_sample_{now}.jpeg'))
    #         # plt.show()
    #         plt.cla()
    # plt.close()
    # print(
    #     f"Training count: {len(data_list_train)}, Validation count: "
    #     f"{len(data_list_val)}, Test count: {len(data_list_test)}")
    print(
        f" Test count: {len(data_list_test)}")
    # 3. Create transforms
    train_transforms = Compose(
        [
            AddChannel(),
            ScaleIntensity(),   # not sure about this
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            EnsureType(),
        ]
    )

    val_transforms = Compose(
        [ AddChannel(), ScaleIntensity(), EnsureType()])

    y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
    y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=num_class)])

    # pre load data:
    # train_ds = PatchDataset(data_list_train, train_transforms, train=True)
    # train_loader = torch.utils.data.DataLoader( train_ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

    # val_ds = PatchDataset(data_list_val, val_transforms, train=True)
    # val_loader = torch.utils.data.DataLoader( val_ds, batch_size=val_batch_size, shuffle=True, num_workers=num_workers)

    test_ds = PatchDataset(data_list_test, val_transforms, train=True)
    test_loader = torch.utils.data.DataLoader( test_ds, batch_size=test_batch_size, shuffle=True, num_workers=num_workers)

    # define network and optimiser
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, device, model_named_hp)
    # loss_function = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    # load pre trained weights if they exist:
    # if checkpoint:
    #     if os.path.exists(os.path.join(experiment_dir, checkpoint)):
    #         model.load_state_dict(torch.load(os.path.join(experiment_dir, checkpoint)))

    # auc_metric = ROCAUCMetric(average="none")

    # # model training
    # best_metric = -1
    # best_metric_epoch = -1
    # epoch_loss_values = []
    # metric_values = []

    # epochs_without_improvement = 0
    # loading_time = overall_timer.get_elapsed_time()
    # training_timer = Timer()
    # training_timer.start()
    # for epoch in range(max_epochs):
    #     if early_stopping and epochs_without_improvement> early_stopping:
    #         print("*" * 10)
    #         print(f"Early stopping specified and there has been no improvement within {early_stopping} epochs. Terminating training early")
    #         break
    #     print("-" * 10)
    #     print(f"epoch {epoch + 1}/{max_epochs}")
    #     model.train()
    #     epoch_loss = 0
    #     step = 0
    #     for batch_data in train_loader:
    #         step += 1
    #         inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = loss_function(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item()
    #         # print(
    #         #     f"{step}/{len(train_ds) // train_loader.batch_size}, "
    #         #     f"train_loss: {loss.item():.4f}")
    #         # epoch_len = len(train_ds) // train_loader.batch_size
    #     epoch_loss /= step
    #     epoch_loss_values.append(epoch_loss)
    #     print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    #     if (epoch + 1) % val_interval == 0:
    #         model.eval()
    #         with torch.no_grad():
    #             y_pred = torch.tensor([], dtype=torch.float32, device=device)
    #             y = torch.tensor([], dtype=torch.long, device=device)
    #             for val_data in val_loader:
    #                 val_images, val_labels = (
    #                     val_data[0].to(device),
    #                     val_data[1].to(device),
    #                 )
    #                 y_pred = torch.cat([y_pred, model(val_images)], dim=0)
    #                 y = torch.cat([y, val_labels], dim=0)
    #             y_onehot = [y_trans(i) for i in decollate_batch(y)]
    #             y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
    #             auc_metric(y_pred_act, y_onehot)
    #             result = auc_metric.aggregate()
    #             mean_auc =  np.mean(result)
    #             auc_metric.reset()
    #             del y_pred_act, y_onehot
    #             metric_values.append(result)
    #             acc_value = torch.eq(y_pred.argmax(dim=1), y)
    #             acc_metric = acc_value.sum().item() / len(acc_value)

    #             if mean_auc > best_metric:

    #                 epochs_without_improvement = -1  # reset the counter
    #                 best_metric = mean_auc
    #                 best_metric_epoch = epoch + 1
    #                 torch.save(model.state_dict(), os.path.join(
    #                     experiment_dir, "best_metric_model.pth"))
    #                 print(f"saved new best metric model at epoch {epoch + 1}")
    #             print(
    #                 f"current epoch: {epoch + 1} current AUC: {mean_auc:.4f}"
    #                 f" current accuracy: {acc_metric:.4f}"
    #                 f" best AUC: {best_metric:.4f}"
    #                 f" at epoch: {best_metric_epoch}"
    #             )
    #             for c, e in enumerate(expected_labels):
    #                 print(
    #                     f"AUC_{e} ({texture_names[c]}) = {result[c]:.4f} "
    #                 )

    #     epochs_without_improvement +=1

    # print(
    #     f"train completed, best_metric: {best_metric:.4f} "
    #     f"at epoch: {best_metric_epoch}") 

    # # Plot the loss and metric
    # plt.figure("train", (12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("Epoch Average Loss")
    # x = [i + 1 for i in range(len(epoch_loss_values))]
    # y = epoch_loss_values
    # plt.xlabel("epoch")
    # plt.plot(x, y)
    # plt.subplot(1, 2, 2)
    # plt.title("Val AUC")
    # x = [val_interval * (i + 1) for i in range(len(metric_values))]
    # y = metric_values
    # plt.xlabel("epoch")
    # plt.plot(x, y)
    # plt.savefig(os.path.join(experiment_dir, f'training_{now}.jpg'))
    # plt.show()
    # training_time = training_timer.stop()

    test_timer = Timer()
    test_timer.start()
    # now we need to do inferences on the test_data
    mean_auc, acc_metric, result, y_onehot_arr, y_pred_onehot_arr = inference(test_loader, y_trans, y_pred_trans, model, device, os.path.join(experiment_dir, "best_metric_model.pth"), texture_names, expected_labels, os.path.join(experiment_dir, f'ROCAUC_repeat_{now}.png'), os.path.join(experiment_dir, f'CM_{now}.png'))
    testing_time = test_timer.stop()
    total_time = overall_timer.stop()

    # I need to save out a record of what was run through the experiment, what epoch we stopped, which was the best metric epoch etc.
    pd = {}
    # pd['best_metric_epoch'] = best_metric_epoch
    # pd['best_metric'] = best_metric
    # pd['metric_values'] = metric_values
    # pd['epoch_loss_values'] = epoch_loss_values
    pd['train_directory'] = train_directory
    pd['test_directory'] = test_directory
    pd['output_dir'] = output_dir
    pd['experiment_dir'] = experiment_dir
    # pd['pickle_prefix_train'] = pickle_prefix_train
    # pd['pickle_prefix_test'] = pickle_prefix_test
    pd['expected_labels'] = expected_labels
    pd['num_images_train'] = num_images_train
    pd['fraction_images_val'] = num_images_val
    pd['num_images_test'] = num_images_test
    pd['deterministic_training'] = deterministic_training
    pd['max_epochs'] = max_epochs
    pd['train_batch_size'] = train_batch_size
    pd['val_batch_size'] = val_batch_size
    pd['test_batch_size'] = test_batch_size
    pd['val_interval'] = val_interval
    pd['num_workers'] = num_workers
    pd['checkpoint'] = checkpoint
    pd['labels_to_change'] = labels_to_change
    pd['texture_names'] = texture_names
    pd['early_stopping'] = early_stopping
    pd['experiment_name'] = experiment_name
    pd['device_name'] = device_name
    pd['test_mean_auc'] = mean_auc
    pd['test_acc_metric'] = acc_metric
    pd['test_result'] = result
    pd['y_onehot_arr'] = y_onehot_arr
    pd['y_pred_onehot_arr'] = y_pred_onehot_arr
    # pd['loading_time'] = loading_time
    # pd['training_time'] = training_time
    pd['testing_time'] = testing_time
    pd['total_time'] = total_time


    with open(os.path.join(experiment_dir, "experiment_config_test.pickle"), 'wb') as f:
        pickle.dump(pd, f, protocol=pickle.HIGHEST_PROTOCOL)


def inference(test_loader, y_trans, y_pred_trans, model, device, model_path, texture_names, expected_labels, roc_name, conf_matrix_name):

    # load the best model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"model path does not exist: {model_path}")
    
    auc_metric = ROCAUCMetric(average="none")

    # now do the inferences
    model.eval()
    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y = torch.tensor([], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():  

        for test_data in test_loader:
            test_images, test_labels = (
                test_data[0].to(device),
                test_data[1].to(device),
            )
            y_pred = torch.cat([y_pred, model(test_images)], dim=0)
            y = torch.cat([y, test_labels], dim=0)
        y_onehot = [y_trans(i) for i in decollate_batch(y)]
        y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
        auc_metric(y_pred_act, y_onehot)
        result = auc_metric.aggregate()
        mean_auc =  np.mean(result)
        y_true_onehot_np_array = np.array([z.numpy() for z in y_onehot ])
        y_pred_onehot_np_array = np.array([z.cpu().numpy() for z in y_pred_act ])
        calculate_roc_metrics(
            y_true_onehot_np_array, 
            y_pred_onehot_np_array, 
            len(expected_labels), 
            texture_names,
            roc_name,
            conf_matrix_name)
        auc_metric.reset()
        del y_pred_act, y_onehot
        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)
        print(
            f" Testing AUC: {mean_auc:.4f}"
            f" Testing accuracy: {acc_metric:.4f}"
        )
        for c, e in enumerate(expected_labels):
            print(
                f"AUC_{e} ({texture_names[c]}) = {result[c]:.4f} "
            )
        
        return mean_auc, acc_metric, result, y_true_onehot_np_array, y_pred_onehot_np_array

def read_pickles2(pickle_dir, dims, x, y, z, fill_fraction, spacing, expected_labels, num_images, labels_to_change, random_shuffler):
    '''
    Reads the outputs from "extract_patches.py" and reassembles them as a shuffled list of patches
    will also change lables from the original to new value if required

    our patch files have names like this:
        patches_3D_2076_32x32x1_1.0_5mm_2.pickle
        patches_#dims_#min_patches_xxyxz_fill_fraction_texture.pickle
    '''
    old = '.'
    new = '\.'
    regex = re.compile(f'(patches_{dims}_)\d*(_{str(x)}x{str(y)}x{str(z)}_{str(fill_fraction).replace(old, new)}_{spacing}mm_)\d*\.pickle')
    files = [fi for fi in os.listdir(pickle_dir) if regex.match(fi)]

    data_list = []
    for label in expected_labels:
        name = ""
        for fi in files:
            if fi.endswith(str(label) +'.pickle'):
                name = os.path.join(pickle_dir, fi)
                break
        with open(name, 'rb') as f:
            tmp_dict = pickle.load(f)
            random_shuffler.shuffle(tmp_dict['patch_list'])
            data_list.extend(tmp_dict['patch_list'][:num_images])
            tmp_dict = None

    if labels_to_change:
        for (orig, new) in labels_to_change:
            for i, l in enumerate(data_list):
                if l['classification'] == orig:
                    data_list[i]['classification'] = new

    random_shuffler.shuffle(data_list)
    return data_list

def plot_confusion_matrix(  savename,
                            cm,
                            target_names,
                            title='Confusion matrix',
                            cmap=None,
                            normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if cmap is None:
        cmap = plt.get_cmap('Blues')


    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap,vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)



    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(savename, dpi=300)

def read_pickles(pickle_dir, pickle_prefix, expected_labels, num_images, labels_to_change, random_shuffler):
    '''
    Reads the outputs from "extract_patches.py" and reassembles them as a shuffled list of patches
    will also change lables from the original to new value if required

    our patch files have names like this:
        patches_3D_2076_32x32x1_1.0_5mm_2.pickle
        patches_#dims_#min_patches_x_y_z_fill_fraction_texture.pickle
    '''

    data_list = []
    for label in expected_labels:
        name = os.path.join(pickle_dir, pickle_prefix + str(label) +'.pickle')
        with open(name, 'rb') as f:
            tmp_dict = pickle.load(f)
            random_shuffler.shuffle(tmp_dict['patch_list'])
            data_list.extend(tmp_dict['patch_list'][:num_images])
            tmp_dict = None

    if labels_to_change:
        for (orig, new) in labels_to_change:
            for i, l in enumerate(data_list):
                if l['classification'] == orig:
                    data_list[i]['classification'] = new
    
    random_shuffler.shuffle(data_list)
    return data_list

def get_model(model_name : str, device: torch.device, model_named_hp: dict):
    '''
        Get's the required network for the specified model name and model_named_hp and returns it attached to the device
    '''
    VALID_MODELS = {
        "DenseNet", 
        "DenseNet121",
        "DenseNet169",
        "DenseNet201",
        "DenseNet264",
        "EfficientNet", 
        "EfficientNetB0", 
        "EfficientNetB1", 
        "EfficientNetB2", 
        "EfficientNetB3", 
        "EfficientNetB4", 
        "EfficientNetB5", 
        "EfficientNetB6", 
        "EfficientNetB7", 
        "ResNet",
        "SENet",
        "SENet154"
        "SEResNet50",
        "SEResNet101",
        "SEResNet152",
        "SEResNext50",
        "SEResNext101",
        }
    if model_name not in VALID_MODELS:
        raise ValueError(f"model name must be one of {str(VALID_MODELS)}")

    net = getattr(nets, model_name)
    return net( **model_named_hp).to(device)

def calculate_roc_metrics(y_true : np.ndarray, y_pred : np.ndarray, n_classes: int, class_names : list, roc_name: str, conf_matrix_name: str) -> dict:
    # see https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    # Compute ROC curve and ROC area for each class
    lw = 2  # linewidth
    thresh = dict()
    fpr = dict()
    tpr = dict()
    roc_auc = dict() 
    dice = []
    accuracy = []
    jaccard = []

    for i in range(n_classes):
        fpr[i], tpr[i], thresholds = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        thresh[i] = thresholds[np.argmax(tpr[i] - fpr[i])]
        cm = metrics.ConfusionMatrix(y_true[:, i], y_pred[:, i])

        dice.append(metrics.dice(confusion_matrix=cm))
        accuracy.append(metrics.accuracy(confusion_matrix=cm))
        jaccard.append(metrics.jaccard(confusion_matrix=cm))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], thresholds = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    thresh["micro"] = thresholds[np.argmax(tpr["micro"] - fpr["micro"])]

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "black"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            # label="class {0} ({2}) (area = {1:0.2f}) (threshold = {3:0.3f})".format(i+1, roc_auc[i], class_names[i], thresh[i]),
            label="class {0} ({2}) (area = {1:0.2f})".format(i+1, roc_auc[i], class_names[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass Reciever Operating Curve")
    plt.legend(loc="lower right")
    plt.savefig(roc_name, dpi=300)
    plt.close()

    cm = confusion_matrix(y_true=np.argmax(y_true, axis=-1), y_pred=np.argmax(y_pred, axis=-1), labels=np.unique(np.argmax(y_true, axis=-1)))
    plot_confusion_matrix(savename=conf_matrix_name, cm=cm, target_names=class_names, normalize=True)

    result = {
        "thresh" : thresh,
        "fpr" : fpr,
        "tpr" : tpr,
        "roc_auc" : roc_auc,
        "accuracy" : accuracy,
        "dice" : dice,
        "jaccard" : jaccard,
        "confusion_matrix" : cm
    }

    return result

def load_cmd_args():
    
    description = "Trains a deep-learning model to classify patches"

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--train_dir", "--trd", help="The path to the directory containing training files (these are split for train and validation)", type=str, required=True
    )
    parser.add_argument(
        "--test_dir", "--ted", help="The path to the directory containing test files", type=str, required=True
    )
    parser.add_argument(
        "--output_dir", "-o", help="The path to the top output dir, experiment results will be saved in subdirectories of this", type=str, required=True
    )    
    parser.add_argument(
        "--pickle_prefix_train", "--pptr", help="The training pickle prefix for this experiment", type=str, required=True
    )
    parser.add_argument(
        "--pickle_prefix_test", "--ppte", help="The test pickle prefix for this experiment", type=str, required=True
    )
    parser.add_argument(
        "--expected_labels", "-e", help="The expected labels for this experiment", nargs="+", type=int, required=True
    )
    parser.add_argument(
        "--texture_names", help="The names of the expected labels in the same order", nargs="+", type=str, required=True
    )
    parser.add_argument(
        "--num_patches_train", "--nptr", help="The number of patches (per class) for training in this experiment", type=int, required=True
    )
    parser.add_argument(
        "--num_patches_test", "--npte", help="The number of patches (per class) for testing in this experiment", type=int, required=True
    )
    parser.add_argument(
        "--fraction_val", "-f", help="The fraction of training patches for validation in this experiment", type=float, required=True
    )
    parser.add_argument(
        "--deterministic_training", "-d", 
        help="Deterministic (not random) training for this experiment- NB some of the networks are non determministic", 
        action="store_true", #this means if the cmd line arg is present it will store true, otherwise will be false which is what we want.
    )
    parser.add_argument(
        "--max_epochs", "--max", help="The maximum number of training epochs in this experiment", type=int, default=10000
    )
    parser.add_argument(
        "--train_batch_size", "--trbs", help="The training batch size in this experiment", type=int, default=10000
    )
    parser.add_argument(
        "--val_batch_size", "--vbs", help="The validation batch size in this experiment", type=int, default=10000
    )
    parser.add_argument(
        "--test_batch_size", "--tebs", help="The testing batch size in this experiment", type=int, default=10000
    )
    parser.add_argument(
        "--val_interval", "-v", help="The validation interval in this experiment", type=int, default=20
    )
    parser.add_argument(
        "--workers", "-w", help="The number of processor cores to use in this experiment", type=int, default=8
    )
    parser.add_argument(
        "--checkpoint", "-c", help="File name for the checkpoint", type=str, default='best_metric_model.pth'
    )
    parser.add_argument(
        "--labels_to_change", "-l", help="Labels to change to meet requirement to have labels from 0 to n with no gaps,\n \
        these are entered as a space delimited list of pairs of comma separated ints, old_val,new_val \n \
        e.g \"-l 6,5 7,6 9,7\"", type=lambda a: tuple(map(int, a.split(','))), nargs='+'
    )
    parser.add_argument(
        "--early_stopping_patience", "-p", help="The early stopping patience in epochs", default=100, type=int
    )
    parser.add_argument(
        "--experiment_name", "-n", help="The name for this experiment default is: \
        #nTrain_\
        #nTest_\
        #fractionVal_\
        pickle_prefix_train_\
        picle_prefix_test_\
        modelName\
        ", default=None, type=str
    )
    parser.add_argument(
        "--device_name", "--dn", help="The device to train the experiment on", type=str, default="cuda:0"        
    )
    parser.add_argument(
        "--model_name", help="The model to train with", type=str, default="DenseNet121"
    )
    parser.add_argument(
        "--model_named_hp", "--hp", help="hyper parameters for the model ,\n \
        these are entered as a python dictionary! \n \
        e.g \"--hp { 'spatial_dims':2, 'in_channels':1, 'out_channels':8 }\"", type=lambda a: ast.literal_eval(a), required=True
    )
    parser.add_argument(
        "--kernel_x", "-x", help="X kernel size", type=int, required=True 
    )
    parser.add_argument(
        "--kernel_y", "-y", help="Y kernel size", type=int, required=True 
    )
    parser.add_argument(
        "--kernel_z", "-z", help="X kernel size", type=int, required=True 
    )
    parser.add_argument(
        "--fill_fraction", "--ff", help="Fill fraction for this experiment", type=float, required=True 
    )
    args = parser.parse_args()
    
    if not args.experiment_name:
        mp = [key[:4] + "-" + str(args.model_named_hp[key]) for key in args.model_named_hp]
        name =  "Tr" + str(args.num_patches_train) + "_Ts" + \
                str(args.num_patches_test) + "_Val" + \
                str(args.fraction_val) + "_X" + \
                str(args.kernel_x) + "_Y" + \
                str(args.kernel_y) + "_Z" + \
                str(args.kernel_z) + "_ff" + \
                str(args.fill_fraction) + "_labs" + \
                "-".join([str(item) for item in args.expected_labels]) + "_" + \
                str(args.model_name) + "_" + \
                "_".join(mp)
        args.experiment_name = name
    return args    

if __name__ == '__main__':

    args = load_cmd_args()

    train(  args.train_dir,
            args.test_dir,
            args.output_dir,
            args.pickle_prefix_train,
            args.pickle_prefix_test,
            args.expected_labels, 
            args.num_patches_train,
            args.fraction_val,
            args.num_patches_test, 
            deterministic_training=args.deterministic_training, 
            max_epochs=args.max_epochs, 
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size,
            test_batch_size=args.test_batch_size,
            val_interval=args.val_interval,
            num_workers=args.workers,
            checkpoint=args.checkpoint,
            labels_to_change=args.labels_to_change,
            texture_names=args.texture_names,
            early_stopping=args.early_stopping_patience,
            experiment_name=args.experiment_name,
            device_name=args.device_name,
            model_name=args.model_name,
            model_named_hp=args.model_named_hp,
            )
