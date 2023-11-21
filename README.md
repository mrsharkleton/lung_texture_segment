# Lung Texture Segmentation
Lung parenchymal texture labelling with coarse segmentation.

This project is licensed under the terms of the MIT license.

## Installation
``` shell
conda create --name lung_texture python=3.9
conda activate lung_texture
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
```

## To extract a dataset
``` python 
python lung_texture_segment/extract_patches.py -h
```

``` shell
usage: extract_patches.py [-h] --base_data_dir BASE_DATA_DIR [--test_patches TEST_PATCHES]
                          [--val_patches VAL_PATCHES] [--train_patches TRAIN_PATCHES]
                          [--patch_size PATCH_SIZE [PATCH_SIZE ...]]
                          [--fill_fraction FILL_FRACTION]
                          [--expected_labels EXPECTED_LABELS [EXPECTED_LABELS ...]]

Description of your script.

  optional arguments:
    -h, --help            show this help message and exit
    --base_data_dir BASE_DATA_DIR
                          Base directory for data.
    --test_patches TEST_PATCHES
                          Number of patches for the test set.
    --val_patches VAL_PATCHES
                          Number of patches for the validation set.
    --train_patches TRAIN_PATCHES
                          Number of patches for the training set.
    --patch_size PATCH_SIZE [PATCH_SIZE ...]
                          Patch size as a list of three integers [depth, height, width].
    --fill_fraction FILL_FRACTION
                          Fill fraction for data augmentation.
    --expected_labels EXPECTED_LABELS [EXPECTED_LABELS ...]
                          List of expected label values.
```

## To train a model
``` python
python lung_texture_segment/training.py -h
```

``` shell
usage: training.py [-h] --train_dir TRAIN_DIR --test_dir TEST_DIR --output_dir OUTPUT_DIR
                   --pickle_prefix_train PICKLE_PREFIX_TRAIN --pickle_prefix_test
                   PICKLE_PREFIX_TEST --expected_labels EXPECTED_LABELS [EXPECTED_LABELS ...]
                   --texture_names TEXTURE_NAMES [TEXTURE_NAMES ...] --num_patches_train
                   NUM_PATCHES_TRAIN --num_patches_test NUM_PATCHES_TEST --fraction_val
                   FRACTION_VAL [--deterministic_training] [--max_epochs MAX_EPOCHS]
                   [--train_batch_size TRAIN_BATCH_SIZE] [--val_batch_size VAL_BATCH_SIZE]
                   [--test_batch_size TEST_BATCH_SIZE] [--val_interval VAL_INTERVAL]
                   [--workers WORKERS] [--checkpoint CHECKPOINT]
                   [--labels_to_change LABELS_TO_CHANGE [LABELS_TO_CHANGE ...]]
                   [--early_stopping_patience EARLY_STOPPING_PATIENCE]
                   [--experiment_name EXPERIMENT_NAME] [--device_name DEVICE_NAME]
                   [--model_name MODEL_NAME] --model_named_hp MODEL_NAMED_HP --kernel_x KERNEL_X
                   --kernel_y KERNEL_Y --kernel_z KERNEL_Z --fill_fraction FILL_FRACTION

Trains a deep-learning model to classify patches

optional arguments:
  -h, --help            show this help message and exit
  --train_dir TRAIN_DIR, --trd TRAIN_DIR
                        The path to the directory containing training files (these are split for
                        train and validation)
  --test_dir TEST_DIR, --ted TEST_DIR
                        The path to the directory containing test files
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        The path to the top output dir, experiment results will be saved in
                        subdirectories of this
  --pickle_prefix_train PICKLE_PREFIX_TRAIN, --pptr PICKLE_PREFIX_TRAIN
                        The training pickle prefix for this experiment
  --pickle_prefix_test PICKLE_PREFIX_TEST, --ppte PICKLE_PREFIX_TEST
                        The test pickle prefix for this experiment
  --expected_labels EXPECTED_LABELS [EXPECTED_LABELS ...], -e EXPECTED_LABELS [EXPECTED_LABELS ...]
                        The expected labels for this experiment
  --texture_names TEXTURE_NAMES [TEXTURE_NAMES ...]
                        The names of the expected labels in the same order
  --num_patches_train NUM_PATCHES_TRAIN, --nptr NUM_PATCHES_TRAIN
                        The number of patches (per class) for training in this experiment
  --num_patches_test NUM_PATCHES_TEST, --npte NUM_PATCHES_TEST
                        The number of patches (per class) for testing in this experiment
  --fraction_val FRACTION_VAL, -f FRACTION_VAL
                        The fraction of training patches for validation in this experiment
  --deterministic_training, -d
                        Deterministic (not random) training for this experiment- NB some of the
                        networks are non determministic
  --max_epochs MAX_EPOCHS, --max MAX_EPOCHS
                        The maximum number of training epochs in this experiment
  --train_batch_size TRAIN_BATCH_SIZE, --trbs TRAIN_BATCH_SIZE
                        The training batch size in this experiment
  --val_batch_size VAL_BATCH_SIZE, --vbs VAL_BATCH_SIZE
                        The validation batch size in this experiment
  --test_batch_size TEST_BATCH_SIZE, --tebs TEST_BATCH_SIZE
                        The testing batch size in this experiment
  --val_interval VAL_INTERVAL, -v VAL_INTERVAL
                        The validation interval in this experiment
  --workers WORKERS, -w WORKERS
                        The number of processor cores to use in this experiment
  --checkpoint CHECKPOINT, -c CHECKPOINT
                        File name for the checkpoint
  --labels_to_change LABELS_TO_CHANGE [LABELS_TO_CHANGE ...], -l LABELS_TO_CHANGE [LABELS_TO_CHANGE ...]
                        Labels to change to meet requirement to have labels from 0 to n with no
                        gaps, these are entered as a space delimited list of pairs of comma
                        separated ints, old_val,new_val e.g "-l 6,5 7,6 9,7"
  --early_stopping_patience EARLY_STOPPING_PATIENCE, -p EARLY_STOPPING_PATIENCE
                        The early stopping patience in epochs
  --experiment_name EXPERIMENT_NAME, -n EXPERIMENT_NAME
                        The name for this experiment default is: #nTrain_ #nTest_ #fractionVal_
                        pickle_prefix_train_ picle_prefix_test_ modelName
  --device_name DEVICE_NAME, --dn DEVICE_NAME
                        The device to train the experiment on
  --model_name MODEL_NAME
                        The model to train with
  --model_named_hp MODEL_NAMED_HP, --hp MODEL_NAMED_HP
                        hyper parameters for the model , these are entered as a python
                        dictionary! e.g "--hp { 'spatial_dims':2, 'in_channels':1,
                        'out_channels':8 }"
  --kernel_x KERNEL_X, -x KERNEL_X
                        X kernel size
  --kernel_y KERNEL_Y, -y KERNEL_Y
                        Y kernel size
  --kernel_z KERNEL_Z, -z KERNEL_Z
                        X kernel size
  --fill_fraction FILL_FRACTION, --ff FILL_FRACTION
                        Fill fraction for this experiment
```

## Reconstruct an inference
``` python
python lung_texture_segment/recon_image.py -h
```

``` shell
usage: recon_image.py [-h] --ct_path CT_PATH --lung_label_path LUNG_LABEL_PATH
                      [--kernel KERNEL [KERNEL ...]] [--model_dir MODEL_DIR]
                      [--output_path OUTPUT_PATH] [--stride STRIDE [STRIDE ...]]
                      [--fraction_lung FRACTION_LUNG] [--processors PROCESSORS]

recon_image.py: author: sharkeymj data: July 2022 This reconstructs a coarse semantic labelling
of a thoracic lung scan showing areas of disease.

optional arguments:
  -h, --help            show this help message and exit
  --ct_path CT_PATH, -c CT_PATH
                        The path to a thoracic CT scan to label
  --lung_label_path LUNG_LABEL_PATH, -l LUNG_LABEL_PATH
                        The path to the lung label associated with the thoracic CT scan
  --kernel KERNEL [KERNEL ...], -k KERNEL [KERNEL ...]
                        The kernel size in pixels
  --model_dir MODEL_DIR, -m MODEL_DIR
                        Directory containing models for the different kernels
  --output_path OUTPUT_PATH, -o OUTPUT_PATH
                        Path to save reconstruction to
  --stride STRIDE [STRIDE ...]
                        The stride in pixels between kernels. Must be <=k. k=no overlap (fully
                        strided)
  --fraction_lung FRACTION_LUNG, -f FRACTION_LUNG
                        The fraction of a patch which must be within the lung before it is
                        classified
  --processors PROCESSORS, -p PROCESSORS
                        The number of processors to use for multiprocessing and extracting
                        patches and reconstructing the image
```
