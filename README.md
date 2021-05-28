# This is the code for the paper "3D Convolution for Multidate Crop Recognition from Multitemporal Image Sequences"

## Installing the required python packages

The list of anaconda commands to recreate the environment for this project is in requirements.txt

## Preparing the input images 

Use the helper file tif_to_npy.py to convert the TIF VH and VV input image bands for each date into NPY format, while also converting from dB to intensity values. Set the im_folders variable to a list of folders for each image date.


## Instructions

The script networks/convlstm_networks/train_src/main.py is used to train the networks, with the following options:

1. "dataset" will select the dataset with "cv" for Campo Verde and "lem" for LEM
2. "args.model_type" will select which model to use. Models used in the paper are:

  * BUnet4ConvLSTM (In the paper known as Baseline)
  * BUnet4ConvLSTM_SkipLSTM (In the paper known as BConvLSTM_Skip)
  * Unet3D (In the paper known as Unet3D)
  * Unet3D_ATPP (In the paper known as Unet3D_ATPP)

Parameters for each model can be set within the model delcaration. Other models present in the script were also tested in our research and can be tweaked and evaluated.

The script networks/convlstm_networks/train_src/analysis/analysis_fcn_journal_importantclasses2.py is used to evaluate the models (Figure 7-13 in the paper), with the following options:

1. "dataset" will select the dataset with "cv" for Campo Verde and "lm" for LEM
2. "small_classes_ignore" can be set to ignore classes with few samples in any date. Further configuration can be done if this option is set to True to define specific classes to be ignored or to define a threshold for the classes to be considered, as done in the paper for Figure 12
3. "skip_crf" can be set to turn CRF on and off with further configuration done within the "dense_crf" function
4. "exp_id" can be set within each dataset to define the models to be analysed. For each id an "experiment_groups" can be created as an array of models with size MxN, where M is the number of different models to be evaluated and N defines number of experiments of the same model (up to five). So an array fo size 1x1 will evaluate a single model while an array of 4x5 will evaluate 4 models, each with 5 experiments.

The script networks/convlstm_networks/results/convlstm_results/reconstruct/reconstruct_predictions_from_full_ims2.py is used to reconstruct the full image with the predictions of a specific model (Figure 14 in the paper), with the following options:

1. "dataset" will select the dataset with "cv" for Campo Verde and "lm" for LEM
2. "predictions_path" will define the model to be used
3. "skip_crf" can be set to turn CRF on and off with further configuration done within the "dense_crf" function
