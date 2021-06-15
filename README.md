# MMF: A loss extension for feature learning in openset recognition

This repository is the official implementation of "MMF: A loss extension for feature learning in openset recognition". 

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```


## Training and Evaluation
The experiments will load the dataset from the `data/` directory, then split it into training, validation and test sets in customized ratio. The default ratio (also the ratio we use in the paper) is `0.75(training): 0.25(testing)`. To run the experiments in the paper, use this command:

```train
python run_exp_opennet.py -n <network_type>[cnn, flat] -ds <dataset>[mnist, msadjmat, android] -m <loss_function_type>[ce, ii, triplet] -trc_file <known_classes_file>[mnist_trc_list] -o <output_file>  -p <pre-training model>[false, trans, recon] --transformation <transformation type>[none, rotation, ae-rotation]
```
e.g. 
```
# MNIST dataset
# Using DTAE for pretraining (transformation: rotation)
# Using ii loss for fine-tuning
python run_exp_opennet.py -n cnn -ds mnist -m ii  -trc_file "data/mnist_trc_list" -o "data/results/cnn/mnist" -p recon --transformation ae-rotation
```
