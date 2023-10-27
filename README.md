# Flower-image_Classifier
Project code for Udacity's AI Programming with Python Nanodegree program. In this project, first develop code for an image classifier built with PyTorch, then convert it into a command line application.
This project was established by Udacity and performed within Udacity's GPU enabled workspace.

# Project Breakdown
### Part1
In this first part of the project, implement an image classifier with PyTorch, train it to recognize different species of flowers.

The project is broken down into multiple steps:

* Load and preprocess the image dataset
  * import the data while applying proper transforms and segmenting them into respective training, validation, and testing datasets 
* Building and training the classifier
  * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) 
  * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
  * Train the classifier layers using backpropagation using the pre-trained network to get the features
  * Track the loss and accuracy on the validation set to determine the best hyperparameters
* Use the trained classifier to predict image content
* Save Checkpoint

### Part2
convert image classifier into a command line application.
1. **Train**: Train a new network on a data set with `train.py`
   * ***Basic usage***: `python train.py data_directory`
   * Prints out training loss, validation loss, and validation accuracy as the network trains
   * ***Options***:
      * Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
      * Choose architecture: `python train.py data_dir --arch "vgg13"`
      * Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
      * Use GPU for training: `python train.py data_dir --gpu True`
2. **Predict**: Predict flower name from an image with `predict.py` along with the probability of that name. That is, you'll pass in a single image `/path/to/image` and return the flower name and class probability.
   * ***Basic usage***: `python predict.py /path/to/image checkpoint`
   * ***Options***:
     * Return top K most likely classes: `python predict.py input checkpoint --top_k 3`
     * Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
     * Use GPU for inference: `python predict.py input checkpoint --gpu True`
     
