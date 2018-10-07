tensorflow-examples
=====================

Repository for tensorflow examples and experiments.

## Examples

 * [**1_cnn_using_keras_api**](https://github.com/mdabros/tensorflow-examples/blob/master/src/1_cnn_using_keras_api.py): Train a custom convolutional neural network using the tf.keras API and ImageDataGenerator.
 * [**2_data_augmentation_using_keras_api.py**](https://github.com/mdabros/tensorflow-examples/blob/master/src/2_data_augmentation_using_keras_api.py): Train a custom convolutional neural network tf.keras API and ImageDataGenerator with data augmentation.
 * [**3_transfer_learning_using_keras_applications**](https://github.com/mdabros/tensorflow-examples/blob/master/src/3_transfer_learning_using_keras_applications.py): Train and fine-tune a neural network using a pre-trained model from the keras.applications API.
 * [**4_cnn_visualize_class_activation_heatmaps**](https://github.com/mdabros/tensorflow-examples/blob/master/src/4_cnn_visualize_class_activation_heatmaps.py): Visualize class activation heatmaps from a convolutional neural network, to determine where in the image the network is focusing.

## Installation

### Python 3.6 in Visual Studio 2017
Use **Visual Studio Installer** -> select **Modify**.

Workload:
 * Install **Data science and analytical applications**  workload

Components:
 * Install **Python language support**
 * Install **Python 3 64-bit** (e.g. 3.6.3)

Install Tensor Flow packages:
 * Start Visual Studio, open **Python Environments**
 * Select **Python 3.6 (64-bit)**
 * On the **Overview** combo box select **Install from requirements.txt**
 
### Installing cuDNN
If using the gpu version of tensorflow, make sure to check which version of cuda and cuDNN is supported. 
Then follow the instructions on [cudnn-install-windows](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows)

### Cats and Dogs DataSet
The cats and dogs data set used in the examples, can be downloaded from [kaggle cats and dogs](https://www.kaggle.com/c/dogs-vs-cats/data).

The examples are based on splitting the training set into 3 smaller data sets:
 * Train: the first 2000 cat and the first 2000 dog images.
 * Validation: The next 1000 cat and the next 1000 dog images.
 * Test: The next 1000 cat and the next 1000 dog images.

The 3 sets should each be placed in a separate folder, with sub folders for each class (cat and dog):
 - Train: Data/Train/Cat, Data/Train/Dog.
 - Validation: Data/Validation/Cat, Data/Validation/Dog.
 - Train: Data/Test/Cat, Data/Test/Dog.

This structure will work with the keras `ImageDataGenerator.flow_from_directory` method. 

