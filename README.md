tensorflow-examples
=====================

Repository for tensorflow examples and experiments.

## Examples

 * [**cnn_using_keras_api**](https://github.com/mdabros/tensorflow-examples/blob/master/src/cnn_using_estimator_api.py): Train a custom convolutional neural network using the tf.keras API.
 * [**transfer_learning_using_keras_applications**](https://github.com/mdabros/tensorflow-examples/blob/master/src/transfer_learning_using_tf_hub.py): Train a neural network using the keras.applications and transfer learning.

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

