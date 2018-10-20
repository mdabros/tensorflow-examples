
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras import applications
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import cv2
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
	
	# Setup data paths
	data_directory = r'E:\DataSets\CatsAndDogs\tensorflow'
	img_path = data_directory + r'\train_small\cat\cat.43.jpg'
	model_path = data_directory + r'\cats_and_dogs_small_model_small.h5'

	# load catas and dog model.
	# Note, that this requires a pre-trained model from one of the other steps
	model = tf.keras.models.load_model(model_path)
	print(model.summary())

	img = image.load_img(img_path, target_size=(150, 150))
	img_tensor = image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	# Remember that the model was trained on inputs
	# that were preprocessed in the following way:
	img_tensor /= 255.
	
	# Its shape is (1, 150, 150, 3)
	print(img_tensor.shape)

	plt.imshow(img_tensor[0])
	plt.show()

	# Extracts the outputs of the top 8 layers:
	layer_outputs = [layer.output for layer in model.layers[:8]]
	# Creates a model that will return these outputs, given the model input:
	activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
	
	# This will return a list of 5 Numpy arrays:
	# one array per layer activation
	activations = activation_model.predict(img_tensor)

	# Activation of the first convolutional layer
	first_layer_activation = activations[0]
	print(first_layer_activation.shape)

	# Visualize channel 3
	plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
	plt.show()

	# Visualize channel 30
	plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
	plt.show()

	# Finally visualize all top8 filters
	visualize_all_filters(model, activations)

	# The first layer acts as a collection of various edge detectors. At that stage, 
	# the activations are still retaining almost all of the information present in the initial picture.
	
	# As we go higher-up, the activations become increasingly abstract and less visually interpretable. 
	# They start encoding higher-level concepts such as "cat ear" or "cat eye". 
	# Higher-up presentations carry increasingly less information about the visual contents of the image, 
	# and increasingly more information related to the class of the image.
	
	# The sparsity of the activations is increasing with the depth of the layer: in the first layer, 
	# all filters are activated by the input image, but in the following layers more and more filters are blank. 
	# This means that the pattern encoded by the filter isn't found in the input image.


def visualize_all_filters(model, activations):
	# These are the names of the layers, so can have them as part of our plot
	layer_names = []
	for layer in model.layers[:8]:
		layer_names.append(layer.name)

	images_per_row = 16

	# Now let's display our feature maps
	for layer_name, layer_activation in zip(layer_names, activations):
		# This is the number of features in the feature map
		n_features = layer_activation.shape[-1]

		# The feature map has shape (1, size, size, n_features)
		size = layer_activation.shape[1]

		# We will tile the activation channels in this matrix
		n_cols = n_features // images_per_row
		display_grid = np.zeros((size * n_cols, images_per_row * size))

		# We'll tile each filter into this big horizontal grid
		for col in range(n_cols):
			for row in range(images_per_row):
				channel_image = layer_activation[0,
													:, :,
													col * images_per_row + row]
				# Post-process the feature to make it visually palatable
				channel_image -= channel_image.mean()
				channel_image /= channel_image.std()
				channel_image *= 64
				channel_image += 128
				channel_image = np.clip(channel_image, 0, 255).astype('uint8')
				display_grid[col * size : (col + 1) * size,
								row * size : (row + 1) * size] = channel_image

		# Display the grid
		scale = 1. / size
		plt.figure(figsize=(scale * display_grid.shape[1],
							scale * display_grid.shape[0]))
		plt.title(layer_name)
		plt.grid(False)
		plt.imshow(display_grid, aspect='auto', cmap='viridis')

	plt.show()

if __name__ == "__main__":
	tf.app.run()
