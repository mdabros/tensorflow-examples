import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras import applications
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
	
	# Setup data paths
	data_directory = r'E:\DataSets\CatsAndDogs\tensorflow'
	img_path = data_directory + r'\train_small\cat\cat.43.jpg'
	heat_map_path = data_directory + r'\tabby_cat_heatmap.jpg'

	# load full vgg16 model.
	model = applications.VGG16(weights='imagenet', include_top=False)
	
	# It seems that filter 0 in layer block3_conv1 is responsive to a polka dot pattern.
	plt.imshow(generate_pattern(model, 'block3_conv1', 0))
	plt.show()

	# Now the fun part: we can start visualising every single filter in every layer. 
	# For simplicity, we will only look at the first 64 filters in each layer, 
	# and will only look at the first layer of each convolution block (block1_conv1, block2_conv1, block3_conv1, block4_conv1, block5_conv1). 
	# We will arrange the outputs on a 8x8 grid of 64x64 filter patterns, with some black margins between each filter pattern.

	for layer_name in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']:
		size = 64
		margin = 5

		# This a empty (black) image where we will store our results.
		results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

		for i in range(8):  # iterate over the rows of our results grid
			for j in range(8):  # iterate over the columns of our results grid
				# Generate the pattern for filter `i + (j * 8)` in `layer_name`
				filter_img = generate_pattern(model, layer_name, i + (j * 8), size=size)

				# Put the result in the square `(i, j)` of the results grid
				horizontal_start = i * size + i * margin
				horizontal_end = horizontal_start + size
				vertical_start = j * size + j * margin
				vertical_end = vertical_start + size
				results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

		# Display the results grid
		plt.figure(figsize=(20, 20))
		plt.imshow(results)
		plt.show()
	
	# The filters from the first layer in the model (block1_conv1) encode simple directional edges and colors (or colored edges in some cases).
	# The filters from block2_conv1 encode simple textures made from combinations of edges and colors.
	# The filters in higher-up layers start resembling textures found in natural images: feathers, eyes, leaves, etc.

def deprocess_image(x):
	# normalize tensor: center on 0., ensure std is 0.1
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	# clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)

	# convert to RGB array
	x *= 255
	x = np.clip(x, 0, 255).astype('uint8')
	return x

def generate_pattern(model, layer_name, filter_index, size=150):
	# Build a loss function that maximizes the activation
	# of the nth filter of the layer considered.
	layer_output = model.get_layer(layer_name).output
	loss = tf.keras.backend.mean(layer_output[:, :, :, filter_index])

	# Compute the gradient of the input picture wrt this loss
	grads = tf.keras.backend.gradients(loss, model.input)[0]

	# Normalization trick: we normalize the gradient
	grads /= (tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(grads))) + 1e-5)

	# This function returns the loss and grads given the input picture
	iterate = tf.keras.backend.function([model.input], [loss, grads])
	
	# We start from a gray image with some noise
	input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

	# Run gradient ascent for 40 steps
	step = 1.
	for i in range(40):
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value * step

	img = input_img_data[0]
	return deprocess_image(img)


if __name__ == "__main__":
	tf.app.run()
