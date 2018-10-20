import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras import applications
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import cv2
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

# Visualization class activation heatmap technique:
# The specific implementation we will use is the one described in Grad-CAM: Why did you say that? Visual Explanations 
# from Deep Networks via Gradient-based Localization. # It is very simple: it consists in taking the output feature map 
# of a convolution layer given an input image, 
# and weighing every channel in that feature map by the gradient of the class with respect to the channel. 
# Intuitively, one way to understand this trick is that we are weighting a spatial map of 
# "how intensely the input image activates different channels" by "how important each channel is with regard to the class", 
# resulting in a spatial map of "how intensely the input image activates the class".

def main(unused_argv):
	
	# Setup data paths
	data_directory = r'E:\DataSets\CatsAndDogs\tensorflow'
	img_path = data_directory + r'\train_small\cat\cat.43.jpg'
	heat_map_path = data_directory + r'\tabby_cat_heatmap.jpg'

	# load full vgg16 model.
	model = applications.VGG16(weights='imagenet')
	
	# Print pretrained architecture.
	print(model.summary())

	# `img` is a PIL image of size 224x224
	img = image.load_img(img_path, target_size=(224, 224))

	# `x` is a float32 Numpy array of shape (224, 224, 3)
	x = image.img_to_array(img)

	# We add a dimension to transform our array into a "batch"
	# of size (1, 224, 224, 3)
	x = np.expand_dims(x, axis=0)

	# Finally we preprocess the batch
	# (this does channel-wise color normalization)
	x = preprocess_input(x)

	# Print top 3 predictions.
	preds = model.predict(x)
	print('Predicted:', decode_predictions(preds, top=3)[0])

	# Get top1 prediction class index
	# This is the "tabby" entry in the prediction vector
	top1 = np.argmax(preds[0])
	tabby_cat_output = model.output[:, top1]

	# This is the output feature map of the `block5_conv3` layer,
	# the last convolutional layer in VGG16
	last_conv_layer = model.get_layer('block5_conv3')

	# This is the gradient of the "tabby" class with regard to
	# the output feature map of `block5_conv3`
	grads = tf.keras.backend.gradients(tabby_cat_output, last_conv_layer.output)[0]

	# This is a vector of shape (512,), where each entry
	# is the mean intensity of the gradient over a specific feature map channel
	pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2))

	# This function allows us to access the values of the quantities we just defined:
	# `pooled_grads` and the output feature map of `block5_conv3`,
	# given a sample image
	iterate = tf.keras.backend.function([model.input], [pooled_grads, last_conv_layer.output[0]])

	# These are the values of these two quantities, as Numpy arrays,
	pooled_grads_value, conv_layer_output_value = iterate([x])

	# We multiply each channel in the feature map array
	# by "how important this channel is" with regard to the "tabby" class.
	for i in range(512):
		conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

	# The channel-wise mean of the resulting feature map
	# is our heatmap of class activation.
	heatmap = np.mean(conv_layer_output_value, axis=-1)

	# Normalize heatmap between 0 and 1 for visualization.
	heatmap = np.maximum(heatmap, 0)
	heatmap /= np.max(heatmap)
	plt.matshow(heatmap)
	plt.show()

	# We use cv2 to load the original image
	img = cv2.imread(img_path)

	# We resize the heatmap to have the same size as the original image
	heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

	# We convert the heatmap to RGB
	heatmap = np.uint8(255 * heatmap)

	# We apply the heatmap to the original image
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

	# 0.4 here is a heatmap intensity factor
	superimposed_img = heatmap * 0.4 + img

	# Save the image to disk
	cv2.imwrite(heat_map_path, superimposed_img)

if __name__ == "__main__":
	tf.app.run()
