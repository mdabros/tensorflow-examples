import tensorflow as tf
from tensorflow.python.keras import applications
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def adjust_image(data):
    # Reshaped to [batch, height, width, channels].
    imgs = tf.reshape(data, [-1, 28, 28, 1])
    # Adjust image size to that of the transfer learning model.
    imgs = tf.image.resize_images(imgs, (224, 224))
    # Convert to RGB image.
    imgs = tf.image.grayscale_to_rgb(imgs)
    return imgs

def main(unused_argv):
	
	# the data, split between train and test sets
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# input image dimensions
	img_rows, img_cols = 224, 224

	x_train = adjust_image(x_train)#x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = adjust_image(x_test)#x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

	# Load the transfer learning model from the model hub.
	# model hub is currently not compatible with tf.keras for fine tuning. 
	# So using keras.applications instead
	# module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_035_96/feature_vector/1")
	
	# load vgg16 without final dense layers.
	vgg16_model = tf.keras.Sequential([applications.VGG16(weights='imagenet', include_top=True)])
	
	## create new dense layers as final classifier.
	#dense_model = tf.keras.Sequential()
	#dense_model.add(tf.keras.layers.Flatten())
	#dense_model.add(tf.keras.layers.Dense(256))
	#dense_model.add(tf.keras.layers.ReLU())
	#dense_model.add(tf.keras.layers.Dropout(0.5))
	#dense_model.add(tf.keras.layers.Dense(10))
	#dense_model.add(tf.keras.layers.Softmax())

	## add new dense layers to vgg model.
	#vgg16_model.add(dense_model)

	# set the first 25 layers (up to the last conv block)
	# to non-trainable (weights will not be updated)
	for layer in vgg16_model.layers[:25]:
		layer.trainable = False

	# compile the model with a SGD/momentum optimizer
	# and a very slow learning rate.
	vgg16_model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
		loss=tf.keras.losses.sparse_categorical_crossentropy,
		metrics=['accuracy'])

	# print model architecture
	print(vgg16_model.summary())

	vgg16_model.fit(x_train, y_train, 
				 epochs=1, 
				 #batch_size=32, 
				 steps_per_epoch = 60000)
	
	# print test loss and metric
	print(vgg16_model.evaluate(x_test, y_test))
	
	# save model
	model_name = 'fine_tune_mnist_model.h5'
	vgg16_model.save(model_name)

if __name__ == "__main__":
	tf.app.run()
