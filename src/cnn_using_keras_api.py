import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):

	# the data, split between train and test sets
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# input image dimensions
	img_rows, img_cols = 28, 28

	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

	model = tf.keras.models.Sequential()

	# Conv1 section
	model.add(tf.keras.layers.Conv2D(32, (3, 3), padding = 'same', input_shape=input_shape))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.MaxPool2D((2, 2), strides = (2, 2)))

	# Conv2 section
	model.add(tf.keras.layers.Conv2D(32, (3, 3), padding = 'same'))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.MaxPool2D((2, 2), strides = (2, 2)))
	
	# Dense section
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(256))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.Dropout(0.4))

	# output layer 10 classes
	model.add(tf.keras.layers.Dense(10))
	model.add(tf.keras.layers.Softmax())

	model.compile(optimizer = tf.keras.optimizers.Adam(0.01),
				  loss = tf.keras.losses.sparse_categorical_crossentropy,
				  metrics=['accuracy'])

	# print model architecture
	print(model.summary())

	model.fit(x_train, y_train, epochs=1, batch_size=32)
	
	# print test loss and metric
	print(model.evaluate(x_test, y_test))
	
	# save model
	model_name = 'mnist_model.h5'
	model.save(model_name)

	# load model. This fails.
	loaded_model = tf.keras.models.load_model(model_name)

	# print test loss and metric for loaded model
	print(loaded_model.evaluate(x_test, y_test))

if __name__ == "__main__":
	tf.app.run()
