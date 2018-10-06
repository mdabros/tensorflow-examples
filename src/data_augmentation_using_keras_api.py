import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):

	# All images will be rescaled by 1./255
	train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
	test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(
		# This is the target directory
		r'E:\DataSets\CatsAndDogs\train\train_small',
		# All images will be resized to 150x150
		target_size=(150, 150),
		batch_size=20,
		# Since we use binary_crossentropy loss, we need binary labels
		class_mode='categorical')

	validation_generator = test_datagen.flow_from_directory(
		r'E:\DataSets\CatsAndDogs\train\validation',
		target_size=(150, 150),
		batch_size=20,
		class_mode='categorical')

	for data_batch, labels_batch in train_generator:
		print('data batch shape:', data_batch.shape)
		print('labels batch shape:', labels_batch.shape)
		break

	input_shape = (150, 150, 3)

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

	# output layer 2 classes
	model.add(tf.keras.layers.Dense(2))
	model.add(tf.keras.layers.Softmax())

	model.compile(optimizer = tf.keras.optimizers.Adam(0.01),
				  loss = tf.keras.losses.sparse_categorical_crossentropy,
				  metrics=['accuracy'])

	# print model architecture
	print(model.summary())

	#model.fit(x_train, y_train, epochs=1, batch_size=32)
	history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	print("acc: " + acc + "val_acc:" + val_acc)
	print("loss: " + loss + "val_loss:" + val_loss)

	# save model
	model_name = 'mnist_model.h5'
	model.save(model_name)

	# load model. This fails.
	loaded_model = tf.keras.models.load_model(model_name)

	# print test loss and metric for loaded model
	print(loaded_model.evaluate(x_test, y_test))

if __name__ == "__main__":
	tf.app.run()
