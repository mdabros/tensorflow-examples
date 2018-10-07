import tensorflow as tf
from tensorflow.python.keras import applications
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
	
	# All images will be rescaled by 1./255, 
	# and random augmentation are added to the training generator
	train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
		rescale=1./255,
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')

	# Note that validation and test generators must not use augmentations!
	validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
	test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(
		# This is the target directory
		r'E:\DataSets\CatsAndDogs\train\train_small',
		# All images will be resized to 150x150
		target_size=(150, 150),
		batch_size=20,
		# Since we use binary_crossentropy loss, we need binary labels
		class_mode='binary')

	validation_generator = validation_datagen.flow_from_directory(
		r'E:\DataSets\CatsAndDogs\train\validation',
		target_size=(150, 150),
		batch_size=20,
		class_mode='binary')

	test_generator = test_datagen.flow_from_directory(
		r'E:\DataSets\CatsAndDogs\train\test',
		target_size=(150, 150),
		batch_size=20,
		class_mode='binary')

	# Load the transfer learning model from the model hub.
	# model hub is currently not compatible with tf.keras for fine tuning. 
	# So using keras.applications instead
	# module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_035_96/feature_vector/1")
	
	# load vgg16 without final dense layers.
	vgg16_model = applications.VGG16(weights='imagenet', 
		include_top=False, 
		input_shape = (150, 150, 3))
	
	# Print pretrained architecture.
	print(vgg16_model.summary())

	## create new dense layers as final classifier.
	model = tf.keras.Sequential()
	model.add(vgg16_model)
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(256, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.5))
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

	# Freeze the vgg16 part of the network
	vgg16_model.trainable = False

	model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
		loss = tf.keras.losses.binary_crossentropy,
		metrics=['accuracy'])

	# Print model architecture.
	print(model.summary())

	# Fit model using the generator method.
	model.fit_generator(
		train_generator,
		steps_per_epoch=100,
		epochs=30,
		validation_data=validation_generator,
		validation_steps=50)

	print('Model evalaution:')
	print(model.evaluate_generator(test_generator))

	# Save model.
	print('Save model')
	model_name = 'cats_and_dogs_small_model.h5'
	model.save(model_name)

	# Load model. This fails.
	print('Load model')
	loaded_model = tf.keras.models.load_model(model_name)

	# Print test loss and metric for loaded model.
	print('Loaded model evalaution:')
	print(loaded_model.evaluate_generator(test_generator))

if __name__ == "__main__":
	tf.app.run()
