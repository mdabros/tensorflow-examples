import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):

	# All images will be rescaled by 1./255
	train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
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


	model = tf.keras.models.Sequential()
	
	model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(512, activation='relu'))
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

	model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=1e-4),
				  loss = tf.keras.losses.binary_crossentropy,
				  metrics=['accuracy'])

	# print model architecture
	print(model.summary())

	#model.fit(x_train, y_train, epochs=1, batch_size=32)
	model.fit_generator(
		train_generator,
		steps_per_epoch=100,
		epochs=30,
		validation_data=validation_generator,
		validation_steps=50)

	print('Model evalaution:')
	print(model.evaluate_generator(test_generator))

	# save model
	print('Save model')
	model_name = 'cats_and_dogs_small_model.h5'
	model.save(model_name)

	# load model. This fails.
	print('Load model')
	loaded_model = tf.keras.models.load_model(model_name)

	# print test loss and metric for loaded model
	print('Loaded model evalaution:')
	print(loaded_model.evaluate_generator(test_generator))

if __name__ == "__main__":
	tf.app.run()
