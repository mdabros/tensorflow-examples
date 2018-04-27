import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # Setup network

    # Input layer
    net = adjust_image(features["x"])

    # Convolutional Layers 1
    net = tf.layers.conv2d(net, 32, (3, 3), padding="same")
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(net, 32, (3, 3), padding="same")
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(net, 32, (3, 3), padding="same")
    net = tf.nn.relu(net)

    # Pooling Layer 1
    net = tf.layers.max_pooling2d(net, (2, 2), strides=2)

    # Convolutional Layers 2
    net = tf.layers.conv2d(net, 32, (3, 3), padding="same")
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(net, 32, (3, 3), padding="same")
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(net, 32, (3, 3), padding="same")
    net = tf.nn.relu(net)

    # Pooling Layer 2
    net = tf.layers.max_pooling2d(net, (2, 2), strides=2)

    # Dense Layer
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, units=256)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(net, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def adjust_image(data):
    # Reshaped to [batch, height, width, channels].
    imgs = tf.reshape(data, [-1, 28, 28, 1])

    return imgs


def main(unused_argv):
    with tf.Graph().as_default() as g:
        # Load MNIST data.
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images  # Returns np.array
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        eval_data = mnist.test.images  # Returns np.array
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        # create input functions for train and evaluate methods.
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=64,
            num_epochs=10,
            shuffle=True)

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        
        # clean up previous model
        model_directory = os.getcwd() + "/cnn-model/"
        if os.path.exists(model_directory):
            os.remove(model_directory)

        # Create an estimator
        classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir=model_directory)

        # Train network.
        classifier.train(input_fn=train_input_fn)

        # Evaluate the model and print results.
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


if __name__ == "__main__":
    tf.app.run()
