#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import reader
import math

tf.logging.set_verbosity(tf.logging.INFO)

data, labels, class_num = reader.read_raw_data()  # Returns np.array
CLASS_NUM = class_num
lenx, leny = data.shape
LAYER_SHAPE_X = int(math.sqrt(leny))
LAYER_SHAPE_Y = int(math.sqrt(leny))
TRAIN_DATA = data[0 : lenx - 20, :]
TRAIN_LABELS = labels[0 : lenx - 20, :]
#train_labels = reader.readtrainlabel("train")
PRED_DATA = data[lenx - 20 : lenx, :]
PRED_LABELS = labels[lenx - 20 : lenx, :]  # Returns np.array
#print(train_data.shape)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel

  input_layer = tf.reshape(features["x"], [-1, LAYER_SHAPE_X, LAYER_SHAPE_Y, 1])
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 7, 7, 1]
  # Output Tensor Shape: [batch_size, 7, 7, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  #print(pool1.shape)
  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 7, 7, 32]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[7, 7],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  x = LAYER_SHAPE_X // 2
  x = x // 2
  y = LAYER_SHAPE_Y // 2
  y = y // 2
  pool2_flat = tf.reshape(pool2, [-1, x * y * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=CLASS_NUM)

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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    train_loss_summary = tf.summary.scalar(name='summary', tensor=loss)
    summary_merge = tf.summary.merge([train_loss_summary])
    writer = tf.summary.FileWriter("./summary/plot_1")
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


def main(unused_argv):
  # Load training and eval data
  #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  # data, labels, class_num = reader.read_raw_data()  # Returns np.array
  # CLASS_NUM = class_num
  # lenx, leny = data.shape
  # LAYER_SHAPE_X = int(math.sqrt(leny))
  # LAYER_SHAPE_Y = int(math.sqrt(leny))
  # train_data = data[0 : lenx - 20, :]
  # train_labels = labels[0 : lenx - 20, :]
  # #train_labels = reader.readtrainlabel("train")
  # pred_data = data[lenx - 20 : lenx, :]
  # pred_labels = labels[lenx - 20 : lenx, :]  # Returns np.array
  # print(train_data.shape)
  train_data = TRAIN_DATA
  train_labels = TRAIN_LABELS
  pred_data = PRED_DATA
  pred_labels = PRED_LABELS
  train_data=train_data.astype('float32')
  pred_data=pred_data.astype('float32')
  #train_labels=train_labels.astype('float32')
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir = './model')

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  tf.summary.merge_all()
  #summary_hook = tf.train.SummarySaverHook(
  #  save_steps=10, output_dir = "./summary/", summary_op='summary')
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=lenx - 20,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=5000,
      hooks=[logging_hook])
  print("Comming back")
  # Evaluate the model and print results

  pred_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": pred_data},
      y=pred_labels,
      num_epochs=1,
      shuffle=False)
  pred_results = mnist_classifier.predict(input_fn=pred_input_fn)
  result_labels = [result['classes'] for result in pred_results]
  err = 0
  for i in range(len(result_labels)):
    if result_labels[i] != pred_labels[i]:
      err = err + 1
  print(err / len(result_labels))
  #for result in pred_results:
  #  print(result['classes'])


if __name__ == "__main__":
  tf.app.run()
