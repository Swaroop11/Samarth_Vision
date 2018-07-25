# app.py

import logging
import random
import time

from flask import Flask, jsonify, request
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf

app = Flask(__name__)
app.config.from_object(__name__)

# This could be added to the Flask configuration
MODEL_PATH = '/Users/GVH/Desktop/cub_train/finetune/serving/optimized_model-1.pb'

# Read the graph definition file
with open(MODEL_PATH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Load the graph stored in `graph_def` into `graph`
graph = tf.Graph() 
with graph.as_default():
    tf.import_graph_def(graph_def, name='')
    
# Enforce that no new nodes are added
graph.finalize()

# Create the session that we'll use to execute the model
sess_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement = True,
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=1
    )
)
sess = tf.Session(graph=graph, config=sess_config)

# Get the input and output operations
input_op = graph.get_operation_by_name('images')
input_tensor = input_op.outputs[0]
output_op = graph.get_operation_by_name('Predictions')
output_tensor = output_op.outputs[0]

# All we need to classify an image is:
# `sess` : we will use this session to run the graph (this is thread safe)
# `input_tensor` : we will assign the image to this placeholder
# `output_tensor` : the predictions will be stored here


@app.route('/')
def classify():

    file_path = request.args['file_path']
    app.logger.info("Classifying image %s" % (file_path),)

    # Load in an image to classify and preprocess it
    image = imread(file_path)
    image = imresize(image, [299, 299])
    image = image.astype(np.float32)
    image = (image - 128.) / 128.
    image = image.ravel()
    images = np.expand_dims(image, 0)
    
    # Get the predictions (output of the softmax) for this image
    t = time.time()
    preds = sess.run(output_tensor, {input_tensor : images})
    dt = time.time() - t
    app.logger.info("Execution time: %0.2f" % (dt * 1000.))
    
    # Single image in this batch
    predictions = preds[0]
    
    # The probabilities should sum to 1
    assert np.isclose(np.sum(predictions), 1)

    class_label = np.argmax(predictions)
    app.logger.info("Image %s classified as %d" % (file_path, class_label))

    return jsonify(predictions.tolist())

if __name__ == '__main__':

    app.run(debug=True, port=8009)