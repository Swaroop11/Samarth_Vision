#Flask RESTApi App for Cloud

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import random
import time
import os
from flask import Flask, jsonify, request, render_template

import numpy as np
import tensorflow as tf

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

print(APP_ROOT)

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  image_reader = tf.image.decode_image(file_reader, channels=3, name="image_gen")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

@app.route('/')
def index():
    return render_template("upload.html")

@app.route("/upload",methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'static\\images\\')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist('file'):
       print(file)
       filename = file.filename
       destination = "".join([target, filename])
       print(destination)
       file.save(destination)

    file_name = destination
    print(file_name)

	#file_name = request.args['file']

    t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
        end=time.time()
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

    for i in top_k:
         print(labels[i], results[i])

    unwell = results[0]
    print("Unwellness of person is : " ,unwell)
    well = results[1]
    print("Wellness of person is : " ,well)
    print(destination)
    return render_template("complete.html" ,well = well, unwell = unwell,filename=filename)


if __name__ == '__main__':
    # TensorFlow configuration/initialization
    model_file = "retrained_graph.pb"
    label_file = "retrained_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Placeholder"
    output_layer = "final_result"

    # Load TensorFlow Graph from disk
    graph = load_graph(model_file)

    # Grab the Input/Output operations
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    # Initialize the Flask Service
    # Obviously, disable Debug in actual Production
    app.run(debug=False, port=8080)	#Debug =False in true production
