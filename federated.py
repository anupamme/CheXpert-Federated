#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import collections
from six.moves import range
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent
import tensorflow_federated as tff
from tensorflow_federated.python.simulation import hdf5_client_data
from tensorflow.keras.optimizers import Adam
import os
import tempfile
import h5py
import six

from configparser import ConfigParser
import data_loader as dl


# In[10]:


'''
1. Put the images in data/images
2. Put the ground truth csv files like this: 
    in data/default_split/client_1.csv, data/default_split/client_2.csv
3. in sample_config_ini: list_of_clients: [client_1, client_2]
4. 
'''
# parser config
config_file = "./sample_config.ini"
cp = ConfigParser()
cp.read(config_file)

output_dir = cp["DEFAULT"].get("output_dir")
class_names = cp["DEFAULT"].get("class_names")
base_model_name = cp["DEFAULT"].get("base_model_name")
image_source_dir = cp["DEFAULT"].get("image_source_dir")

csv_dir = cp["TRAIN"].get("dataset_csv_dir")
batch_size = cp["TRAIN"].getint("batch_size")
epochs = cp["TRAIN"].getint("epochs")
output_weights_name = cp["TRAIN"].get("output_weights_name")
model_weights_path = os.path.join(output_dir, output_weights_name)
initial_learning_rate = cp["TRAIN"].getfloat("initial_learning_rate")

client_list = cp["FEDERATED"].get("client_list").split(",")
shuffle_buffer = cp["FEDERATED"].getint("shuffle_buffer")
num_clients = cp["FEDERATED"].getint("number_clients")

output_dir, class_names, base_model_name, image_source_dir, csv_dir, batch_size, epochs, output_weights_name, model_weights_path, initial_learning_rate, client_list, shuffle_buffer, num_clients

np.random.seed(0)

nest = tf.contrib.framework.nest
tf.compat.v1.enable_v2_behavior()

def load_data():
    output = {}
    for client_item in client_list:
        _file_name = client_item + '.csv'
        _file_path = os.path.join(csv_dir, _file_name)
        _x, _y = dl.load_data_file(_file_path, image_source_dir)
        _client_map = {}
        _client_map['label'] = _y
        _client_map['pixels'] = _x
        output[client_item] = _client_map
    return output

def create_fake_hdf5(arg_data):
    fd, filepath = tempfile.mkstemp()
    # close the pre-opened file descriptor immediately to avoid leaking.
    os.close(fd)
    with h5py.File(filepath, 'w') as f:
        examples_group = f.create_group('examples')
        for user_id, data in six.iteritems(arg_data):
            user_group = examples_group.create_group(user_id)
            for name, values in six.iteritems(data):
                user_group.create_dataset(name, data=values)
    return filepath

def preprocess(dataset):
    def element_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['pixels'], element['pixels'].shape)),
            ('y', tf.reshape(element['label'], element['label'].shape)),
        ])

    return dataset.repeat(epochs).map(element_fn).shuffle(
        shuffle_buffer).batch(batch_size)

client_data_train = hdf5_client_data.HDF5ClientData( create_fake_hdf5(load_data()) )
example_dataset = client_data_train.create_tf_dataset_for_client(
    client_data_train.client_ids[0]
)
preprocessed_example_dataset = preprocess(example_dataset)
sample_batch = nest.map_structure(
    lambda x: x.numpy(), iter(preprocessed_example_dataset).next()
)


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x))
            for x in client_ids]

sample_clients = client_data_train.client_ids[0:num_clients]
federated_train_data = make_federated_data(client_data_train, sample_clients)


def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(
        y_true, y_pred))

def create_compiled_keras_model():
    model = tf.keras.applications.densenet.DenseNet121(include_top=True, input_tensor=None, input_shape=None, pooling="avg", weights=model_weights_path, classes=7)
#    optimizer = Adam(lr=initial_learning_rate)
    optimizer = gradient_descent.SGD(learning_rate=0.02)
    model.compile(
        loss = ['binary_crossentropy'],
        optimizer=optimizer,
#        metrics=[MyBinaryCrossentropy()]
        metrics=[tf.keras.metrics.BinaryAccuracy()]
#        metrics=[tf.keras.metrics.MeanSquaredError()]
    )
    return model

def model_fn():
    keras_model = create_compiled_keras_model()
    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

iterative_process = tff.learning.build_federated_averaging_process(model_fn)
print('done build_federated_averaging_process!')
state = iterative_process.initialize()
for round_num in range(1, 11):
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, metrics))

print('done training!')