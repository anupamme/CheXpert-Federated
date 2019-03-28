'''
1. Load the entire data
2. Convert it into hdf5 format
3. write to file.
4. call federated learning routine
'''

from __future__ import absolute_import, division, print_function


import collections
from six.moves import range
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated import python as tff
from tensorflow_federated.python.simulation import hdf5_client_data
import os
import tempfile
import h5py
import six

from configparser import ConfigParser
import data_loader as dl
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
csv_dir = cp["TRAIN"].get("dataset_csv_dir")
image_source_dir = cp["DEFAULT"].get("image_source_dir")
batch_size = cp["TRAIN"].getint("batch_size")
epochs = cp["TRAIN"].getint("epochs")
client_list = cp["FEDERATED"].get("client_list").split(",")
shuffle_buffer = cp["FEDERATED"].getint("shuffle_buffer")
num_clients = cp["FEDERATED"].getint("number_clients")

def load_data():
    output = {}
    for client_item in client_list:
        _file_name = client_item + '.csv'
        _file_path = os.path.join(csv_dir, _file_name)
        _x, _y = dl.load_data_file(_file_path, image_source_dir)
        _client_map = {}
        for idx, x_item in enumerate(_x):
            _client_map[str(x_item)] = _y[idx]
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
            ('x', tf.reshape(element['pixels'], [-1])),
            ('y', tf.reshape(element['label'], [1])),
        ])

    return dataset.repeat(epochs).map(element_fn).shuffle(
        shuffle_buffer).batch(batch_size)
#    return dataset.repeat(epochs).shuffle(shuffle_buffer).batch(batch_size)

def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x))
            for x in client_ids]

def create_compiled_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])
  
    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred))
 
    model.compile(
        loss=loss_fn,
        optimizer=gradient_descent.SGD(learning_rate=0.02),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

def model_fn():
    keras_model = create_compiled_keras_model()
    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

'''
1. create hdf5 client data object.
2. call the federated learning routine
3. 
'''
if __name__ == "__main__":
    nest = tf.contrib.framework.nest
    np.random.seed(0)
    tf.compat.v1.enable_v2_behavior()
#    client_data_train = hdf5_client_data.HDF5ClientData(create_fake_hdf5(load_data()))
    client_data_train, client_data_test = tff.simulation.datasets.emnist.load_data()
    example_dataset = client_data_train.create_tf_dataset_for_client(
    client_data_train.client_ids[0])
    preprocessed_example_dataset = preprocess(example_dataset)
    sample_clients = client_data_train.client_ids[0:num_clients]
    federated_train_data = make_federated_data(client_data_train, sample_clients)
    iterative_process = tff.learning.build_federated_averaging_process(model_fn)
    state = iterative_process.initialize()
    for round_num in range(2, 11):
        state, metrics = iterative_process.next(state, federated_train_data)
        print('round {:2d}, metrics={}'.format(round_num, metrics))