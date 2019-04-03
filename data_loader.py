import numpy as np
import os
import keras.backend as K

from utils import csv_reader as csv
from utils import img_util as img


def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        if x.ndim == 3:
            # 'RGB'->'BGR'
            x = x[::-1, ...]
            # Zero-center by mean pixel
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x = x[:, ::-1, ...]
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
        # Zero-center by mean pixel
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    x *= 0.017 # scale values

    return x

'''
returns (trainX, trainY), (testX, testY)
trainX.shape: 50000, 320, 320, 3
trainY.shape: 50000, 14

testX.shape: 10000, 320, 320, 3
testY.shape: 10000, 14

may have to change the resolution as images may be in some other resolution.

image_folder:
    train.csv
    train/
    valid.csv
    valid/

read train.csv:
    ignore 1st line
    global_nd_array = []
    for each line:
        1st part is image path
        last part is label
        image_nd_array = convert_image(image_path)
        global_nd_array.append()
'''

'''
replace:
    empty by 0
    -1 by 0
'''
def replace_label(item):
    if item == '':
        return 0.0
    else:
        _fitem = float(item)
        if _fitem == -1.0:
            return 1   # multi_class classification
        else:
            return _fitem
        
def filter_labels(labels: list, allowed_indices=[0,2,6,8,9,10,12]):
    _val = []
    for idx, value in enumerate(labels):
        if idx in allowed_indices:
            _val.append(value)
    return _val

def process_line(parts, image_dir):
    rel_path = parts[0]
    label_vec = list(map(lambda x: replace_label(x), filter_labels(parts[5:])))
    image = img.convert_image(os.path.join(image_dir, rel_path))
    return image, label_vec

def load_data_sub(_file, image_dir):
    x_data = []
    x_label = []
    csv_data = csv.read_csv(_file)
    csv_to_use = csv_data[1:]
    for idx, parts in enumerate(csv_to_use):
        if idx == 1000:
            break
        image, label_vec = process_line(parts, image_dir)
        x_data.append(image)
        x_label.append(label_vec)
    return x_data, x_label

def load_data_file(csv_file, image_dir):
    _x, _y = load_data_sub(csv_file, image_dir)
    return process_data(_x, _y)


def process_data(_features, _labels):
    _type = 'float32'
    _features = preprocess_input(np.array(_features).astype(_type))
    _labels = np.array(_labels).astype(_type)
    return _features, _labels