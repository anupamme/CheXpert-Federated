import numpy as np
import os
import math

from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.keras import ModelFactory
from sklearn.metrics import roc_auc_score

from utility import get_sample_counts
from utility import create_csv

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("file", type=str)

# parser config
config_file = "./sample_config.ini"
cp = ConfigParser()
cp.read(config_file)

# default config
output_dir = cp["DEFAULT"].get("output_dir")
base_model_name = cp["DEFAULT"].get("base_model_name")
class_names = cp["DEFAULT"].get("class_names").split(",")
image_source_dir = cp["DEFAULT"].get("image_source_dir")

# train config
image_dimension = cp["TRAIN"].getint("image_dimension")

# test config
batch_size = cp["TEST"].getint("batch_size")
test_steps = cp["TEST"].get("test_steps")
use_best_weights = cp["TEST"].getboolean("use_best_weights")

# parse weights file path
output_weights_name = cp["TRAIN"].get("output_weights_name")
weights_path = os.path.join(output_dir, output_weights_name)
best_weights_path = os.path.join(output_dir, "best_{output_weights_name}")

print("** load model **")
if use_best_weights:
    print("** use best weights **")
    model_weights_path = best_weights_path
else:
    print("** use last weights **")
    model_weights_path = weights_path

model_factory = ModelFactory()
model = model_factory.get_model(
    class_names,
    model_name=base_model_name,
    use_base_weights=False,
    weights_path=model_weights_path)

def main():
    

    # get test sample count
    test_counts, _ = get_sample_counts(output_dir, "test", class_names)

    # compute steps
    global test_steps
    if test_steps == "auto":
        test_steps = int(test_counts / batch_size)
    else:
        try:
            test_steps = int(test_steps)
        except ValueError:
            raise ValueError("""
                test_steps: {test_steps} is invalid,
                please use 'auto' or integer.
                """)
    print("** test_steps: {test_steps} **")

    

    print("** load test generator **")
    test_sequence = AugmentedImageSequence(
        dataset_csv_file=os.path.join(output_dir, "dev.csv"),
        class_names=class_names,
        source_image_dir=image_source_dir,
        batch_size=batch_size,
        target_size=(image_dimension, image_dimension),
        augmenter=None,
        steps=test_steps,
        shuffle_on_epoch_end=False,
    )

    print("** make prediction **")
    y_hat = model.predict_generator(test_sequence, verbose=1)
    y = test_sequence.get_y_true()

    test_log_path = os.path.join(output_dir, "test.log")
    print("** write log to {test_log_path} **")
    aurocs = []
    with open(test_log_path, "w") as f:
        for i in range(len(class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
                aurocs.append(score)
            except ValueError:
                score = 0
            f.write(str(class_names[i]) + ": " + str(score) + "\n")
        mean_auroc = np.mean(aurocs)
        f.write("-------------------------\n")
        f.write("mean auroc: {mean_auroc}\n")
        print("mean auroc: {mean_auroc}")
        
def do_inferencing(testX, base_val=0.5):
    yPreds = model.predict(testX)
    #yPred = np.argmax(yPreds, axis=1)
    # threshholds
    yPred = list(map(lambda x: list(map(lambda y: find_class(y, base_val), x)), yPreds))
    yPred_np = np.asarray(yPred)
    return yPred_np
        
'''
1. Predicts probability/binary vector for this image
2. 
'''
def predict_single(image_location):
    
    pass

'''
1. create csv for this image_folder

'''
def predict_all(image_folder, _test_file='test', calculate_accuracy=False):
    csv_file_name = _test_file + '.csv'
    csv_file_path = os.path.join(output_dir, csv_file_name)
    images_list = create_csv(image_source_dir, image_folder, csv_file_path)
    # get test sample count
    test_counts, _ = get_sample_counts(output_dir, _test_file, class_names)

    # compute steps
    global test_steps
    if test_steps == "auto":
        test_steps = math.ceil(test_counts / batch_size)
    else:
        try:
            test_steps = int(test_steps)
        except ValueError:
            raise ValueError("""
                test_steps: {test_steps} is invalid,
                please use 'auto' or integer.
                """)
    print("** test_steps: {test_steps} **")
    
    print("** load test generator **")
    test_sequence = AugmentedImageSequence(
        dataset_csv_file=csv_file_path,
        class_names=class_names,
        source_image_dir=image_source_dir,
        batch_size=batch_size,
        target_size=(image_dimension, image_dimension),
        augmenter=None,
        steps=test_steps,
        shuffle_on_epoch_end=False,
    )
    print("** make prediction **")
    y_hat = model.predict_generator(test_sequence, verbose=1)
#    y = test_sequence.get_y_true()
    
    # print y_hat: (224, 7)
    print("** print predictions **")
    test_log_path = os.path.join(output_dir, _test_file + ".pred")
    f = open(test_log_path, 'w')
    _header = 'image_path' + ','.join(class_names)
    f.write(_header + '\n')
    for idx, item in enumerate(y_hat):  # cases
        image_name = images_list[idx]
        output_str = ''
        for idx_score, score_item in enumerate(item): # classes
            output_str += str(class_names[idx_score]) + ": " + str(score_item)
        f.write(image_name + ', ' + output_str + '\n')
    f.close()
    
    if calculate_accuracy:
        test_log_path = os.path.join(output_dir, _test_file + ".log")
        print("** write log to {test_log_path} **")
        aurocs = []
        with open(test_log_path, "w") as f:
            for i in range(len(class_names)):
                try:
                    score = roc_auc_score(y[:, i], y_hat[:, i])
                    aurocs.append(score)
                except ValueError:
                    score = 0
                f.write(str(class_names[i]) + ": " + str(score) + "\n")
            mean_auroc = np.mean(aurocs)
            f.write("-------------------------\n")
            f.write("mean auroc: {mean_auroc}\n")
            print("mean auroc: {mean_auroc}")

if __name__ == "__main__":
    args = parser.parse_args()
    predict_all(args.file)
