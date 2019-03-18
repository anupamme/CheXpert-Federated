import numpy as np
import os
import pandas as pd
from glob import glob


def get_sample_counts(output_dir, dataset, class_names):
    """
    Get total and class-wise positive sample count of a dataset

    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes

    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    df = pd.read_csv(os.path.join(output_dir, dataset + ".csv"))
    total_count = df.shape[0]
    labels = df[class_names].as_matrix()
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts

'''
1. write header: Path,Sex,Age,Frontal/Lateral,AP/PA,No_Finding,Enlarged_Cardiomediastinum,Cardiomegaly,Lung_Opacity,Lung_Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural_Effusion,Pleural_Other,Fracture,Support_Devices
2. write in format:
    folder_name/filename.jpg,_,_,_,_,-1,-1,-1,-1 (14 times)
'''
def create_csv(image_folder, file_name):
    _header = 'Path,Sex,Age,Frontal/Lateral,AP/PA,No_Finding,Enlarged_Cardiomediastinum,Cardiomegaly,Lung_Opacity,Lung_Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural_Effusion,Pleural_Other,Fracture,Support_Devices'
    f = open(filename, 'w')
    f.write(_header + '\n')
    for item in glob("*.jpg"):
        _path = os.path.join(image_folder, item)
        _line = _path + ',_,_,_,_,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1'
        f.write(_line + '\n')
    f.close()
