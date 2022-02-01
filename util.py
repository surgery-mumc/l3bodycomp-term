import os
import json
import pandas as pd
import numpy as np

from barbell2light.utils import duration
from barbell2light.dicom import is_dicom_file


class FileObject:

    def __init__(self, path):
        self.path = path
        self.name = os.path.split(path)[1]


class Image:

    def __init__(self, file_path):
        self.file_obj = FileObject(file_path)
        self.job_status = ''
        self.pred_file_name = None
        self.pred_file_path = None
        self.json_file_name = None
        self.json_file_path = None
        self.png_file_name = None
        self.png_file_path = None

    def save(self):
        pass


class TagFile:

    def __init__(self, file_path, image):
        self.file_obj = FileObject(file_path)
        self.image = image
        self.job_status = ''
        self.json_file_name = None
        self.json_file_path = None
        self.pixel_file_path = None

    def save(self):
        pass


def get_dcm_files(files):
    dcm_file_objs = []
    for f in files:
        if is_dicom_file(f):
            dcm_file_objs.append(f)
    return dcm_file_objs


def get_tag_files(files):
    tag_files = []
    for f in files:
        if f.name.endswith('.tag'):
            tag_files.append(f)
    return tag_files


def get_time_req(images):
    return duration(int(11 + 1 * len(images)))


def cm2inch(value):
    return value/2.54


def load_model(model_dir):
    import tensorflow as tf
    d = model_dir
    print('Loading model {}...'.format(d))
    if os.path.isfile(os.path.join(d, 'saved_model.pb')):
        return tf.keras.models.load_model(d, compile=False)
    else:
        print('[ERROR] No model found in {}'.format(d))
        return None


def load_contour_model(model_dir):
    import tensorflow as tf
    d = model_dir
    print('Loading contour model {}...'.format(d))
    if os.path.isfile(os.path.join(d, 'saved_model.pb')):
        return tf.keras.models.load_model(d, compile=False)
    else:
        print('[ERROR] No contour model found in {}'.format(d))
    return None


def load_params(params_file):
    pf = params_file
    with open(pf, 'r') as f:
        return json.load(f)


def normalize(img, min_bound, max_bound):
    img = (img - min_bound) / (max_bound - min_bound)
    img[img > 1] = 0
    img[img < 0] = 0
    c = (img - np.min(img))
    d = (np.max(img) - np.min(img))
    img = np.divide(c, d, np.zeros_like(c), where=d != 0)
    return img


def calculate_area(labels, label, pixel_spacing):
    mask = np.copy(labels)
    mask[mask != label] = 0
    mask[mask == label] = 1
    area = np.sum(mask) * (pixel_spacing[0] * pixel_spacing[1]) / 100.0
    return area


def calculate_smra(image, label, labels):
    mask = np.copy(labels)
    mask[mask != label] = 0
    mask[mask == label] = 1
    subtracted = image * mask
    smra = np.sum(subtracted) / np.sum(mask)
    return smra


def convert_labels_to_123(ground_truth):
    new_ground_truth = np.copy(ground_truth)
    new_ground_truth[new_ground_truth == 1] = 1
    new_ground_truth[new_ground_truth == 5] = 2
    new_ground_truth[new_ground_truth == 7] = 3
    return new_ground_truth


def convert_labels_to_157(prediction):
    new_prediction = np.copy(prediction)
    new_prediction[new_prediction == 1] = 1
    new_prediction[new_prediction == 2] = 5
    new_prediction[new_prediction == 3] = 7
    return new_prediction


def calculate_dice_score(ground_truth, prediction, label):
    numerator = prediction[ground_truth == label]
    numerator[numerator != label] = 0
    n = ground_truth[prediction == label]
    n[n != label] = 0
    if np.sum(numerator) != np.sum(n):
        raise RuntimeError('Mismatch in Dice score calculation!')
    denominator = (np.sum(prediction[prediction == label]) + np.sum(ground_truth[ground_truth == label]))
    dice_score = np.sum(numerator) * 2.0 / denominator
    return dice_score


def collect_scores(tag_files):
    columns = {
        'file_name': [],
        'smra_pred': [],
        'smra_true': [],
        'muscle_area_pred': [],
        'muscle_area_true': [],
        'vat_area_pred': [],
        'vat_area_true': [],
        'sat_area_pred': [],
        'sat_area_true': [],
        'dice_muscle': [],
        'dice_vat': [],
        'dice_sat': [],
    }
    # for img in images:
    for tag_file in tag_files:
        img = tag_file.image
        if img.json_file_path is not None:
            with open(img.json_file_path, 'r') as f:
                img_data = json.load(f)
                columns['file_name'].append(img.file_obj.name)
                columns['smra_pred'].append(img_data['smra'])
                columns['muscle_area_pred'].append(img_data['muscle_area'])
                columns['vat_area_pred'].append(img_data['vat_area'])
                columns['sat_area_pred'].append(img_data['sat_area'])
            # tag_file = get_tag_file_model(img)
            if tag_file.json_file_path is not None:
                with open(tag_file.json_file_path, 'r') as f:
                    tag_data = json.load(f)
                    columns['smra_true'].append(tag_data['smra'])
                    columns['muscle_area_true'].append(tag_data['muscle_area'])
                    columns['vat_area_true'].append(tag_data['vat_area'])
                    columns['sat_area_true'].append(tag_data['sat_area'])
                # TODO: calculate Dice score
                prediction = np.load(img.pred_file_path)
                ground_truth = np.load(tag_file.pixel_file_path)
                # ground_truth = convert_labels_to_123(ground_truth)
                dice_muscle = calculate_dice_score(ground_truth, prediction, label=1)
                columns['dice_muscle'].append(dice_muscle)
                dice_vat = calculate_dice_score(ground_truth, prediction, label=5)
                columns['dice_vat'].append(dice_vat)
                dice_sat = calculate_dice_score(ground_truth, prediction, label=7)
                columns['dice_sat'].append(dice_sat)
                print('dice_muscle: {}, dice_vat: {}, dice_sat: {}'.format(dice_muscle, dice_vat, dice_sat))
            else:
                columns['smra_true'].append(0)
                columns['muscle_area_true'].append(0)
                columns['vat_area_true'].append(0)
                columns['sat_area_true'].append(0)
                columns['dice_muscle'].append(0)
                columns['dice_vat'].append(0)
                columns['dice_sat'].append(0)
        else:
            print('Collecting scores for image without JSON: {}'.format(img.file_obj.name))
    return pd.DataFrame(data=columns)
