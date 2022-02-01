import pydicom
import numpy as np
import matplotlib.pyplot as plt

from barbell2light.dicom import get_pixels, tag2numpy
from predict import PredictScores
from util import *

# Labels for different tissue types. Note that Leroy's algorithm outputs labels 1, 2, and 3
# for these respective tissues!
TAG_MUSCLE = 1
TAG_VAT = 5
TAG_SAT = 7


def validate_model_on_images_and_tag_files(tag_files, model_dir, contour_model_dir, params_file):
    model = load_model(model_dir)
    contour_model = load_contour_model(contour_model_dir)
    params = load_params(params_file)
    for tag_file in tag_files:
        tag_file.image.job_status = 'running'
        tag_file.image.save()
        ground_truth = ValidateScores(tag_file)
        ground_truth.execute()
        prediction = PredictScores(tag_file.image, model, contour_model, params)
        prediction.execute()


class ValidateScores:

    def __init__(self, tag_file_model):
        self.tag_file_model = tag_file_model

    @staticmethod
    def update_labels(pixels, file_name):
        # http://www.tomovision.com/Sarcopenia_Help/index.htm
        labels_to_keep = [0, 1, 5, 7]
        labels_to_remove = [2, 12, 14]
        for label in np.unique(pixels):
            if label in labels_to_remove:
                pixels[pixels == label] = 0
        for label in np.unique(pixels):
            if label not in labels_to_keep:
                print(f'label {label} not in {labels_to_keep}')
                return None
        if len(np.unique(pixels)) != 4:
            print('[{}] Incorrect nr. of labels: {}'.format(file_name, len(np.unique(pixels))))
            return None
        return pixels

    def get_tag_file_pixels(self, tag_file_model, shape):
        converter = tag2numpy.Tag2NumPy(shape)
        converter.set_input_tag_file_path(tag_file_model.file_obj.path)
        converter.execute()
        pixels = converter.get_output_numpy_array()
        return self.update_labels(pixels, tag_file_model.file_obj.name)

    def create_png(self, tag_file_model):
        image_file_path = tag_file_model.image.file_obj.path
        image_file_dir = os.path.split(image_file_path)[0]
        image_id = os.path.splitext(os.path.split(image_file_path)[1])[0]
        prediction_file_name = '{}_pred.npy'.format(image_id)
        prediction_file_path = os.path.join(image_file_dir, prediction_file_name)
        # If there is no prediction yet, return. First run PredictScores
        if not os.path.isfile(prediction_file_path):
            return None
        prediction_pixels = np.load(prediction_file_path)
        image = pydicom.read_file(image_file_path)
        image_pixels = get_pixels(image, normalize=True)
        tag_pixels = self.get_tag_file_pixels(tag_file_model, image_pixels.shape)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 3, 1)
        plt.imshow(image_pixels, cmap='gray')
        ax.axis('off')
        ax = fig.add_subplot(1, 3, 2)
        plt.imshow(tag_pixels, cmap='viridis')
        ax.axis('off')
        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(prediction_pixels, cmap='viridis')
        ax.axis('off')
        png_file_name = '{}.png'.format(image_id)
        png_file_path = os.path.join(image_file_dir, png_file_name)
        plt.savefig(png_file_path, bbox_inches='tight')
        plt.close('all')
        return png_file_path

    def execute(self):
        image_model = self.tag_file_model.image
        p = pydicom.read_file(image_model.file_obj.path)
        pixel_spacing = p.PixelSpacing
        image_pixels = get_pixels(p, normalize=True)
        tag_pixels = self.get_tag_file_pixels(self.tag_file_model, image_pixels.shape)
        if tag_pixels is not None:
            smra = calculate_smra(image_pixels, TAG_MUSCLE, tag_pixels)
            muscle_area = calculate_area(tag_pixels, TAG_MUSCLE, pixel_spacing)
            vat_area = calculate_area(tag_pixels, TAG_VAT, pixel_spacing)
            sat_area = calculate_area(tag_pixels, TAG_SAT, pixel_spacing)
            print('TAG file: SMRA = {}, muscle area = {}, VAT area = {}, SAT area = {}'.format(smra, muscle_area, vat_area, sat_area))
            pixel_file_name = os.path.split(self.tag_file_model.file_obj.path)[1]
            pixel_file_name = os.path.splitext(pixel_file_name)[0] + '.npy'
            pixel_file_path = os.path.join(os.path.split(self.tag_file_model.file_obj.path)[0], pixel_file_name)
            np.save(pixel_file_path, tag_pixels)
        else:
            print('tag_pixels is None')
            return

        json_file_name = os.path.split(self.tag_file_model.file_obj.path)[1]
        json_file_name = os.path.splitext(json_file_name)[0] + '_tag.json'
        json_file_path = os.path.join(os.path.split(self.tag_file_model.file_obj.path)[0], json_file_name)

        with open(json_file_path, 'w') as f:
            json.dump({
                'smra': smra,
                'muscle_area': muscle_area,
                'vat_area': vat_area,
                'sat_area': sat_area
            }, f, indent=4)
        print(f'Written JSON file path {json_file_path}')

        self.tag_file_model.pixel_file_path = pixel_file_path
        self.tag_file_model.json_file_name = json_file_name
        self.tag_file_model.json_file_path = json_file_path
        self.tag_file_model.save()
