import pydicom
import matplotlib.pyplot as plt

from barbell2light.dicom import get_pixels
from util import *

PREDICTED_MUSCLE = 1
PREDICTED_VAT = 5
PREDICTED_SAT = 7


def predict_images(images, model_dir, contour_model_dir, params_file):
    model = load_model(model_dir)
    contour_model = load_contour_model(contour_model_dir)
    params = load_params(params_file)
    for img in images:
        calculation = PredictScores(img, model, contour_model, params)
        calculation.execute()
        print(img)


class PredictScores:

    def __init__(self, image, model, contour_model, params):
        self.image = image
        self.model = model
        self.contour_model = contour_model
        self.params = params

    @staticmethod
    def create_png(image):
        image_file_path = image.file_obj.path
        image_file_dir = os.path.split(image_file_path)[0]
        image_id = os.path.splitext(os.path.split(image_file_path)[1])[0]
        prediction_file_name = '{}_pred.npy'.format(image_id)
        prediction_file_path = os.path.join(image_file_dir, prediction_file_name)
        if not os.path.isfile(prediction_file_path):
            return None
        prediction = np.load(prediction_file_path)
        image = pydicom.read_file(image_file_path)
        image = get_pixels(image, normalize=True)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        ax.axis('off')
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(prediction, cmap='viridis')
        ax.axis('off')
        png_file_name = '{}.png'.format(image_id)
        png_file_path = os.path.join(image_file_dir, png_file_name)
        plt.savefig(png_file_path, bbox_inches='tight')
        plt.close('all')
        return png_file_name, png_file_path

    def predict_contour(self, contour_model, src_img):
        ct = np.copy(src_img)
        ct = normalize(
            ct, self.params['min_bound_contour'], self.params['max_bound_contour'])
        img2 = np.expand_dims(ct, 0)
        img2 = np.expand_dims(img2, -1)
        pred = contour_model.predict([img2])
        pred_squeeze = np.squeeze(pred)
        pred_max = pred_squeeze.argmax(axis=-1)
        mask = np.uint8(pred_max)
        return mask

    def execute(self):

        if self.model is None:
            self.image.job_status = 'failed'
            self.image.save()
            return

        self.image.job_status = 'running'
        self.image.save()
        p = pydicom.read_file(self.image.file_obj.path)

        # Run segmentation
        img1 = get_pixels(p, normalize=True)
        if self.contour_model is not None:
            mask = self.predict_contour(self.contour_model, img1)
            img1 = normalize(img1, self.params['min_bound'], self.params['max_bound'])
            img1 = img1 * mask
        else:
            img1 = normalize(img1, self.params['min_bound'], self.params['max_bound'])
            print('[WARNING] Segmenting image without contour-detection')
        img1 = img1.astype(np.float32)
        img2 = np.expand_dims(img1, 0)
        img2 = np.expand_dims(img2, -1)
        pred = self.model.predict([img2])
        pred_squeeze = np.squeeze(pred)
        pred_max = pred_squeeze.argmax(axis=-1)
        pred_file_name = os.path.split(self.image.file_obj.path)[1]
        pred_file_name = os.path.splitext(pred_file_name)[0] + '_pred.npy'
        pred_file_path = os.path.join(os.path.split(self.image.file_obj.path)[0], pred_file_name)
        # # Convert 1,2,3 labels back to Alberta protocol 1,5,7 labels
        pred_max = convert_labels_to_157(pred_max)
        np.save(pred_file_path, pred_max)
        self.image.pred_file_name = pred_file_name
        self.image.pred_file_path = pred_file_path
        self.image.job_status = 'finished'

        # Create PNG
        self.image.png_file_name, self.image.png_file_path = self.create_png(self.image)

        # Calculate SMRA
        img = get_pixels(p, normalize=True)
        labels = pred_max
        smra = calculate_smra(img, PREDICTED_MUSCLE, labels)

        # Calculate muscle, SAT and VAT areas
        pixel_spacing = p.PixelSpacing
        muscle_area = calculate_area(labels, PREDICTED_MUSCLE, pixel_spacing)
        vat_area = calculate_area(labels, PREDICTED_VAT, pixel_spacing)
        sat_area = calculate_area(labels, PREDICTED_SAT, pixel_spacing)
        json_file_name = os.path.split(self.image.file_obj.path)[1]
        json_file_name = os.path.splitext(json_file_name)[0] + '.json'
        json_file_path = os.path.join(os.path.split(self.image.file_obj.path)[0], json_file_name)
        print('Predicted segmentation: SMRA = {}, muscle area = {}, VAT area = {}, SAT area = {}'.format(smra, muscle_area, vat_area, sat_area))

        # Save scores to JSON and update image object
        with open(json_file_path, 'w') as f:
            json.dump({
                'smra': smra,
                'muscle_area': muscle_area,
                'vat_area': vat_area,
                'sat_area': sat_area
            }, f, indent=4)

        self.image.json_file_name = json_file_name
        self.image.json_file_path = json_file_path
        self.image.save()

        return img, np.load(self.image.pred_file_path)
