import os
import shutil
import argparse

from validate import *
from barbell2light.dicom import is_dicom_file, get_tag_file_for_dicom


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('tf_model_dir', help='Directory containing TensorFlow model files')
    parser.add_argument('tf_contour_model_dir', help='ZIP file containing TensorFlow contour model (optional)', default='')
    parser.add_argument('tf_params_file', help='JSON file containing TensorFlow model parameters')
    parser.add_argument('l3_root_dir', help='Root directory containing L3 images and TAG files')
    parser.add_argument('--output_dir', help='Output directory where to store calculated results (cannot already exist)', default='./output')
    return parser.parse_args()


def main(args):

    import tensorflow as tf
    print(f'is_gpu_available() {tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)}')
    print(f'list_physical_devices() {tf.config.list_physical_devices("GPU")}')
    print(f'is_built_with_cuda() {tf.test.is_built_with_cuda()}')

    shutil.copytree(args.l3_root_dir, args.output_dir)

    tag_files = []

    for f in os.listdir(args.output_dir):
        if not f.startswith('._'):
            file_path = os.path.join(args.output_dir, f)
            if is_dicom_file(file_path):
                img_file_path = file_path
                tag_file_path = get_tag_file_for_dicom(img_file_path)
                if tag_file_path is not None:
                    img_file = Image(img_file_path)
                    tag_file = TagFile(tag_file_path, img_file)
                    tag_files.append(tag_file)

    validate_model_on_images_and_tag_files(
        tag_files, args.tf_model_dir, args.tf_contour_model_dir, args.tf_params_file)

    scores = collect_scores(tag_files)
    scores.to_csv(os.path.join(args.output_dir, 'scores.csv'))


if __name__ == '__main__':
    import sys
    # sys.argv = [
    #     'validate_l3_autoseg_model.py',
    #     '/Users/Ralph/data/surfdrive/projects/hpb/bodycomp/20211109_demo_glasgow/models/latest/model',
    #     '/Users/Ralph/data/surfdrive/projects/hpb/bodycomp/20211109_demo_glasgow/models/latest/contour_model',
    #     '/Users/Ralph/data/surfdrive/projects/hpb/bodycomp/20211109_demo_glasgow/models/latest/params.json',
    #     '/Users/Ralph/data/surfdrive/projects/hpb/bodycomp/20211109_demo_glasgow/validation/trauma',
    # ]
    sys.argv = [
        'validate_l3_autoseg_model.py',
        '/home/local/UNIMAAS/r.brecheisen/data/glasgow/models/latest/model',
        '/home/local/UNIMAAS/r.brecheisen/data/glasgow/models/latest/contour_model',
        '/home/local/UNIMAAS/r.brecheisen/data/glasgow/models/latest/params.json',
        '/home/local/UNIMAAS/r.brecheisen/data/glasgow/validation/pancreas',
        '--output=/home/local/UNIMAAS/r.brecheisen/data/glasgow/output',
    ]
    main(get_args())
