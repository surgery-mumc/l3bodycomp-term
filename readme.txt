README
======

You will have received 3 files:

- model.zip
- contour_model.zip
- params.json

Unpack the ZIP files to separate directories, e.g., "./model" and "./contour_model".

In the file *validate_model.sh* set the following environment variables:

WARNING: do not use spaces in directory paths!!

export MODEL=<path to model directory>
export CONTOUR_MODEL=<path to contour model directory>
export PARAMS_FLE=<path to params file>
export DATA_DIR=<path to directory containing L3 images and TAG files>
export OUTPUT_DIR=<path to (non-existing) output directory where results are stored>

Run the tool:

./validate_model.sh
