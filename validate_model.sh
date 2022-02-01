#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:.

export MODEL_DIR=/home/local/UNIMAAS/r.brecheisen/data/glasgow/models/latest/model
export CONTOUR_MODEL_DIR=/home/local/UNIMAAS/r.brecheisen/data/glasgow/models/latest/contour_model
export PARAMS_FILE=/home/local/UNIMAAS/r.brecheisen/data/glasgow/models/latest/params.json
export DATA_DIR=/home/local/UNIMAAS/r.brecheisen/data/glasgow/validation/pancreas
export OUTPUT_DIR=/home/local/UNIMAAS/r.brecheisen/data/glasgow/output

python validate_model.py \
  ${MODEL_DIR} \
  ${CONTOUR_MODEL_DIR} \
  ${PARAMS_FILE} \
  ${DATA_DIR} \
  --output_dir=${OUTPUT_DIR}
