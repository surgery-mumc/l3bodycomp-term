#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:.

python validate_model.py \
  validate_l3_autoseg_model.py \
  "/home/local/UNIMAAS/r.brecheisen/data/glasgow/models/latest/model" \
  "/home/local/UNIMAAS/r.brecheisen/data/glasgow/models/latest/contour_model" \
  "/home/local/UNIMAAS/r.brecheisen/data/glasgow/models/latest/params.json" \
  "/home/local/UNIMAAS/r.brecheisen/data/glasgow/validation/pancreas" \
  --output=/home/local/UNIMAAS/r.brecheisen/data/glasgow/output
