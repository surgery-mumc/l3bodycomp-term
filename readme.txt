README
======

Before running the tool create a new virtual environment:

    python -m venv .venv/l3bodycomp-term

Then activate the virtual environment

    source .venv/l3bodycomp-term/bin/activate

Then install the required Python packages in this environment using the requirements.txt file

    pip install -r requirements.txt

You will have received 3 files:

    - model.zip
    - contour_model.zip
    - params.json

Unpack the ZIP files to separate directories, e.g., "./model" and "./contour_model".

In the file validate_model.sh set the following environment variables:

    export MODEL=<path to model directory>
    export CONTOUR_MODEL=<path to contour model directory>
    export PARAMS_FLE=<path to params file>
    export DATA_DIR=<path to directory containing L3 images and TAG files>
    export OUTPUT_DIR=<path to (non-existing) output directory where results are stored>

WARNING: do not use spaces in directory paths!!
WARNING: the OUTPUT_DIR cannot exist! If you rerun the tool, delete it first

Then run the tool (with virtual environment activated):

    ./validate_model.sh
