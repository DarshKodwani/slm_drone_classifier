# SLM Drone Classifier

## Overview

This repository contains code for the SLM Drone Classifier project. The goal of this project is to use SLMs as classifiers and extract the information from prompts that might be used to control an autonomous grocery delivery drone.

## Key Files

- `slm_extracting_parameters.py`: Script for running the SLM to extract parameters from a prompt that has been passed to it. The parameters will be used as inputs to a function.
- `slm_function_calling.py`: Script for running the SLM to classify what function needs to be called based on the prompt.
- `slm_drone_prompts.json`: Example data for prompts that will be used by the SLM to classify the name of function
- `slm_parameter_extraction.json`: Example data for prompts that will be used to extract parameters from.
- `firstTimeSetup.sh`: bash script to setup the python environment to run the scripts

## Setup and Usage

### Setting up the Virtual Environment

1. Run the firstTimeSetup bash script:
    ```bash
    ./firstTimeSetup.sho
    cd slm_drone_classifier
    ```

### Running the Scripts

1. Classifying the function:
    ```bash
    python slm_function_calling.py
    ```

2. Extracting parameters:
    ```bash
    python slm_extracting_parameters.py
    ```