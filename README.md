# Digital Consciousness Model

A Bayesian model for evaluating consciousness in various systems using PyMC.

## Overview

This repository contains the implementation of the Digital Consciousness Model (DCM), which uses a Bayesian approach to evaluate consciousness stances and their supporting evidence through nested Beta-Bernoulli hierarchies.

## Features

- **Bayesian Modeling**: Uses PyMC for probabilistic programming
- **Evidence Integration**: Incorporates multiple levels of evidence through hierarchical models
- **Flexible Configuration**: Configurable evidential strength constants and sampling parameters
- **Fully Offline**: Model specifications are read from the local `model_data.json` file; no network access is required (this is a recent update)
- **Results Management**: Processes and prints model results locally

## Installation

### Requirements

- Python 3.8+
- PyMC
- NumPy

### Setup

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pymc numpy
```

## Usage

Run the main model:

```bash
python dcm_model.py
```

Run sensitivity tests:

```bash
python sensitivity_test_model.py
```

## Model Components

### Configuration
The `ModelConfig` class contains all model parameters including:
- Evidential strength constants (BASE, OVERWHELMING, STRONG, MODERATE, WEAK)
- Sampling parameters (number of samples, tuning steps, chains)
- Prior parameters (default alpha and beta values)
- Path to the local data file

### Key Classes

- **DataFetcher**: Loads model data from the local JSON file
- **EvidenceProcessor**: Processes evidence and observations, calculates Beta parameters
- **BayesianModelBuilder**: Builds and manages PyMC Bayesian models
- **ResultsManager**: Manages model results and output

## Model Structure

The model evaluates consciousness stances using a hierarchical structure:

1. **Stance**: Top-level hypothesis with Beta prior
2. **Features**: Supporting evidence with support and demandingness levels
3. **Indicators**: Observable evidence from specific systems

Each level uses Beta-Bernoulli conjugate pairs, where:
- Beta distributions model probability beliefs
- Bernoulli variables model binary outcomes
- Support and demandingness levels determine parameter values

## JSON File location (updated recently)
The model specifications are stored in a JSON file located at:
```
model_data.json
```

### Observation averages

Both scripts generate indicator observations from the selected system's
`observation_stats[system]["average"]`. A null or `-1` average means missing
evidence and adds no observed Bernoulli variables. Other averages must be finite
probabilities between zero and one, inclusive; numeric strings are supported.

Invalid averages raise a `ValueError` identifying the indicator and system.
The error stops model construction before MCMC sampling instead of silently
dropping the invalid indicator. The scripts do not clamp, recompute or repair
the supplied expert data.

### Regression tests

After installing the dependencies, run:

```bash
python -m unittest discover -s tests -v
```

These tests construct real PyMC graphs and check missing, invalid and valid
observations, pooled draw order, and error propagation. The invalid-input test
intercepts the sampler boundary; the suite does not run posterior sampling.

## License

This project is licensed under the MIT License.

## Citation

https://arxiv.org/abs/2601.17060
