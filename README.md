# Digital Consciousness Model

A Bayesian model for evaluating consciousness in various systems using PyMC.

## Overview

This repository contains the implementation of the Digital Consciousness Model (DCM), which uses a Bayesian approach to evaluate consciousness stances and their supporting evidence through nested Beta-Bernoulli hierarchies.

## Features

- **Bayesian Modeling**: Uses PyMC for probabilistic programming
- **Evidence Integration**: Incorporates multiple levels of evidence through hierarchical models
- **Flexible Configuration**: Configurable evidential strength constants and sampling parameters
- **API Integration**: Fetches model specifications from the DCM API
- **Results Management**: Processes and posts model results to external endpoints

## Installation

### Requirements

- Python 3.8+
- PyMC
- NumPy
- Requests

### Setup

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pymc numpy requests
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
- API endpoints

### Key Classes

- **DataFetcher**: Handles data retrieval from external APIs
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

## API Integration

The model fetches specifications from:
```
https://dcm.rethinkpriorities.org/schemes/133/json
```

Results are posted to:
```
https://dcm.rethinkpriorities.org/model_runs
```

## License

[Add your license information here]

## Citation

[Add citation information here]

## Contact

[Add contact information here]
