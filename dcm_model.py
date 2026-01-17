"""
Model 9 (Monte‑Carlo + Beta/Bernoulli) implemented from a JSON specification.

A Bayesian model for evaluating stances and their supporting evidence using PyMC.
Each stance is modeled as a Bernoulli RV with Beta priors, and evidence is
incorporated through nested Beta-Bernoulli hierarchies.
"""

from __future__ import annotations

# for arvo's setup:
# import pytensor
# pytensor.config.gcc__cxxflags = "-fbracket-depth=1024"  # or higher, try 2048
import pytensor
import os

# Set unique compiledir per process (e.g., based on process ID)
pytensor.config.compiledir = f'/tmp/pytensor_cache_{os.getpid()}'

import json
import logging
import random
import requests
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pymc as pm


@dataclass
class ModelConfig:
    """Configuration parameters for the Bayesian model."""

    # Evidential strength constants
    BASE: int = 3
    OVERWHELMING: int = 50
    STRONG: int = 8
    MODERATE: int = 3
    WEAK: float = 1.5

    # Sampling parameters
    NUM_SAMPLES: int = 3000
    NUM_TUNE: int = 500
    NUM_CHAINS: int = 4

    # Prior parameters
    DEFAULT_ALPHA: int = 1
    DEFAULT_BETA: int = 5

    # API endpoint
    API_ENDPOINT: str = (
        "https://dcm.rethinkpriorities.org/schemes/133/json?just_one=true&variant=0&subvariant=0"
    )
    POST_ENDPOINT: str = "https://dcm.rethinkpriorities.org/model_runs"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


class DataFetcher:
    """Handles data retrieval from external APIs."""

    def __init__(self, api_endpoint: str):
        self.api_endpoint = api_endpoint
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch_data(self) -> List[Dict]:
        """Fetch model data from the API."""
        self.logger.info(f"Fetching data from API: {self.api_endpoint}")
        try:
            response = requests.get(self.api_endpoint)
            response.raise_for_status()
            data = response.json()
            self.logger.info(f"Successfully fetched {len(data)} data items")
            return data
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch data from API: {e}")
            raise RuntimeError(f"Failed to fetch data from API: {e}")


class EvidenceProcessor:
    """Handles processing of evidence and observations."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_random_system(self, obj: Dict) -> Tuple[str, int, str]:
        """Recursively find a random system with observations."""
        self.logger.debug("Finding random system with observations")
        if "observations" in obj:
            systems = list(set(obj["observations"].keys()))
            system = random.choice(systems)
            name = random.choice(obj["observations"][system]["names"])
            position = obj["observations"][system]["names"].index(name)
            self.logger.info(
                f"Selected system: {system}, observation: {name} (position {position})"
            )
            return (system, position, name)
        elif "evidencers" in obj:
            try:
                random_evidencer = random.choice(obj["evidencers"])
            except:
                random_evidencer = random.choice(obj["evidencers"])
            return self.get_random_system(random_evidencer)

        else:
            self.logger.error("No observations found in the data structure")
            raise ValueError("No observations found in the data structure")

    def convert_observations_to_values(
        self, observations: List[str], position: int
    ) -> List[int]:
        """Convert observation probabilities to binary values, filtering out -1 values."""
        raw_value = float(observations[position])

        # Skip -1 values entirely
        if raw_value == -1:
            self.logger.debug(
                f"Skipping observation with value -1 at position {position}"
            )
            return []

        # Convert valid observations to binary
        binary_value = 0 if raw_value < random.random() else 1
        return [binary_value]

    def get_beta_parameters(
        self, support: str, demandingness: str
    ) -> Tuple[float, float, float, float]:
        """
        Calculate Beta distribution parameters based on support and demandingness.

        Returns:
            Tuple of (alpha_present, beta_present, alpha_absent, beta_absent)
        """
        self.logger.debug(
            f"Calculating Beta parameters for support='{support}', demandingness='{demandingness}'"
        )

        # Calculate base absence parameters
        absence_alpha, absence_beta = self._get_demandingness_parameters(demandingness)

        # Calculate support factor
        support_factor = self._get_support_factor(support, demandingness)

        presence_alpha = int(absence_alpha * support_factor[0])
        presence_beta = int(absence_beta * support_factor[1])
        params = (
            presence_alpha * 10 / (presence_alpha + presence_beta),
            presence_beta * 10 / (presence_alpha + presence_beta),
            absence_alpha * 10 / (absence_alpha + absence_beta),
            absence_beta * 10 / (absence_alpha + absence_beta),
        )

        self.logger.debug(
            f"Beta parameters: present=({params[0]}, {params[1]}), absent=({params[2]}, {params[3]})"
        )
        return params

    def _get_demandingness_parameters(self, demandingness: str) -> Tuple[int, int]:
        """Get base parameters for demandingness level."""
        base = self.config.BASE

        demandingness_map = {
            "overwhelmingly demanding": (base, int(base * self.config.OVERWHELMING)),
            "strongly demanding": (base, int(base * self.config.STRONG)),
            "moderately demanding": (base, int(base * self.config.MODERATE)),
            "weakly demanding": (base, int(base * self.config.WEAK)),
            "neutral": (base, base),
            "weakly undemanding": (int(base * self.config.WEAK), base),
            "moderately undemanding": (int(base * self.config.MODERATE), base),
            "strongly undemanding": (int(base * self.config.STRONG), base),
            "overwhelmingly undemanding": (int(base * self.config.OVERWHELMING), base),
        }

        if demandingness not in demandingness_map:
            raise ValueError(f"Unrecognized demandingness: {demandingness!r}")

        return demandingness_map[demandingness]

    def _get_support_factor(
        self, support: str, demandingness: str
    ) -> Tuple[float, float]:
        """Get multiplicative factor for support level."""
        # Calculate demandingness factor for support scaling
        demandingness_factor = {
            "overwhelmingly demanding": self.config.STRONG,
            "strongly demanding": self.config.MODERATE,
            "moderately demanding": self.config.WEAK,
            "weakly demanding": 1,
        }.get(demandingness, 1)

        support_map = {
            "overwhelming support": (
                self.config.OVERWHELMING * demandingness_factor,
                1,
            ),
            "strong support": (self.config.STRONG * demandingness_factor, 1),
            "moderate support": (self.config.MODERATE * demandingness_factor, 1),
            "weak support": (self.config.WEAK * demandingness_factor, 1),
            "no bearing": (1, 1),
            "overwhelming undermining": (1, self.config.OVERWHELMING),
            "strong undermining": (1, self.config.STRONG),
            "moderate undermining": (1, self.config.MODERATE),
            "weak undermining": (1, self.config.WEAK),
        }

        if support not in support_map:
            raise ValueError(f"Unrecognized support: {support!r}")

        return support_map[support]


class BayesianModelBuilder:
    """Builds and manages PyMC Bayesian models."""

    def __init__(self, config: ModelConfig, evidence_processor: EvidenceProcessor):
        self.config = config
        self.evidence_processor = evidence_processor
        self.variable_names = []
        self.candidate_system = None
        self.observation_position = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_observation_context(self, system_name: str, position: int):
        """Set the context for observations."""
        self.candidate_system = system_name
        self.observation_position = position
        self.logger.info(
            f"Set observation context: system='{system_name}', position={position}"
        )

    def sanitize_name(self, name: str) -> str:
        """Create a valid variable name from the input string."""
        sanitized = name.replace(" ", "_").lower()

        if sanitized not in self.variable_names:
            self.variable_names.append(sanitized)
            return sanitized
        else:
            # Handle duplicate names
            counter = len([x for x in self.variable_names if x.startswith(sanitized)])
            unique_name = f"{sanitized}_{counter}"
            self.variable_names.append(unique_name)
            return unique_name

    def create_beta_bernoulli_variable(
        self, evidencer: Dict, parent_node
    ) -> Optional[pm.Variable]:
        """Create a Beta-Bernoulli pair for an evidencer. Returns None if no valid observations."""
        name = self.sanitize_name(evidencer["name"])
        self.logger.debug(
            f"Creating variable for evidencer: {evidencer['name']} -> {name}"
        )

        # Get Beta parameters
        alpha_pres, beta_pres, alpha_abs, beta_abs = (
            self.evidence_processor.get_beta_parameters(
                evidencer.get("support", "no bearing"),
                evidencer.get("demandingness", "neutral"),
            )
        )

        # Create Beta priors for parent present/absent
        beta_present = pm.Beta(f"{name}_beta_pres", alpha=alpha_pres, beta=beta_pres)
        beta_absent = pm.Beta(f"{name}_beta_abs", alpha=alpha_abs, beta=beta_abs)

        # Mixture model based on parent state
        prob = pm.Deterministic(
            f"{name}_p",
            pm.math.switch(parent_node, beta_present, beta_absent),
        )

        # Handle different evidencer types
        if evidencer["type"].lower() == "indicator" and evidencer.get("observations"):
            self.logger.debug(f"Creating indicator variable for {name}")
            return self._create_indicator_variable(evidencer, name, prob)
        else:
            self.logger.debug(f"Creating Bernoulli variable for {name}")
            return pm.Bernoulli(f"{name}_bern", p=prob)

    def _create_indicator_variable(
        self, evidencer: Dict, name: str, prob
    ) -> Optional[pm.Variable]:
        """Create an indicator variable with observations. Returns None if no valid observations."""
        if not self.candidate_system or self.observation_position is None:
            raise ValueError("Observation context not set")

        obs_list = None
        # Check if this system has observations for this evidencer
        if self.candidate_system not in evidencer["observations"]:
            self.logger.debug(
                f"No observations for system {self.candidate_system} in evidencer {name}"
            )
            return None

        obs_average = evidencer["observation_stats"][self.candidate_system]["average"]
        binary_value = 0 if float(obs_average) < random.random() else 1
        obs_list = [binary_value]

        # If no valid observations (all were -1), skip this evidencer
        if not obs_list:
            self.logger.info(
                f"Skipping evidencer {name} - no valid observations (all -1 values)"
            )
            return None

        # Create observed Bernoulli variables for valid observations
        for idx, obs in enumerate(obs_list):
            if obs not in (0, 1):
                raise ValueError(f"Indicator observations must be 0/1; got {obs!r}")
            pm.Bernoulli(f"{name}_bern_obs{idx}", p=prob, observed=obs)

        # Return deterministic mean for downstream use
        return pm.Deterministic(f"{name}_evidence_mean", pm.math.mean(obs_list))

    def add_evidencer(self, parent_node, evidencer: Dict):
        """Add a single evidencer to the model."""
        self.logger.debug(
            f"Adding evidencer: {evidencer['name']} (type: {evidencer['type']})"
        )
        var_evidencer = self.create_beta_bernoulli_variable(evidencer, parent_node)

        # If the evidencer was skipped (no valid observations), don't add sub-evidencers
        if var_evidencer is None:
            self.logger.debug(
                f"Evidencer {evidencer['name']} was skipped, not adding sub-evidencers"
            )
            return

        # Recursively add sub-evidencers for features
        if evidencer["type"].lower() in {"feature", "subfeature"}:
            sub_evidencers = evidencer.get("evidencers", [])
            if sub_evidencers:
                self.logger.debug(
                    f"Adding {len(sub_evidencers)} sub-evidencers for {evidencer['name']}"
                )
                self.add_evidencers(var_evidencer, sub_evidencers)

    def add_evidencers(self, parent_node, evidencers: List[Dict]):
        """Add multiple evidencers to the model."""
        self.logger.info(f"Adding {len(evidencers)} evidencers to model")
        valid_evidencers = 0

        for i, evidencer in enumerate(evidencers, 1):
            self.logger.debug(
                f"Processing evidencer {i}/{len(evidencers)}: {evidencer['name']}"
            )
            try:
                self.add_evidencer(parent_node, evidencer)
                valid_evidencers += 1
            except Exception as e:
                self.logger.warning(f"Failed to add evidencer {evidencer['name']}: {e}")

        self.logger.info(
            f"Successfully added {valid_evidencers} out of {len(evidencers)} evidencers"
        )

    def build_model(self, model_spec: Dict):
        """Build the complete PyMC model from specification."""
        stance_name = self.sanitize_name(model_spec["name"])
        self.logger.info(f"Building PyMC model for stance: {model_spec['name']}")

        with pm.Model() as model:
            self.logger.info("Creating stance prior")
            # Create stance prior
            stance_p = pm.Beta(
                f"{stance_name}_beta",
                alpha=self.config.DEFAULT_ALPHA,
                beta=self.config.DEFAULT_BETA,
            )
            stance_bern = pm.Bernoulli(f"{stance_name}_bern", p=stance_p)

            # Add evidencers
            evidencers = model_spec.get("evidencers", [])
            if evidencers:
                self.logger.info(f"Adding {len(evidencers)} top-level evidencers")
                self.add_evidencers(stance_bern, evidencers)

            # Sample from the model
            self.logger.info(
                f"Starting MCMC sampling: {self.config.NUM_SAMPLES} samples, {self.config.NUM_TUNE} tune, {self.config.NUM_CHAINS} chains"
            )
            start_time = time.time()

            # Sample from the model - return raw arrays to avoid SQLite
            with model:
                trace_raw = pm.sample(
                    self.config.NUM_SAMPLES,
                    tune=self.config.NUM_TUNE,
                    chains=self.config.NUM_CHAINS,
                    cores=self.config.NUM_CHAINS,
                    return_inferencedata=False,  # Return raw arrays
                )

            sampling_time = time.time() - start_time
            self.logger.info(f"MCMC sampling completed in {sampling_time:.2f} seconds")

            # Convert to simple dict format to avoid ArviZ/SQLite
            self.logger.info("Converting trace to dictionary format")
            trace = self._convert_trace_to_dict(trace_raw, model)

        return trace

    def _convert_trace_to_dict(self, trace_raw, model):
        """Convert raw PyMC trace to simple dict format."""
        trace_dict = {"posterior": {}}

        for var_name in trace_raw.varnames:
            # Get the samples for this variable
            samples = trace_raw[var_name]
            trace_dict["posterior"][var_name] = type(
                "MockArray",
                (),
                {
                    "values": samples,
                    "flatten": lambda: type(
                        "MockFlat", (), {"mean": lambda: np.mean(samples)}
                    )(),
                },
            )()

        return trace_dict


class ResultsManager:
    """Manages model results and output."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_posterior_variable_name(self, data: Dict) -> str:
        """Get the posterior variable name for a data node."""
        return data["name"].replace(" ", "_").lower() + "_bern"

    def add_posterior_probabilities(
        self, data: Dict, trace, system_name: str, position: int
    ):
        """Recursively add posterior probabilities to the data structure."""
        var_name = self.get_posterior_variable_name(data)
        self.logger.debug(f"Processing posterior for: {data['name']} -> {var_name}")

        if var_name in trace["posterior"]:
            probability = trace["posterior"][var_name].values.flatten().mean()
            self.logger.debug(f"Found posterior probability: {probability:.4f}")
        elif "observations" in data and system_name in data["observations"]:
            # Check if the observation value is -1 (invalid)
            obs_value = float(data["observations"][system_name]["values"][position])
            if obs_value == -1:
                probability = -1  # Keep -1 to indicate no valid observation
                self.logger.debug("Observation value is -1, marking as invalid")
            else:
                probability = obs_value
                self.logger.debug(f"Using observation value: {probability}")
        else:
            probability = -1
            self.logger.debug("No posterior or observation found, using -1")

        data["posterior"] = probability

        # Recursively process evidencers
        if "evidencers" in data:
            self.logger.debug(f"Processing {len(data['evidencers'])} sub-evidencers")
            for evidencer in data["evidencers"]:
                if isinstance(evidencer, dict):
                    self.add_posterior_probabilities(
                        evidencer, trace, system_name, position
                    )

    def print_results(
        self, data: Dict, system_name: str, stance_name: str, indent: int = 0
    ):
        """Print model results in a hierarchical format."""
        indent_space = " " * indent

        if indent == 0:
            print(f"Running on {system_name} {stance_name}")

        print(f"{indent_space}{data['name']}")

        if "posterior" in data:
            if data["posterior"] == -1:
                print(f"{indent_space}  Posterior: N/A (no valid observations)")
            else:
                print(f"{indent_space}  Posterior: {round(data['posterior'], 4)}")

        if "demandingness" in data:
            prior = self._get_prior_probability(data["demandingness"])
            print(f"{indent_space}  Default Prior: {round(prior, 2)}")

        if "support" in data:
            print(f"{indent_space}  Support: {data['support']}")

        if "evidencers" in data:
            for evidencer in data["evidencers"]:
                self.print_results(evidencer, system_name, stance_name, indent + 2)

    def _get_prior_probability(self, demandingness: str) -> float:
        """Calculate prior probability based on demandingness."""
        base = self.config.BASE

        priors = {
            "overwhelmingly demanding": base / (base + base * self.config.OVERWHELMING),
            "strongly demanding": base / (base + base * self.config.STRONG),
            "moderately demanding": base / (base + base * self.config.MODERATE),
            "weakly demanding": base / (base + base * self.config.WEAK),
            "neutral": base / (base * 2),
            "weakly undemanding": self.config.WEAK
            * base
            / (base + base * self.config.WEAK),
            "moderately undemanding": self.config.MODERATE
            * base
            / (base + base * self.config.MODERATE),
            "strongly undemanding": base
            * self.config.STRONG
            / (base + base * self.config.STRONG),
            "overwhelmingly undemanding": base
            * self.config.OVERWHELMING
            / (base + base * self.config.OVERWHELMING),
        }

        return priors.get(demandingness, 0.5)

    def post_results(self, model_data: Dict) -> requests.Response:
        """Post results to the external API."""
        self.logger.info("Posting results to external API")
        payload = {
            "model_run": {
                "scheme_id": 133,
                "variant_number": 0,
                "subvariant_number": 0,
                "consciousness_probability": model_data["posterior"],
                "output": model_data,
                "parameters": {
                    "weights": {
                        "strong": self.config.STRONG,
                        "overwhelming": self.config.OVERWHELMING,
                        "weak": self.config.WEAK,
                        "moderate": self.config.MODERATE,
                    },
                    "samples": self.config.NUM_SAMPLES,
                    "tune": self.config.NUM_TUNE,
                    "stance_alpha": self.config.DEFAULT_ALPHA,
                    "stance_beta": self.config.DEFAULT_BETA,
                },
            }
        }

        try:
            response = requests.post(self.config.POST_ENDPOINT, json=payload)
            response.raise_for_status()
            self.logger.info("Results posted successfully")
            self.logger.debug(f"API response: {response.json()}")
            return response
        except requests.RequestException as e:
            self.logger.error(f"Failed to post results: {e}")
            raise


def main():
    """Main execution function."""
    # Set up logging
    logger = setup_logging("INFO")  # Change to "DEBUG" for more detailed output
    logger.info("Starting Bayesian Model Analysis")

    # Initialize components
    logger.info("Initializing components")
    config = ModelConfig()
    data_fetcher = DataFetcher(config.API_ENDPOINT)
    evidence_processor = EvidenceProcessor(config)
    model_builder = BayesianModelBuilder(config, evidence_processor)
    results_manager = ResultsManager(config)

    # Fetch and select data
    logger.info("Fetching and selecting data")
    all_data = data_fetcher.fetch_data()
    #  target_stances = ['Biological Analogy', 'Person-like', 'Cognitive Complexity', 'Global Workspace Theory', 'Higher Order Theory']
    target_stances = [item["name"] for item in all_data]

    selected_stance = random.choice(target_stances)
    logger.info(f"Selected stance: {selected_stance}")

    model_data = next(
        (item for item in all_data if item["name"] == selected_stance), None
    )
    #  model_data = all_data

    if not model_data:
        logger.error(f"No data found for stance: {selected_stance}")
        raise ValueError(f"No data found for stance: {selected_stance}")

    # Get random observation system
    logger.info("Selecting random observation system")
    system_name, position, observation_name = evidence_processor.get_random_system(
        model_data
    )

    # Set up model context
    logger.info("Setting up model context")
    model_builder.set_observation_context(system_name, position)
    model_data["response_set_name"] = observation_name
    model_data["candidate_system_name"] = system_name

    # Build and run model
    logger.info("Building and running Bayesian model")
    trace = model_builder.build_model(model_data)

    # Process results
    logger.info("Processing results")
    results_manager.add_posterior_probabilities(
        model_data, trace, system_name, position
    )
    posterior = (
        trace["posterior"][results_manager.get_posterior_variable_name(model_data)]
        .values.flatten()
        .mean()
    )
    logger.info(f"Main posterior probability: {posterior:.4f}")

    # Output results
    logger.info("Writing and posting results")
    results_manager.print_results(model_data, system_name, selected_stance)
    results_manager.post_results(model_data)

    logger.info("Analysis completed successfully")


if __name__ == "__main__":
    main()
