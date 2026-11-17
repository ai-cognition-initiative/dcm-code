"""Observation integrity through the existing real PyMC construction paths."""

import copy
import math
import random
import unittest
from unittest.mock import patch

# The main script configures its process-local PyTensor cache before PyMC is
# imported, just as it does when run directly.
import dcm_model
import sensitivity_test_model

import numpy as np
import pymc as pm


MODULES = (dcm_model, sensitivity_test_model)
SYSTEM = "Test system"


def indicator(average):
    return {
        "name": "Test indicator",
        "type": "Indicator",
        "support": "moderate support",
        "demandingness": "weakly demanding",
        "observations": {SYSTEM: {"names": ["Example"], "values": [average]}},
        "observation_stats": {SYSTEM: {"average": average, "std": 0}},
    }


def make_builder(module, pooled=False):
    config = module.ModelConfig()
    if module is sensitivity_test_model:
        config.SENSITIVITY_TEST_VARIANT_NUMBER = 1 if pooled else 0
        config.SENSITIVITY_TEST_SUBVARIANT_NUMBER = 0
    processor = module.EvidenceProcessor(config)
    builder = module.BayesianModelBuilder(config, processor)
    builder.set_observation_context(SYSTEM, 0)
    return builder


def add_nodes(builder, nodes):
    with pm.Model() as model:
        prior = pm.Beta("test_prior", alpha=1, beta=5)
        parent = pm.Bernoulli("test_parent", p=prior)
        builder.add_evidencers(parent, nodes)
    return model


def observed_values(model):
    return [np.asarray(model.rvs_to_values[rv].data).item() for rv in model.observed_RVs]


class ObservationProbabilityTests(unittest.TestCase):
    def test_missing_averages_add_no_observations(self):
        for module in MODULES:
            for average in (-1, "-1.0", None):
                with self.subTest(module=module.__name__, average=average):
                    model = add_nodes(make_builder(module), [indicator(average)])
                    self.assertEqual(observed_values(model), [])

    def test_missing_observations_do_not_consume_random_draws(self):
        for module in MODULES:
            for average in (-1, None):
                with self.subTest(module=module.__name__, average=average):
                    random.seed(1729)
                    expected_state = random.getstate()
                    add_nodes(make_builder(module), [indicator(average)])
                    self.assertEqual(random.getstate(), expected_state)

    def test_raw_observation_converter_skips_missing(self):
        for module in MODULES:
            for value in (-1, "-1", None):
                with self.subTest(module=module.__name__, value=value):
                    processor = module.EvidenceProcessor(module.ModelConfig())
                    self.assertEqual(processor.convert_observations_to_values([value], 0), [])

    def test_valid_numeric_and_string_endpoints(self):
        for module in MODULES:
            for average, expected in ((0, 0), ("0.0", 0), (1, 1), ("1.0", 1)):
                with self.subTest(module=module.__name__, average=average):
                    random.seed(1729)
                    model = add_nodes(make_builder(module), [indicator(average)])
                    self.assertEqual(observed_values(model), [expected])

    def test_valid_probability_retains_the_selected_random_draw(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                random.seed(1729)
                model = add_nodes(make_builder(module), [indicator("0.5")])
                self.assertEqual(observed_values(model), [0])

    def test_pooled_observations_keep_count_and_draw_order(self):
        random.seed(1729)
        model = add_nodes(make_builder(sensitivity_test_model, pooled=True), [indicator("0.5")])
        self.assertEqual(observed_values(model), [0, 0, 1, 1, 0])

    def test_pooled_missing_observations_are_not_zero_measurements(self):
        for average in (-1, None):
            with self.subTest(average=average):
                model = add_nodes(make_builder(sensitivity_test_model, pooled=True), [indicator(average)])
                self.assertEqual(observed_values(model), [])

    def test_invalid_averages_raise_contextual_errors(self):
        for module in MODULES:
            for average in (6.2333333333333333, -0.1, math.nan, math.inf, -math.inf, "NaN", "Infinity", "not a probability"):
                with self.subTest(module=module.__name__, average=repr(average)):
                    with self.assertRaisesRegex(ValueError, "test_indicator.*Test system"):
                        add_nodes(make_builder(module), [indicator(average)])

    def test_raw_converter_rejects_invalid_probabilities(self):
        for module in MODULES:
            for value in (6.2333333333333333, -0.1, math.nan, math.inf, -math.inf, "bad"):
                with self.subTest(module=module.__name__, value=repr(value)):
                    processor = module.EvidenceProcessor(module.ModelConfig())
                    with self.assertRaises(ValueError):
                        processor.convert_observations_to_values([value], 0)

    def test_invalid_nested_input_never_reaches_sampler(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                bad = indicator("6.2333333333333333")
                spec = {
                    "name": "Example stance",
                    "type": "Stance",
                    "evidencers": [{"name": "Feature", "type": "Feature", "evidencers": [bad]}],
                }
                builder = make_builder(module)
                # Real graph construction, explicit sampler-boundary double.
                # A failure must propagate through every existing recursive catch.
                with patch.object(module.pm, "sample", side_effect=AssertionError("Invalid data reached MCMC")) as sampler:
                    with self.assertRaisesRegex(ValueError, "Test system"):
                        builder.build_model(copy.deepcopy(spec))
                    sampler.assert_not_called()

    def test_other_systems_invalid_data_is_not_used(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                node = indicator(1)
                node["observation_stats"]["Other system"] = {"average": "invalid"}
                random.seed(1729)
                self.assertEqual(observed_values(add_nodes(make_builder(module), [node])), [1])

    def test_absent_selected_system_keeps_no_observed_evidence(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                node = indicator("invalid")
                node["observations"] = {}
                # Preserve the existing indicator-with-observations branch:
                # there are observations, but only for an unselected system.
                node["observations"]["Other system"] = {"names": ["Example"], "values": [1]}
                self.assertEqual(observed_values(add_nodes(make_builder(module), [node])), [])

    def test_unrelated_evidencer_errors_keep_existing_skip_behavior(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                with self.assertLogs("BayesianModelBuilder", level="WARNING"):
                    model = add_nodes(make_builder(module), [{"name": "Broken node"}, indicator(1)])
                self.assertEqual(observed_values(model), [1])

    def test_zero_probability_at_zero_random_draw(self):
        for module in MODULES:
            for entry_point in ("raw_converter", "graph"):
                with self.subTest(module=module.__name__, entry_point=entry_point):
                    # Zero belongs to random.random()'s supported [0, 1) range.
                    with patch.object(module.random, "random", return_value=0.0):
                        if entry_point == "raw_converter":
                            processor = module.EvidenceProcessor(module.ModelConfig())
                            values = processor.convert_observations_to_values([0], 0)
                        else:
                            values = observed_values(add_nodes(make_builder(module), [indicator(0)]))
                    self.assertEqual(values, [0])


if __name__ == "__main__":
    unittest.main()
