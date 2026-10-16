from __future__ import annotations

import unittest

from simulation.telemedicine_simulation import SimulationConfig, run_simulation


class TelemedicineSimulationTests(unittest.TestCase):
    def test_simulation_routes_gradable_referable_cases_and_drains_queues(self) -> None:
        result = run_simulation(
            SimulationConfig(
                annual_patient_volume=300,
                simulation_days=30,
                operating_hours_per_day=8,
                image_acquisition_rate_per_hour=40,
                image_quality_assessment_time_seconds=5,
                ai_processing_time_seconds=2,
                image_transmission_size_mb=2,
                available_bandwidth_mbps=20,
                number_of_ophthalmologists=4,
                review_time_per_case_minutes=8,
                processing_capacity=4,
                random_seed=19,
            )
        )
        results = result["results"]

        self.assertGreater(results["generated_arrivals"], 0)
        self.assertEqual(
            results["total_cases_with_final_outcome"],
            results["generated_arrivals"],
        )
        self.assertEqual(
            sum(results["final_review_outcomes"].values()),
            results["ophthalmologist_reviews_completed"]
            + results["ungradable_images"],
        )
        self.assertEqual(
            results["referable_cases_routed"],
            results["ophthalmologist_reviews_completed"],
        )
        self.assertEqual(
            results["completed_screening_results"]
            + results["referable_cases_routed"]
            + results["ungradable_images"],
            results["generated_arrivals"],
        )
        self.assertGreaterEqual(results["pending_cases_at_horizon"], 0)
        self.assertIn("transmission", results["stage_resources"])

    def test_resource_stress_increases_queue_wait(self) -> None:
        common = {
            "annual_patient_volume": 2_000,
            "simulation_days": 30,
            "random_seed": 41,
            "referable_fraction": 0.4,
            "ungradable_fraction": 0.03,
        }
        roomy_result = run_simulation(
            SimulationConfig(
                **common,
                image_acquisition_rate_per_hour=50,
                number_of_ophthalmologists=8,
                review_time_per_case_minutes=8,
                available_bandwidth_mbps=50,
                image_transmission_size_mb=2,
                processing_capacity=8,
            )
        )["results"]
        constrained_review_result = run_simulation(
            SimulationConfig(
                **common,
                image_acquisition_rate_per_hour=50,
                number_of_ophthalmologists=1,
                review_time_per_case_minutes=30,
                available_bandwidth_mbps=50,
                image_transmission_size_mb=2,
                processing_capacity=1,
            )
        )["results"]
        constrained_network_result = run_simulation(
            SimulationConfig(
                **common,
                image_acquisition_rate_per_hour=50,
                number_of_ophthalmologists=8,
                review_time_per_case_minutes=8,
                available_bandwidth_mbps=0.1,
                image_transmission_size_mb=100,
                processing_capacity=1,
            )
        )["results"]
        roomy = roomy_result["stage_resources"]
        constrained_review = constrained_review_result["stage_resources"]
        constrained_network = constrained_network_result["stage_resources"]

        self.assertEqual(
            roomy_result["generated_arrivals"],
            constrained_review_result["generated_arrivals"],
        )
        self.assertEqual(
            roomy_result["generated_arrivals"],
            constrained_network_result["generated_arrivals"],
        )
        self.assertGreater(
            constrained_review["review"]["mean_queue_wait_minutes"],
            roomy["review"]["mean_queue_wait_minutes"],
        )
        self.assertGreater(
            constrained_network["transmission"]["mean_queue_wait_minutes"],
            roomy["transmission"]["mean_queue_wait_minutes"],
        )

    def test_rejects_invalid_parameters_and_review_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "annual_patient_volume"):
            run_simulation(SimulationConfig(annual_patient_volume=0))

        with self.assertRaisesRegex(ValueError, "processing_capacity must be an integer"):
            run_simulation(SimulationConfig(processing_capacity=1.5))

        with self.assertRaisesRegex(ValueError, "sum to 1"):
            run_simulation(
                SimulationConfig(
                    reviewer_outcome_probabilities={
                        "ai_prediction_accepted": 0.5,
                        "ai_prediction_modified": 0.2,
                        "image_requires_manual_review": 0.2,
                        "image_ungradable": 0.2,
                    }
                )
            )


if __name__ == "__main__":
    unittest.main()
