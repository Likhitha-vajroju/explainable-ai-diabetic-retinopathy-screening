"""Discrete-event resource-allocation simulation for rural DR telemedicine."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REVIEW_OUTCOMES = (
    "ai_prediction_accepted",
    "ai_prediction_modified",
    "image_requires_manual_review",
    "image_ungradable",
)


@dataclass(frozen=True)
class SimulationConfig:
    annual_patient_volume: int = 100_000
    daily_arrival_rate: float | None = None
    simulation_days: int = 365
    operating_hours_per_day: float = 8.0
    image_acquisition_rate_per_hour: float = 30.0
    image_quality_assessment_time_seconds: float = 0.0063
    ai_processing_time_seconds: float = 2.5
    image_transmission_size_mb: float = 5.0
    available_bandwidth_mbps: float = 10.0
    number_of_ophthalmologists: int = 5
    review_time_per_case_minutes: float = 15.0
    processing_capacity: int = 4
    referable_fraction: float = 0.2
    ungradable_fraction: float = 0.05
    reviewer_outcome_probabilities: dict[str, float] | None = None
    random_seed: int = 2026

    def validate(self) -> None:
        positive = {
            "annual_patient_volume": self.annual_patient_volume,
            "simulation_days": self.simulation_days,
            "operating_hours_per_day": self.operating_hours_per_day,
            "image_acquisition_rate_per_hour": self.image_acquisition_rate_per_hour,
            "image_quality_assessment_time_seconds": self.image_quality_assessment_time_seconds,
            "ai_processing_time_seconds": self.ai_processing_time_seconds,
            "image_transmission_size_mb": self.image_transmission_size_mb,
            "available_bandwidth_mbps": self.available_bandwidth_mbps,
            "number_of_ophthalmologists": self.number_of_ophthalmologists,
            "review_time_per_case_minutes": self.review_time_per_case_minutes,
            "processing_capacity": self.processing_capacity,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be greater than zero.")
        if self.operating_hours_per_day > 24:
            raise ValueError("operating_hours_per_day cannot exceed 24.")
        if self.daily_arrival_rate is not None and (
            not math.isfinite(self.daily_arrival_rate) or self.daily_arrival_rate <= 0
        ):
            raise ValueError("daily_arrival_rate must be greater than zero when set.")
        for name, fraction in (
            ("referable_fraction", self.referable_fraction),
            ("ungradable_fraction", self.ungradable_fraction),
        ):
            if not math.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.referable_fraction + self.ungradable_fraction > 1:
            raise ValueError("referable_fraction + ungradable_fraction cannot exceed 1.")
        if self.reviewer_outcome_probabilities is not None:
            if set(self.reviewer_outcome_probabilities) != set(REVIEW_OUTCOMES):
                raise ValueError(
                    "reviewer_outcome_probabilities must define all four reviewer outcomes."
                )
            if any(
                not math.isfinite(value) or value < 0
                for value in self.reviewer_outcome_probabilities.values()
            ):
                raise ValueError("Reviewer outcome probabilities cannot be negative.")
            if not math.isclose(
                sum(self.reviewer_outcome_probabilities.values()), 1.0, abs_tol=1e-9
            ):
                raise ValueError("Reviewer outcome probabilities must sum to 1.")


@dataclass
class Patient:
    arrival_time: float


@dataclass
class StageStats:
    servers: int
    service_seconds: float
    busy: int = 0
    queue: deque[tuple[Patient, float]] = field(default_factory=deque)
    waits: list[float] = field(default_factory=list)
    max_queue: int = 0
    service_count: int = 0
    total_service_seconds: float = 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1)]


def run_simulation(config: SimulationConfig) -> dict[str, Any]:
    """Run one stochastic year (or configured horizon) and drain its queues."""
    config.validate()
    arrival_rng = random.Random(config.random_seed)
    outcome_rng = random.Random(config.random_seed + 1)
    day_seconds = 24 * 60 * 60
    horizon_seconds = config.simulation_days * day_seconds
    operating_seconds = config.operating_hours_per_day * 3600
    total_active_seconds = config.simulation_days * operating_seconds
    daily_arrival_rate = (
        config.daily_arrival_rate
        if config.daily_arrival_rate is not None
        else config.annual_patient_volume / config.simulation_days
    )
    transmission_seconds = (
        config.image_transmission_size_mb * 8_000_000
        / (config.available_bandwidth_mbps * 1_000_000)
    )
    stations = {
        "acquisition": StageStats(
            servers=1,
            service_seconds=3600 / config.image_acquisition_rate_per_hour,
        ),
        "image_quality": StageStats(
            servers=1,
            service_seconds=config.image_quality_assessment_time_seconds,
        ),
        "ai_processing": StageStats(
            servers=config.processing_capacity,
            service_seconds=config.ai_processing_time_seconds,
        ),
        "transmission": StageStats(servers=1, service_seconds=transmission_seconds),
        "review": StageStats(
            servers=config.number_of_ophthalmologists,
            service_seconds=config.review_time_per_case_minutes * 60,
        ),
    }
    events: list[tuple[float, int, str, Any]] = []
    sequence = 0

    def schedule(time: float, event_type: str, payload: Any) -> None:
        nonlocal sequence
        sequence += 1
        heapq.heappush(events, (time, sequence, event_type, payload))

    def enter_stage(stage_name: str, patient: Patient, time: float) -> None:
        stage = stations[stage_name]
        if stage.busy < stage.servers:
            stage.busy += 1
            wait = max(0.0, time - (
                patient.arrival_time if stage_name == "acquisition" else time
            ))
            stage.waits.append(wait)
            stage.service_count += 1
            stage.total_service_seconds += stage.service_seconds
            schedule(
                next_completion_time(time, stage.service_seconds),
                "complete",
                (stage_name, patient),
            )
        else:
            stage.queue.append((patient, time))
            stage.max_queue = max(stage.max_queue, len(stage.queue))

    def finish_stage(stage_name: str, time: float) -> None:
        stage = stations[stage_name]
        if stage.queue:
            patient, queued_at = stage.queue.popleft()
            wait = time - queued_at
            stage.waits.append(wait)
            stage.service_count += 1
            stage.total_service_seconds += stage.service_seconds
            schedule(
                next_completion_time(time, stage.service_seconds),
                "complete",
                (stage_name, patient),
            )
        else:
            stage.busy -= 1

    arrivals = 0
    generated_ungradable = 0
    referable_cases = 0
    completed_screenings = 0
    completed_reviews = 0
    final_outcomes = {name: 0 for name in REVIEW_OUTCOMES}
    end_to_end_seconds: list[float] = []
    completed_by_horizon = 0
    last_event_time = 0.0
    outcomes = config.reviewer_outcome_probabilities or {
        "ai_prediction_accepted": 0.75,
        "ai_prediction_modified": 0.15,
        "image_requires_manual_review": 0.07,
        "image_ungradable": 0.03,
    }

    def record_outcome(patient: Patient, time: float, outcome: str) -> None:
        nonlocal completed_by_horizon
        final_outcomes[outcome] += 1
        end_to_end_seconds.append(time - patient.arrival_time)
        if time <= horizon_seconds:
            completed_by_horizon += 1

    def to_wall_time(
        active_seconds: float,
        *,
        move_opening_arrivals_to_next_shift: bool = True,
    ) -> float:
        day_index = math.floor(active_seconds / operating_seconds)
        within_shift = active_seconds - day_index * operating_seconds
        if (
            move_opening_arrivals_to_next_shift
            and active_seconds > 0
            and within_shift <= 1e-9
        ):
            return day_index * day_seconds
        return day_index * day_seconds + within_shift

    def to_active_time(wall_time: float) -> float:
        day_index = math.floor(wall_time / day_seconds)
        within_day = wall_time - day_index * day_seconds
        return day_index * operating_seconds + min(within_day, operating_seconds)

    def next_completion_time(start_time: float, service_seconds: float) -> float:
        active_end = to_active_time(start_time) + service_seconds
        return to_wall_time(
            active_end,
            move_opening_arrivals_to_next_shift=False,
        )

    def generate_arrival(time: float) -> None:
        nonlocal arrivals
        active_time = to_active_time(time)
        if active_time >= total_active_seconds:
            return
        patient = Patient(arrival_time=time)
        arrivals += 1
        enter_stage("acquisition", patient, time)
        next_active_time = active_time + arrival_rng.expovariate(
            daily_arrival_rate / operating_seconds
        )
        if next_active_time < total_active_seconds:
            schedule(to_wall_time(next_active_time), "arrival", None)

    first_active_time = arrival_rng.expovariate(
        daily_arrival_rate / operating_seconds
    )
    if first_active_time < total_active_seconds:
        schedule(to_wall_time(first_active_time), "arrival", None)
    while events:
        time, _, event_type, payload = heapq.heappop(events)
        last_event_time = max(last_event_time, time)
        if event_type == "arrival":
            generate_arrival(time)
            continue

        stage_name, patient = payload
        finish_stage(stage_name, time)
        if stage_name == "acquisition":
            enter_stage("image_quality", patient, time)
        elif stage_name == "image_quality":
            if outcome_rng.random() < config.ungradable_fraction:
                generated_ungradable += 1
                record_outcome(patient, time, "image_ungradable")
            else:
                enter_stage("ai_processing", patient, time)
        elif stage_name == "ai_processing":
            if outcome_rng.random() < config.referable_fraction:
                referable_cases += 1
                enter_stage("transmission", patient, time)
            else:
                completed_screenings += 1
                end_to_end_seconds.append(time - patient.arrival_time)
                if time <= horizon_seconds:
                    completed_by_horizon += 1
        elif stage_name == "transmission":
            enter_stage("review", patient, time)
        elif stage_name == "review":
            completed_reviews += 1
            outcome = outcome_rng.choices(
                list(outcomes),
                weights=list(outcomes.values()),
                k=1,
            )[0]
            record_outcome(patient, time, outcome)

    makespan_seconds = max(horizon_seconds, last_event_time)
    total_completed = completed_screenings + completed_reviews + generated_ungradable
    stage_results: dict[str, dict[str, float | int]] = {}
    for name, stage in stations.items():
        waits = stage.waits
        available_resource_seconds = (
            stage.servers
            * operating_seconds
            * max(1, math.ceil(makespan_seconds / day_seconds))
        )
        stage_results[name] = {
            "servers": stage.servers,
            "cases_processed": stage.service_count,
            "mean_queue_wait_minutes": (
                sum(waits) / len(waits) / 60 if waits else 0.0
            ),
            "p95_queue_wait_minutes": _percentile(waits, 0.95) / 60,
            "maximum_queue_length": stage.max_queue,
            "utilization": min(
                1.0,
                stage.total_service_seconds
                / available_resource_seconds,
            ),
            "service_time_seconds_per_case": stage.service_seconds,
        }

    return {
        "simulation_type": "synthetic resource-allocation/deployment simulation",
        "random_seed": config.random_seed,
        "parameters": {
            **asdict(config),
            "effective_daily_arrival_rate": daily_arrival_rate,
            "transmission_time_seconds_per_image": transmission_seconds,
        },
        "results": {
            "simulation_days": config.simulation_days,
            "planning_annual_patient_volume": config.annual_patient_volume,
            "generated_arrivals": arrivals,
            "arrival_rate_patients_per_day": daily_arrival_rate,
            "completed_screening_results": completed_screenings,
            "referable_cases_routed": referable_cases,
            "ophthalmologist_reviews_completed": completed_reviews,
            "ungradable_images": generated_ungradable,
            "total_cases_with_final_outcome": total_completed,
            "pending_cases_at_horizon": max(0, arrivals - completed_by_horizon),
            "completed_within_simulation_horizon": completed_by_horizon,
            "throughput_per_simulated_year": (
                completed_by_horizon * 365 / config.simulation_days
            ),
            "mean_end_to_end_hours": (
                sum(end_to_end_seconds) / len(end_to_end_seconds) / 3600
                if end_to_end_seconds else 0.0
            ),
            "p95_end_to_end_hours": _percentile(end_to_end_seconds, 0.95) / 3600,
            "final_review_outcomes": final_outcomes,
            "stage_resources": stage_results,
            "simulation_makespan_days_including_queue_drain": (
                makespan_seconds / day_seconds
            ),
        },
        "interpretation": (
            "Simulated capacity and waiting-time estimates under the supplied "
            "assumptions; not evidence of real-world clinical performance or "
            "proof that a service can serve the planning volume."
        ),
    }


def load_config(path: Path) -> SimulationConfig:
    values = json.loads(path.read_text(encoding="utf-8"))
    return SimulationConfig(**values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("scenario_100k.json"),
        help="JSON file containing SimulationConfig parameters.",
    )
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    args = parser.parse_args()
    report = run_simulation(load_config(args.config))
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
