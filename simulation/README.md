# Rural telemedicine resource-allocation simulation

This module explores **resource allocation and deployment scenarios** for a
hypothetical rural primary health centre (PHC) connected to district-level
ophthalmologist review:

```text
Rural PHC
  -> fundus image acquisition
  -> image-quality assessment
  -> AI screening
  -> referable-case routing
  -> network transmission
  -> ophthalmologist review
  -> final review outcome
```

It is a planning simulation only. The 100,000-patient scenario is a demand
target for experiments, **not a claim that this prototype or any real service
can serve 100,000 or more patients annually**. Synthetic inputs and assumed
referral/outcome proportions are not clinical results.

## Run the discrete-event simulation

The Python model uses only the standard library and produces a JSON report:

```powershell
.\.venv\Scripts\python.exe simulation\telemedicine_simulation.py `
  --config simulation\scenario_100k.json `
  --output $env:TEMP\telemedicine-100k.json
```

Try alternative resource configurations with
`scenario_resource_stress.json` and `scenario_resource_expansion.json`. Edit or
copy a JSON configuration to test other values; use a different `random_seed`
to explore stochastic variation. If `daily_arrival_rate` is `null`, it is
derived from `annual_patient_volume / simulation_days`. If specified, the daily
rate drives arrivals and the annual volume remains a planning target for
comparison.

Arrivals follow a Poisson process during PHC operating hours, on every
simulated day (the model has no holidays or site-closure calendar). Each case joins a
first-in/first-out queue at each resource stage. Staffed stages process work
only during the configured daily operating window; unfinished work waits for
the next day's opening. Image quality failures are recorded as ungradable
outcomes and bypass AI. AI-screened cases are split into assumed referable and
non-referable proportions. Non-referable results complete as screening
outcomes; referable results queue for image transmission and ophthalmologist
review. Review outcome proportions are configurable. The simulation continues
processing after the arrival horizon to report queue-drain time and cases still
pending at the horizon.

## Build the Simulink model

With MATLAB, Simulink, and MATLAB Function block support installed, run:

```matlab
run('simulation/build_simulink_model.m')
sim('rural_dr_telemedicine_resource_allocation')
```

The builder creates
`simulation/rural_dr_telemedicine_resource_allocation.slx`. It reads
`scenario_100k.json` by default. To point it at another configuration, set
`DR_SIMULATION_SCENARIO` to that JSON path before running the builder. The
Simulink model is a deterministic, daily-step **fluid-queue approximation**.
The Python discrete-event model is the more detailed implementation: it
represents individual stochastic arrivals, FIFO resource queues, service
times, and time-of-day operating windows. The two have intentionally different
detail and should not be expected to produce identical results.

Simulink saves time series to the MATLAB workspace:

- `cumulativeArrivals`
- `cumulativeFinalOutcomes`
- `cumulativeReferableCases`
- `cumulativeOphthalmologistReviews`
- `totalQueuedCases`
- `estimatedQueueWaitHours`

## Parameters

Each JSON scenario accepts:

| Parameter | Meaning |
| --- | --- |
| `annual_patient_volume` | Planning target; when no daily rate is given, determines the mean arrival rate over the simulated days. |
| `daily_arrival_rate` | Optional mean patients per operating day; when set, this drives arrivals instead of the annual target. |
| `simulation_days` | Arrival horizon and annualization basis. |
| `operating_hours_per_day` | Daily staffed service window, in hours (maximum 24). |
| `image_acquisition_rate_per_hour` | Fundus image acquisitions per staffed hour at the PHC. |
| `image_quality_assessment_time_seconds` | Quality-check service time per image; the included scenarios use the qualified local function timing below. |
| `ai_processing_time_seconds` | Per-image preprocessing plus screening service time. The included value is an explicit placeholder until inference can be benchmarked. |
| `image_transmission_size_mb` | Assumed decimal megabytes transmitted per case. |
| `available_bandwidth_mbps` | Effective decimal megabits per second for the modeled link. |
| `number_of_ophthalmologists` | Parallel reviewer capacity at district level. |
| `review_time_per_case_minutes` | Assumed ophthalmologist time for each routed case. |
| `processing_capacity` | Number of parallel AI processing workers. |
| `referable_fraction` | Assumed fraction of gradable AI-screened cases routed to review. Not model-measured or clinically estimated. |
| `ungradable_fraction` | Assumed fraction of cases rejected at image-quality assessment. |
| `reviewer_outcome_probabilities` | Assumed split of reviewer outcomes; probabilities for all four decisions must sum to 1. |
| `random_seed` | Seed for reproducible arrivals and routing. |

## Outputs

The Python JSON report includes generated arrivals, outcomes completed during
the arrival horizon, estimated annualized throughput, cases pending at the
horizon, cases ultimately completed after queues drain, end-to-end time,
queue-drain makespan, and per-stage resource count, utilization, mean and 95th
percentile queue wait, maximum queue length, processed cases, and assumed
service time. Utilization and annualized figures are conditional model outputs,
not observed service performance.

## Prototype timing measurements

`benchmark_prototype.py` measures image-quality assessment and image
preprocessing against a local image. If a trained model checkpoint is
available, it also measures preprocessing plus prediction:

```powershell
.\.venv\Scripts\python.exe simulation\benchmark_prototype.py `
  --image path\to\authorized-test-image.jpg `
  --checkpoint models\checkpoints\dr_resnet18\best_model.pt `
  --output $env:TEMP\prototype-timings.json
```

Use the measured median prediction time to replace
`ai_processing_time_seconds` in a scenario, noting that prediction timing
includes preprocessing. If no checkpoint is available, the script reports
prediction latency as not measured rather than inventing a value. These
single-machine timings exclude Grad-CAM, retinal evidence generation, network
transfer, clinical review, and operational overhead. Do not treat them as a
deployment service-level measurement.

### Connecting measurements to scenario inputs

The benchmark and the deployment simulators are separate tools; there is no
live connection to the screening API. To carry measured prototype
characteristics into an experiment:

1. Run the benchmark with an authorized image representative of the target
   input format/resolution and, when available, the same checkpoint and device
   intended for inference.
2. Copy the measured quality-assessment median into
   `image_quality_assessment_time_seconds`.
3. If prediction status is `measured`, copy its median into
   `ai_processing_time_seconds`. This timing includes preprocessing plus
   prediction. Do not add the separate preprocessing time again.
4. Keep `image_transmission_size_mb`, `available_bandwidth_mbps`, staffing,
   service hours, referral fractions, and reviewer time tied to their own
   observed sources or label them as assumptions. The benchmark does not
   measure these values.
5. Save the resulting scenario JSON and record the benchmark JSON, image
   properties, hardware/device, checkpoint identity, warmups, and repeats
   alongside the simulation output.

The simulator treats quality assessment and AI screening as sequential
service stages, while the AI service time already includes preprocessing when
derived from the benchmark's prediction timing. The separate preprocessing
measurement is useful for component profiling but should not be added to that
end-to-end AI service time. Neither simulator currently ingests real API
telemetry or actual reviewer decisions.

At implementation time, the actual image-quality assessment and preprocessing
functions were timed locally using a generated, non-retinal 256 x 256 test
image (one warm-up and three measured runs): image quality assessment median
was approximately **0.0063 seconds/image**, and preprocessing median was
approximately **0.0421 seconds/image**. The scenarios use the measured,
rounded image-quality time. These timings are specific to this test image and
development environment and are not representative clinical-image or
deployment benchmarks. No trained checkpoint is present in the workspace, so
AI prediction time could not be measured; the configured 2.5 seconds/image is
an illustrative placeholder, not a measured result. Replace it with the
benchmark output for the intended model, image size, device, and deployment
hardware before interpreting capacity.

## Assumptions and limitations

- Apart from the explicitly qualified local image-quality timing above,
  scenario values are **illustrative assumptions**, not measurements from a
  healthcare site. The 100,000 annual volume is a demand scenario only.
- The included local workspace has no trained model checkpoint and no MATLAB
  or Simulink executable. Therefore its AI latency cannot currently be
  measured here and a generated `.slx` cannot be executed in this environment.
  Run the benchmark with an appropriately available local checkpoint, and run
  the builder in MATLAB/Simulink to generate and validate the Simulink model.
- The simulation has a single PHC acquisition resource and one shared network
  transmission server; it does not model multiple centres, intermittent or
  asymmetric connectivity, retries, compression, data loss, power cuts,
  transport, consent, staff shifts, no-shows, equipment failure, image
  recapture, escalation after a reviewer disagreement, or costs.
- Quality pass rates, referral fractions, reviewer-outcome proportions, and
  service rates are configurable assumptions. The model does not estimate
  clinical sensitivity, specificity, diagnostic accuracy, patient outcomes, or
  disease progression.
- Independent Poisson arrivals and fixed average service times simplify real
  patient flow and workload variability. Results are sensitive to scenario
  assumptions and random seed; compare multiple seeds and calibrated inputs
  before using the model for planning.
- The Simulink model uses deterministic daily fluid flows and aggregate queues;
  its wait estimate is a workload/capacity approximation, not a discrete-event
  patient waiting-time distribution.
- Neither implementation constitutes a validated operational, clinical,
  regulatory, privacy, or safety assessment.
