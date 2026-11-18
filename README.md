# EV Charging Scheduler Using Particle Swarm Optimization

A Python example that uses Particle Swarm Optimization (PSO) to find a lower-cost charging schedule for two electric vehicles (EVs), based on their availability and electricity prices.

## Requirements

- Python 3.13 (the version used to test this project)
- NumPy

## Setup and run

Create and activate a virtual environment, then install NumPy:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
```

Run the script from the project folder:

```bash
python "pso-ev scheduling .py"
```

On Windows, create the environment with `py -3.13 -m venv .venv` and activate it with `.venv\Scripts\activate`.

## Example

With the sample data and random seed in the script, the program reports:

| Metric | Initial schedule | Optimized schedule |
|---|---:|---:|
| Total cost | 1.55 | 1.15 |
| Peak power | 5.00 kW | 5.00 kW |
| Average waiting time | 0.00 hours | 0.50 hours |

The example's cost reduction is **25.81%**. The cost values use the electricity prices specified in the script; no currency is assigned.

## How it works

The program initializes a group of candidate schedules ("particles"), evaluates each schedule's electricity cost, and iteratively updates the candidates toward lower-cost schedules. It then displays the initial and optimized schedules and their metrics.

## Current assumptions and limitations

- The sample has two EVs and three one-hour time slots.
- Each EV charges for one slot at its specified power.
- EV1 can start in slot 1 or 2. EV2's charging slot is fixed at slot 3.
- The optimization objective minimizes charging cost. Peak power and waiting time are reported, but are not included in the objective.
- This is an illustrative example, not a general-purpose EV charging optimizer.
