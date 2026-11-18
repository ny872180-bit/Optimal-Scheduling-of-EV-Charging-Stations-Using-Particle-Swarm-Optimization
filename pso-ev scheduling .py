
import numpy as np

# -----------------------------------------
# 1. EV DATA FROM THE REPORT
# -----------------------------------------

ev_data = {
    "EV1": {
        "arrival": 1,
        "departure": 3,
        "energy": 4,
        "power": 4
    },
    "EV2": {
        "arrival": 3,
        "departure": 4,
        "energy": 5,
        "power": 5
    }
}

# Electricity price for each time slot
prices = np.array([0.20, 0.10, 0.15])

# One hour per slot
slot_duration = 1

# PSO parameters
num_particles = 30
max_iterations = 100

w = 0.7
c1 = 2.0
c2 = 2.0

rng = np.random.default_rng(42)


# -----------------------------------------
# 2. INITIAL CHARGING SCHEDULE
# -----------------------------------------

# Each value represents the charging start slot.
# Initial schedule: EV1 starts in slot 1, EV2 in slot 3.

initial_schedule = np.array([1, 3])


# -----------------------------------------
# 3. DECODE PARTICLE INTO SCHEDULE
# -----------------------------------------

def decode_schedule(position):

    # EV1 can start in slot 1 or slot 2.
    # EV2 can only start in slot 3.

    start_ev1 = int(np.clip(np.rint(position[0]), 1, 2))
    start_ev2 = 3

    return np.array([start_ev1, start_ev2])


# -----------------------------------------
# 4. CALCULATE SCHEDULE PERFORMANCE
# -----------------------------------------

def calculate_metrics(schedule):

    total_cost = 0
    slot_power = np.zeros(3)
    waiting_times = []

    for i, ev_name in enumerate(["EV1", "EV2"]):

        ev = ev_data[ev_name]

        start = schedule[i]
        arrival = ev["arrival"]
        departure = ev["departure"]
        power = ev["power"]
        energy = ev["energy"]

        # Check arrival and departure constraints
        if start < arrival or start >= departure:
            return None

        # Check whether required energy is delivered
        delivered_energy = power * slot_duration

        if delivered_energy < energy:
            return None

        # Convert 1-based slot to array index
        slot_index = start - 1

        # Add charging power to the selected slot
        slot_power[slot_index] += power

        # Calculate electricity cost
        total_cost += (
            power * slot_duration * prices[slot_index]
        )

        # Calculate waiting time
        waiting_times.append(start - arrival)

    peak_power = np.max(slot_power)
    average_waiting = np.mean(waiting_times)

    return {
        "cost": total_cost,
        "peak": peak_power,
        "waiting": average_waiting,
        "slot_power": slot_power
    }


# -----------------------------------------
# 5. FITNESS FUNCTION
# -----------------------------------------

def fitness(position):

    schedule = decode_schedule(position)
    metrics = calculate_metrics(schedule)

    if metrics is None:
        return 100000

    # Cost-minimization objective.
    # This setting reproduces the illustrative
    # cost comparison in the report.

    return metrics["cost"]


# -----------------------------------------
# 6. INITIALIZE PARTICLES
# -----------------------------------------

# Each particle has two position values:
# [EV1 starting slot, EV2 starting slot]

positions = np.zeros((num_particles, 2))
velocities = np.zeros((num_particles, 2))

# EV1: random position between slots 1 and 2
positions[:, 0] = rng.uniform(1, 2, num_particles)

# EV2: fixed starting slot 3
positions[:, 1] = 3

pbest_positions = positions.copy()

pbest_scores = np.array([
    fitness(p) for p in positions
])

best_index = np.argmin(pbest_scores)

gbest_position = pbest_positions[best_index].copy()
gbest_score = pbest_scores[best_index]


# -----------------------------------------
# 7. PSO OPTIMIZATION
# -----------------------------------------

print("Starting PSO Optimization...\n")

for iteration in range(max_iterations):

    for i in range(num_particles):

        r1 = rng.random()
        r2 = rng.random()

        # Velocity update
        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest_positions[i] - positions[i])
            + c2 * r2 * (gbest_position - positions[i])
        )

        # Position update
        positions[i] += velocities[i]

        # Keep positions within valid limits
        positions[i, 0] = np.clip(positions[i, 0], 1, 2)
        positions[i, 1] = 3

        # Evaluate new position
        current_score = fitness(positions[i])

        # Update personal best
        if current_score < pbest_scores[i]:

            pbest_scores[i] = current_score
            pbest_positions[i] = positions[i].copy()

        # Update global best
        if current_score < gbest_score:

            gbest_score = current_score
            gbest_position = positions[i].copy()

    if iteration % 10 == 0:
        print(
            f"Iteration {iteration + 1}: "
            f"Best Cost = {gbest_score:.2f}"
        )


# -----------------------------------------
# 8. FINAL OPTIMIZED SCHEDULE
# -----------------------------------------

optimized_schedule = decode_schedule(gbest_position)

initial_metrics = calculate_metrics(initial_schedule)
optimized_metrics = calculate_metrics(optimized_schedule)


# -----------------------------------------
# 9. DISPLAY RESULTS
# -----------------------------------------

def display_schedule(title, schedule, metrics):

    print("\n" + title)
    print("-" * 45)

    print("EV       Slot 1    Slot 2    Slot 3")

    for i, ev_name in enumerate(["EV1", "EV2"]):

        power = ev_data[ev_name]["power"]
        row = [0, 0, 0]

        row[schedule[i] - 1] = power

        print(
            f"{ev_name}      "
            f"{row[0]} kW     "
            f"{row[1]} kW     "
            f"{row[2]} kW"
        )

    print(f"\nTotal Cost: {metrics['cost']:.2f}")
    print(f"Peak Power: {metrics['peak']:.2f} kW")
    print(f"Average Waiting Time: {metrics['waiting']:.2f} hours")


display_schedule(
    "INITIAL CHARGING SCHEDULE",
    initial_schedule,
    initial_metrics
)

display_schedule(
    "PSO OPTIMIZED CHARGING SCHEDULE",
    optimized_schedule,
    optimized_metrics
)


# -----------------------------------------
# 10. COST REDUCTION
# -----------------------------------------

cost_reduction = (
    (initial_metrics["cost"] - optimized_metrics["cost"])
    / initial_metrics["cost"]
) * 100

print("\nFINAL PERFORMANCE")
print("-" * 45)

print(f"Initial Cost: {initial_metrics['cost']:.2f}")
print(f"Optimized Cost: {optimized_metrics['cost']:.2f}")

print(f"Cost Reduction: {cost_reduction:.2f}%")

print("\nPSO Optimization Completed!")
