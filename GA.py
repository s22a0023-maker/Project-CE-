# ==============================
# SECTION 1: CASE STUDY SELECTION
# ==============================

"""
Traffic Light Optimization Problem

Goal:
Optimize traffic signal green times to reduce traffic congestion.

Decision Variables:
- Green time for each traffic phase (seconds)

Constraints:
- Minimum green time
- Maximum green time

Objectives:
- Minimize vehicle waiting time
- Minimize queue length
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt


# ==============================
# SECTION 2: DATASET LOADING
# ==============================

@st.cache_data
def load_data():
    return pd.read_csv("traffic_dataset.csv")

data = load_data()

# Required dataset columns
flow_rate = data["flow_rate"].values
waiting_time = data["waiting_time"].values
vehicle_count = data["vehicle_count"].values

st.title("🚦 Traffic Light Optimization using Genetic Algorithm")
st.subheader("Dataset Preview")
st.dataframe(data.head())


# ==============================
# SECTION 3.1: GA PARAMETERS
# ==============================

st.sidebar.header("GA Parameters")

population_size = st.sidebar.slider("Population Size", 20, 200, 50)
generations = st.sidebar.slider("Number of Generations", 50, 500, 200)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.5, 1.0, 0.8)

min_green = st.sidebar.slider("Min Green Time (sec)", 5, 20, 10)
max_green = st.sidebar.slider("Max Green Time (sec)", 30, 90, 60)

num_phases = 4  # Four-way intersection

# ==============================
# SECTION 3.2: FITNESS FUNCTION
# ==============================

def fitness_function(chromosome):
    """
    Multi-objective fitness (weighted sum):
    - Waiting time minimization
    - Queue length minimization
    """
    total_wait = 0
    total_queue = 0

    for i in range(len(flow_rate)):
        green_effect = sum(chromosome) / (flow_rate[i] + 1)

        wait = max(0, waiting_time[i] - green_effect)
        queue = max(0, vehicle_count[i] - green_effect)

        total_wait += wait
        total_queue += queue

    # Weighted multi-objective fitness
    fitness = 0.7 * total_wait + 0.3 * total_queue
    return fitness

# ==============================
# SECTION 3.3: GA OPERATORS
# ==============================

# ==============================
# OPTIMIZED GA OPERATORS
# ==============================

def create_individual():
    return np.random.randint(min_green, max_green + 1, size=num_phases)

def create_population():
    return [create_individual() for _ in range(population_size)]

def selection(population, fitness_values):
    idx = np.random.choice(len(population), 3, replace=False)
    best_idx = idx[np.argmin([fitness_values[i] for i in idx])]
    return population[best_idx].copy()

def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        point = random.randint(1, num_phases - 1)
        child = np.concatenate((parent1[:point], parent2[point:]))
        return child
    return parent1.copy()

def mutation(individual):
    mutation_mask = np.random.rand(num_phases) < mutation_rate
    individual[mutation_mask] = np.random.randint(
        min_green, max_green + 1, np.sum(mutation_mask)
    )
    return individual

# ==============================
# SECTION 3.4: GENETIC ALGORITHM
# ==============================

# ==============================
# FAST GENETIC ALGORITHM
# ==============================

def genetic_algorithm():
    population = create_population()

    best_fitness_history = []
    avg_fitness_history = []

    start_time = time.time()
    stagnation_counter = 0
    best_overall = float("inf")

    for gen in range(generations):
        # Calculate fitness ONCE per generation
        fitness_values = np.array([fitness_function(ind) for ind in population])

        best_idx = np.argmin(fitness_values)
        best_fitness = fitness_values[best_idx]

        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(np.mean(fitness_values))

        # Early stopping
        if best_fitness < best_overall:
            best_overall = best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= 30:
            st.info(f"Early stopping at generation {gen}")
            break

        # Elitism: keep best individual
        new_population = [population[best_idx].copy()]

        while len(new_population) < population_size:
            p1 = selection(population, fitness_values)
            p2 = selection(population, fitness_values)
            child = crossover(p1, p2)
            child = mutation(child)
            new_population.append(child)

        population = new_population

    end_time = time.time()

    best_solution = population[0]
    return best_solution, best_fitness_history, avg_fitness_history, end_time - start_time

# ==============================
# SECTION 4: PERFORMANCE ANALYSIS
# ==============================

if st.button("▶ Run Genetic Algorithm"):
    best_solution, best_curve, avg_curve, exec_time = genetic_algorithm()

    st.subheader("Best Traffic Signal Timings (seconds)")
    st.write(best_solution)

    st.subheader("Convergence Analysis")
    fig, ax = plt.subplots()
    ax.plot(best_curve, label="Best Fitness")
    ax.plot(avg_curve, label="Average Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Value")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Performance Metrics")
    st.write(f"Final Best Fitness: {best_curve[-1]:.2f}")
    st.write(f"Execution Time: {exec_time:.2f} seconds")
    st.write(f"Convergence Generation: {np.argmin(best_curve)}")


# ==============================
# SECTION 5: EXTENDED ANALYSIS
# ==============================

st.subheader("Multi-Objective Optimization Analysis")

st.markdown("""
**Objectives Considered:**
- Minimize average vehicle waiting time
- Minimize queue length

**Approach Used:**
- Weighted-sum multi-objective Genetic Algorithm
- Waiting time weight = 0.7
- Queue length weight = 0.3

**Impact:**
- Improves traffic flow efficiency
- Balances congestion reduction and fairness across lanes
""")


# ==============================
# SECTION 6: STREAMLIT FEATURES
# ==============================

st.markdown("""
### Streamlit Dashboard Capabilities
- Interactive GA parameter tuning
- Dataset visualization
- Real-time convergence curves
- Performance metric reporting
- Multi-objective trade-off explanation
""")

