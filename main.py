import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("traffic_dataset.csv")

df = load_data()

# CHANGE COLUMN NAMES HERE IF NEEDED
FLOW_NS = "vehicle_count"
FLOW_EW = "average_speed"

traffic_ns = df[FLOW_NS].mean()
traffic_ew = df[FLOW_EW].mean()

# ---------------------------------------------
# TRAFFIC SIMULATION FUNCTION
# ---------------------------------------------
def simulate_traffic(green_ns, green_ew, yellow):
    """
    Simple delay estimation model
    """
    cycle_time = green_ns + green_ew + yellow

    delay_ns = max(0, traffic_ns * (cycle_time - green_ns))
    delay_ew = max(0, traffic_ew * (cycle_time - green_ew))

    avg_delay = (delay_ns + delay_ew) / 2
    queue_length = (delay_ns + delay_ew) / 10

    return avg_delay, queue_length

# ---------------------------------------------
# GENETIC ALGORITHM FUNCTIONS
# ---------------------------------------------
def initialize_population(pop_size):
    """
    Chromosome: [green_ns, green_ew, yellow]
    """
    population = []
    for _ in range(pop_size):
        green_ns = np.random.uniform(20, 60)
        green_ew = np.random.uniform(20, 60)
        yellow = np.random.uniform(3, 6)
        population.append([green_ns, green_ew, yellow])
    return np.array(population)

def fitness_function(chromosome):
    delay, _ = simulate_traffic(*chromosome)
    return 1 / (delay + 1e-6), delay

def tournament_selection(pop, fitness, k=3):
    idx = np.random.choice(len(pop), k)
    best = idx[np.argmax([fitness[i] for i in idx])]
    return pop[best]

def crossover(p1, p2):
    point = np.random.randint(1, len(p1))
    c1 = np.concatenate((p1[:point], p2[point:]))
    c2 = np.concatenate((p2[:point], p1[point:]))
    return c1, c2

def mutation(chrom, rate):
    for i in range(len(chrom)):
        if np.random.rand() < rate:
            chrom[i] += np.random.normal(0, 2)
            chrom[i] = max(chrom[i], 1)
    return chrom

# ---------------------------------------------
# MULTI-OBJECTIVE FITNESS
# ---------------------------------------------
def multi_objective_fitness(chromosome):
    delay, queue = simulate_traffic(*chromosome)
    return delay, queue

# ---------------------------------------------
# STREAMLIT UI
# ---------------------------------------------
st.title("🚦 Traffic Light Optimization using Genetic Algorithm")

st.sidebar.header("GA Parameters")
population_size = st.sidebar.slider("Population Size", 20, 200, 100)
generations = st.sidebar.slider("Generations", 20, 300, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.3, 0.05)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.5, 1.0, 0.8)
multi_objective = st.sidebar.checkbox("Enable Multi-Objective Optimization")

# ---------------------------------------------
# RUN GA
# ---------------------------------------------
if st.button("Run Optimization"):

    start_time = time.time()

    population = initialize_population(population_size)
    best_delay_history = []
    pareto_data = []

    for gen in range(generations):
        fitness_scores = []
        delay_scores = []

        for chrom in population:
            fit, delay = fitness_function(chrom)
            fitness_scores.append(fit)
            delay_scores.append(delay)

        best_idx = np.argmax(fitness_scores)
        best_delay_history.append(delay_scores[best_idx])

        new_population = []

        while len(new_population) < population_size:
            p1 = tournament_selection(population, fitness_scores)
            p2 = tournament_selection(population, fitness_scores)

            if np.random.rand() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = mutation(c1, mutation_rate)
            c2 = mutation(c2, mutation_rate)

            new_population.extend([c1, c2])

        population = np.array(new_population[:population_size])

        if multi_objective:
            for chrom in population:
                d, q = multi_objective_fitness(chrom)
                pareto_data.append((d, q))

    best_solution = population[best_idx]
    final_delay, final_queue = simulate_traffic(*best_solution)
    runtime = time.time() - start_time

    # ---------------------------------------------
    # RESULTS
    # ---------------------------------------------
    st.subheader("📊 Optimization Results")

    st.write(f"**Optimal Green Time (NS):** {best_solution[0]:.2f} seconds")
    st.write(f"**Optimal Green Time (EW):** {best_solution[1]:.2f} seconds")
    st.write(f"**Yellow Time:** {best_solution[2]:.2f} seconds")
    st.write(f"**Average Vehicle Delay:** {final_delay:.2f}")
    st.write(f"**Average Queue Length:** {final_queue:.2f}")
    st.write(f"**Execution Time:** {runtime:.2f} seconds")

    # Convergence Plot
    st.subheader("📈 Convergence Curve")
    fig1, ax1 = plt.subplots()
    ax1.plot(best_delay_history)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Average Delay")
    ax1.set_title("GA Convergence")
    st.pyplot(fig1)

    # Pareto Front
    if multi_objective:
        st.subheader("⚖ Pareto Front (Delay vs Queue Length)")
        pareto = np.array(pareto_data)
        fig2, ax2 = plt.subplots()
        ax2.scatter(pareto[:, 0], pareto[:, 1], alpha=0.5)
        ax2.set_xlabel("Average Delay")
        ax2.set_ylabel("Queue Length")
        st.pyplot(fig2)
