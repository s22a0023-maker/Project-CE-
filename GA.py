import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt

# ==============================
# Load Dataset
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("/mnt/data/traffic_dataset.csv")

data = load_data()

# Assume dataset has:
# traffic_volume, avg_wait_time, queue_length
traffic_volume = data["traffic_volume"].values
base_wait_time = data["avg_wait_time"].values
queue_length = data["queue_length"].values

# ==============================
# GA PARAMETERS (Streamlit UI)
# ==============================
st.title("🚦 Traffic Light Optimization using Genetic Algorithm")

population_size = st.sidebar.slider("Population Size", 20, 200, 50)
generations = st.sidebar.slider("Generations", 50, 500, 200)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.5, 1.0, 0.8)
min_green = st.sidebar.slider("Min Green Time (s)", 5, 20, 10)
max_green = st.sidebar.slider("Max Green Time (s)", 30, 90, 60)

num_phases = 4  # Example: 4-direction intersection

# ==============================
# Fitness Function (Multi-objective)
# ==============================
def fitness_function(chromosome):
    """
    chromosome = [g1, g2, g3, g4]
    """
    total_wait = 0
    total_queue = 0

    for i in range(len(traffic_volume)):
        green_effect = sum(chromosome) / (traffic_volume[i] + 1)
        wait = max(0, base_wait_time[i] - green_effect)
        queue = max(0, queue_length[i] - green_effect)

        total_wait += wait
        total_queue += queue

    # Weighted sum (multi-objective)
    fitness = 0.7 * total_wait + 0.3 * total_queue
    return fitness

# ==============================
# GA OPERATORS
# ==============================
def create_individual():
    return [random.randint(min_green, max_green) for _ in range(num_phases)]

def create_population():
    return [create_individual() for _ in range(population_size)]

def selection(population):
    tournament = random.sample(population, 3)
    tournament.sort(key=lambda x: fitness_function(x))
    return tournament[0]

def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        point = random.randint(1, num_phases - 1)
        return parent1[:point] + parent2[point:]
    return parent1.copy()

def mutation(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(min_green, max_green)
    return individual

# ==============================
# GENETIC ALGORITHM
# ==============================
def genetic_algorithm():
    population = create_population()
    best_fitness_history = []
    avg_fitness_history = []

    start_time = time.time()

    for gen in range(generations):
        new_population = []

        fitness_values = [fitness_function(ind) for ind in population]
        best_fitness_history.append(min(fitness_values))
        avg_fitness_history.append(np.mean(fitness_values))

        for _ in range(population_size):
            p1 = selection(population)
            p2 = selection(population)
            child = crossover(p1, p2)
            child = mutation(child)
            new_population.append(child)

        population = new_population

    end_time = time.time()

    best_solution = min(population, key=lambda x: fitness_function(x))
    return best_solution, best_fitness_history, avg_fitness_history, end_time - start_time

# ==============================
# RUN GA
# ==============================
if st.button("▶ Run Genetic Algorithm"):
    best_solution, best_curve, avg_curve, exec_time = genetic_algorithm()

    st.subheader("✅ Best Optimized Traffic Signal Timings (seconds)")
    st.write(best_solution)

    st.subheader("📉 Convergence Analysis")
    fig, ax = plt.subplots()
    ax.plot(best_curve, label="Best Fitness")
    ax.plot(avg_curve, label="Average Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Value")
    ax.legend()
    st.pyplot(fig)

    st.subheader("⚙ Performance Metrics")
    st.write(f"**Final Best Fitness:** {best_curve[-1]:.2f}")
    st.write(f"**Execution Time:** {exec_time:.2f} seconds")
    st.write(f"**Convergence Generation:** {np.argmin(best_curve)}")

# ==============================
# MULTI-OBJECTIVE INSIGHT
# ==============================
st.subheader("📊 Multi-objective Insight")
st.markdown("""
- **Objective 1:** Minimize total vehicle waiting time  
- **Objective 2:** Minimize queue length  
- **Method:** Weighted-sum GA (0.7 waiting, 0.3 queue)  

This demonstrates how GA balances competing traffic efficiency goals.
""")
