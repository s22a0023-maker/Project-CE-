import streamlit as st
import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
import time

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Traffic Light Optimization (GA)",
    layout="wide"
)

st.title("🚦 Traffic Light Optimization using Genetic Algorithm")
st.write("Computational Evolution Case Study")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("traffic_dataset.csv")

df = load_data()

st.subheader("Traffic Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Sidebar Parameters
# -------------------------------
st.sidebar.header("GA Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 10, 100, 30)
GENERATIONS = st.sidebar.slider("Generations", 10, 200, 50)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

TRAFFIC_FLOW = st.sidebar.slider(
    "Traffic Flow (vehicles/hour)", 
    int(df["flow_rate"].min()), 
    int(df["flow_rate"].max()), 
    int(df["flow_rate"].mean())
)

QUEUE_LENGTH = st.sidebar.slider(
    "Average Queue Length", 
    int(df["lane_occupancy"].min()), 
    int(df["lane_occupancy"].max()), 
    int(df["lane_occupancy"].mean())
)

MODE = st.sidebar.radio(
    "Optimization Mode",
    ["Single Objective", "Multi Objective"]
)

# -------------------------------
# GA Functions
# -------------------------------
MIN_GREEN = 10
MAX_GREEN = 60

def initialize_population(size):
    return [
        [random.randint(MIN_GREEN, MAX_GREEN),
         random.randint(MIN_GREEN, MAX_GREEN)]
        for _ in range(size)
    ]

def single_objective_fitness(individual, flow_rate):
    avg_wait = flow_rate / sum(individual)
    return 1 / (1 + avg_wait)

def multi_objective_fitness(individual, flow_rate, lane_occupancy):
    wait_score = 1 / (1 + (flow_rate / sum(individual)))
    queue_score = 1 / (1 + lane_occupancy)
    return 0.6 * wait_score + 0.4 * queue_score

def selection(population, fitnesses):
    return population[np.argmax(fitnesses)]

def crossover(parent1, parent2):
    point = random.randint(0, 1)
    return parent1[:point] + parent2[point:]

def mutation(individual, rate):
    if random.random() < rate:
        idx = random.randint(0, 1)
        individual[idx] = random.randint(MIN_GREEN, MAX_GREEN)
    return individual

# -------------------------------
# GA Execution
# -------------------------------
def run_ga(mode):
    population = initialize_population(POP_SIZE)
    best_fitness_history = []
    best_solution = None

    start_time = time.time()

    for _ in range(GENERATIONS):
        if mode == "Single Objective":
            fitnesses = [
                single_objective_fitness(ind, TRAFFIC_FLOW)
                for ind in population
            ]
        else:
            fitnesses = [
                multi_objective_fitness(ind, TRAFFIC_FLOW, QUEUE_LENGTH)
                for ind in population
            ]

        best_idx = np.argmax(fitnesses)
        best_solution = population[best_idx]
        best_fitness_history.append(fitnesses[best_idx])

        new_population = []
        for _ in range(POP_SIZE):
            parent1 = selection(population, fitnesses)
            parent2 = random.choice(population)
            child = crossover(parent1, parent2)
            child = mutation(child, MUTATION_RATE)
            new_population.append(child)

        population = new_population

    exec_time = time.time() - start_time
    return best_solution, best_fitness_history, exec_time

# -------------------------------
# Run Optimization
# -------------------------------
st.subheader("Optimization Results")

if st.button("Run Genetic Algorithm"):
    with st.spinner("Running Genetic Algorithm..."):
        best_solution, fitness_history, exec_time = run_ga(MODE)

    col1, col2 = st.columns(2)

    with col1:
        st.success("Best Traffic Light Timing Found")
        st.write(f"🚦 Phase 1 Green Time: **{best_solution[0]} seconds**")
        st.write(f"🚦 Phase 2 Green Time: **{best_solution[1]} seconds**")
        st.write(f"⏱ Execution Time: **{exec_time:.4f} seconds**")

    with col2:
        fig, ax = plt.subplots()
        ax.plot(fitness_history)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness Value")
        ax.set_title("GA Convergence Curve")
        st.pyplot(fig)

# -------------------------------
# Performance Analysis Section
# -------------------------------
st.subheader("Performance Analysis")

st.markdown("""
**Key Metrics Evaluated:**
- **Convergence Rate:** Speed at which fitness stabilizes
- **Accuracy:** Ability to reduce waiting time and congestion
- **Computational Efficiency:** Execution time

**Observations:**
- Rapid improvement during early generations
- Stable convergence after sufficient iterations
- Multi-objective optimization balances competing goals
""")

# -------------------------------
# Conclusion
# -------------------------------
st.subheader("Conclusion")

st.markdown("""
This Streamlit-based system demonstrates how **Genetic Algorithms** can be effectively applied to 
traffic light optimization problems. The interactive dashboard allows users to explore parameter 
effects, compare optimization strategies, and visualize convergence behavior in real time.
""")

st.success("✅ End of Computational Evolution Case Study")
