import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Traffic Light Optimization (GA)",
    layout="wide"
)

st.title("🚦 Traffic Light Optimization using Genetic Algorithm")
st.write("Computational Evolution Case Study – Realistic GA Implementation")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("traffic_dataset.csv")

df = load_data()

st.subheader("Traffic Dataset Preview")
st.dataframe(df.head())

# ===============================
# SIDEBAR PARAMETERS
# ===============================
st.sidebar.header("GA Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 20, 100, 40)
GENERATIONS = st.sidebar.slider("Generations", 20, 200, 80)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.3, 0.08)

TRAFFIC_FLOW = st.sidebar.slider(
    "Traffic Flow (vehicles/hour)",
    int(df["flow_rate"].min()),
    int(df["flow_rate"].max()),
    int(df["flow_rate"].mean())
)

QUEUE_LENGTH = st.sidebar.slider(
    "Opposite Direction Flow (vehicles/hour)",
    int(df["vehicle_count"].min()),
    int(df["vehicle_count"].max()),
    int(df["vehicle_count"].mean())
)

# ===============================
# GA CONFIGURATION
# ===============================
MIN_CYCLE = 40
MAX_CYCLE = 120
SATURATION_FLOW = 1800  # veh/hour/lane

# ===============================
# INITIAL POPULATION
# ===============================
def initialize_population(size):
    population = []
    for _ in range(size):
        cycle = random.randint(MIN_CYCLE, MAX_CYCLE)
        green_ratio = random.uniform(0.3, 0.7)
        population.append([cycle, green_ratio])
    return population

# ===============================
# TRAFFIC DELAY MODEL (WEBSTER)
# ===============================
def average_delay(flow, green, cycle):
    g_c = green / cycle

    if g_c <= 0 or g_c >= 1:
        return 1e6

    x = flow / (SATURATION_FLOW * g_c)

    if x >= 1:
        return 1e6

    delay = (cycle * (1 - g_c) ** 2) / (2 * (1 - x))
    return delay

# ===============================
# FITNESS FUNCTION
# ===============================
def fitness(individual):
    cycle, ratio = individual
    green_1 = cycle * ratio
    green_2 = cycle * (1 - ratio)

    delay_1 = average_delay(TRAFFIC_FLOW, green_1, cycle)
    delay_2 = average_delay(QUEUE_LENGTH, green_2, cycle)

    total_delay = delay_1 + delay_2
    return 1 / (1 + total_delay)

# ===============================
# TOURNAMENT SELECTION
# ===============================
def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    return max(selected, key=lambda x: x[1])[0]

# ===============================
# CROSSOVER
# ===============================
def crossover(p1, p2):
    alpha = random.random()
    child_cycle = int(alpha * p1[0] + (1 - alpha) * p2[0])
    child_ratio = alpha * p1[1] + (1 - alpha) * p2[1]
    return [child_cycle, child_ratio]

# ===============================
# MUTATION
# ===============================
def mutation(individual):
    if random.random() < MUTATION_RATE:
        individual[0] += random.randint(-10, 10)
        individual[0] = int(np.clip(individual[0], MIN_CYCLE, MAX_CYCLE))

    if random.random() < MUTATION_RATE:
        individual[1] += random.uniform(-0.05, 0.05)
        individual[1] = float(np.clip(individual[1], 0.3, 0.7))

    return individual

# ===============================
# GA EXECUTION
# ===============================
def run_ga():
    population = initialize_population(POP_SIZE)
    fitness_history = []

    start_time = time.time()

    for _ in range(GENERATIONS):
        fitnesses = [fitness(ind) for ind in population]

        # Elitism
        elite_idx = np.argmax(fitnesses)
        elite = population[elite_idx]
        fitness_history.append(fitnesses[elite_idx])

        new_population = [elite]

        while len(new_population) < POP_SIZE:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            child = crossover(p1, p2)
            child = mutation(child)
            new_population.append(child)

        population = new_population

    exec_time = time.time() - start_time
    return elite, fitness_history, exec_time

# ===============================
# RUN OPTIMIZATION
# ===============================
st.subheader("Optimization Results")

if st.button("Run Genetic Algorithm"):
    with st.spinner("Optimizing traffic signal timings..."):
        best_solution, fitness_history, exec_time = run_ga()

    cycle, ratio = best_solution
    green_1 = int(cycle * ratio)
    green_2 = int(cycle * (1 - ratio))

    col1, col2 = st.columns(2)

    with col1:
        st.success("Optimal Traffic Signal Timing")
        st.write(f"🚦 Cycle Length: **{cycle} seconds**")
        st.write(f"🚦 Phase 1 Green: **{green_1} seconds**")
        st.write(f"🚦 Phase 2 Green: **{green_2} seconds**")
        st.write(f"⏱ Execution Time: **{exec_time:.4f} seconds**")

    with col2:
        fig, ax = plt.subplots()
        ax.plot(fitness_history)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.set_title("GA Convergence Curve")
        st.pyplot(fig)

# ===============================
# PERFORMANCE ANALYSIS
# ===============================
st.subheader("Performance Analysis")

st.markdown("""
**Performance Metrics:**
- Convergence speed across generations  
- Reduction in total vehicle delay  
- Computational efficiency  

**Findings:**
- GA rapidly improves signal timing in early generations  
- Elitism ensures stable convergence  
- Optimized green splits reduce congestion effectively  
""")

# ===============================
# CONCLUSION
# ===============================
st.subheader("Conclusion")

st.markdown("""
This study demonstrates the effectiveness of **Genetic Algorithms** in optimizing 
traffic signal timings under varying traffic conditions. The model integrates 
realistic delay estimation and evolutionary operators, making it suitable for 
real-world traffic control simulations and academic evaluation.
""")

st.success("✅ End of Computational Evolution Case Study")
