import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Traffic Light Optimization (GA)",
    layout="wide"
)

st.title("🚦 Traffic Light Optimization using Genetic Algorithm")
st.caption("JIE42903 – Evolutionary Computing")

# --------------------------------------------------
# Upload Dataset
# --------------------------------------------------
st.subheader("Upload Traffic Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file (Traffic Data)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("Dataset uploaded successfully")

st.subheader("Raw Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Data Cleaning & Feature Selection
# --------------------------------------------------
st.subheader("Data Cleaning & Feature Selection")

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_columns) < 2:
    st.error("Dataset must contain at least two numeric columns.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    flow_col = st.selectbox(
        "Select Traffic Flow Column (vehicles/hour)",
        numeric_columns
    )

with col2:
    queue_col = st.selectbox(
        "Select Queue / Vehicle Count Column",
        numeric_columns,
        index=1
    )

clean_df = df[[flow_col, queue_col]].dropna().astype(float)
clean_df.columns = ["flow_rate", "vehicle_count"]

st.subheader("Cleaned Dataset (GA Input)")
st.dataframe(clean_df.head())

# --------------------------------------------------
# Sidebar GA Parameters
# --------------------------------------------------
st.sidebar.header("Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 50, 300, 100)
GENERATIONS = st.sidebar.slider("Generations", 50, 300, 150)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.05, 0.5, 0.15)

TRAFFIC_FLOW = st.sidebar.slider(
    "Traffic Flow (vehicles/hour)",
    int(clean_df["flow_rate"].min()),
    int(clean_df["flow_rate"].max()),
    int(clean_df["flow_rate"].mean())
)

QUEUE_LENGTH = st.sidebar.slider(
    "Average Queue Length",
    int(clean_df["vehicle_count"].min()),
    int(clean_df["vehicle_count"].max()),
    int(clean_df["vehicle_count"].mean())
)

MODE = st.sidebar.radio(
    "Optimization Mode",
    ["Single Objective", "Multi Objective"]
)

if MODE == "Multi Objective":
    WAIT_WEIGHT = st.sidebar.slider("Waiting Time Weight", 0.1, 0.9, 0.6)
    QUEUE_WEIGHT = 1 - WAIT_WEIGHT

# --------------------------------------------------
# GA Configuration
# --------------------------------------------------
MIN_GREEN = 10
MAX_GREEN = 60
ELITE_RATE = 0.05

# --------------------------------------------------
# GA Functions
# --------------------------------------------------
def initialize_population(size):
    return [
        [random.randint(MIN_GREEN, MAX_GREEN),
         random.randint(MIN_GREEN, MAX_GREEN)]
        for _ in range(size)
    ]

def single_objective_fitness(individual, flow):
    waiting_time = flow / sum(individual)
    return 1 / (1 + waiting_time)

def multi_objective_fitness(individual, flow, queue, w1, w2):
    wait_score = 1 / (1 + (flow / sum(individual)))
    queue_score = 1 / (1 + queue)
    return w1 * wait_score + w2 * queue_score

def tournament_selection(pop, fitness, k=5):
    selected = random.sample(list(zip(pop, fitness)), k)
    return max(selected, key=lambda x: x[1])[0]

def crossover(p1, p2):
    if random.random() < 0.8:
        return [
            random.choice([p1[0], p2[0]]),
            random.choice([p1[1], p2[1]])
        ]
    return p1.copy()

def mutation(ind, rate):
    if random.random() < rate:
        i = random.randint(0, 1)
        ind[i] = random.randint(MIN_GREEN, MAX_GREEN)
    return ind

# --------------------------------------------------
# GA Execution
# --------------------------------------------------
def run_ga(mode):
    pop = initialize_population(POP_SIZE)
    best_fitness_history = []
    best_solution = None

    start = time.time()

    for gen in range(GENERATIONS):
        if mode == "Single Objective":
            fitness = [single_objective_fitness(i, TRAFFIC_FLOW) for i in pop]
        else:
            fitness = [
                multi_objective_fitness(
                    i, TRAFFIC_FLOW, QUEUE_LENGTH,
                    WAIT_WEIGHT, QUEUE_WEIGHT
                ) for i in pop
            ]

        # Elitism
        elite_count = max(1, int(ELITE_RATE * POP_SIZE))
        elite_idx = np.argsort(fitness)[-elite_count:]
        elites = [pop[i] for i in elite_idx]

        best_solution = elites[-1]
        best_fitness_history.append(max(fitness))

        new_pop = elites.copy()

        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(pop, fitness)
            p2 = tournament_selection(pop, fitness)
            child = crossover(p1, p2)
            child = mutation(child, MUTATION_RATE)
            new_pop.append(child)

        pop = new_pop

    exec_time = time.time() - start
    return best_solution, best_fitness_history, exec_time

# --------------------------------------------------
# Run Optimization
# --------------------------------------------------
st.subheader("Optimization Output")

if st.button("Run Genetic Algorithm"):
    with st.spinner("Running Genetic Algorithm..."):
        best_sol, fitness_hist, exec_time = run_ga(MODE)

    col1, col2 = st.columns(2)

    with col1:
        st.success("Optimal Traffic Light Timing")
        st.metric("Phase 1 Green Time (seconds)", best_sol[0])
        st.metric("Phase 2 Green Time (seconds)", best_sol[1])
        st.metric("Execution Time (seconds)", f"{exec_time:.4f}")

    with col2:
        fig1, ax1 = plt.subplots()
        ax1.plot(fitness_hist)
        ax1.set_title("GA Convergence Curve")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness Value")
        st.pyplot(fig1)

    st.subheader("Optimized Traffic Light Green Time Distribution")

    fig2, ax2 = plt.subplots()
    ax2.bar(["Phase 1", "Phase 2"], best_sol)
    ax2.set_ylabel("Green Time (seconds)")
    ax2.set_title("Optimized Green Signal Timing")
    st.pyplot(fig2)

st.success("✅ Genetic Algorithm Optimization Completed Successfully")
