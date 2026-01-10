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

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully")

    st.subheader("Raw Dataset Preview")
    st.dataframe(df.head())
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()


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
        index=1 if len(numeric_columns) > 1 else 0
    )

# Clean dataset
clean_df = df[[flow_col, queue_col]].copy()
clean_df = clean_df.dropna()
clean_df = clean_df.astype(float)

clean_df.columns = ["flow_rate", "vehicle_count"]

st.subheader("Cleaned Dataset (GA Input)")
st.dataframe(clean_df.head())

st.info(
    f"Cleaned data contains {len(clean_df)} rows "
    "and is ready for Genetic Algorithm optimization."
)




# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 20, 80, 40)
GENERATIONS = st.sidebar.slider("Generations", 20, 150, 60)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.3, 0.1)

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

def selection(pop, fitness):
    return pop[np.argmax(fitness)]

def crossover(p1, p2):
    return [p1[0], p2[1]] if random.random() < 0.5 else [p2[0], p1[1]]

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
    fitness_history = []
    best_solution = None

    start = time.time()

    for _ in range(GENERATIONS):
        if mode == "Single Objective":
            fitness = [single_objective_fitness(i, TRAFFIC_FLOW) for i in pop]
        else:
            fitness = [
                multi_objective_fitness(
                    i, TRAFFIC_FLOW, QUEUE_LENGTH, WAIT_WEIGHT, QUEUE_WEIGHT
                ) for i in pop
            ]

        best_idx = np.argmax(fitness)
        best_solution = pop[best_idx]
        fitness_history.append(fitness[best_idx])

        new_pop = []
        for _ in range(POP_SIZE):
            p1 = selection(pop, fitness)
            p2 = random.choice(pop)
            child = crossover(p1, p2)
            child = mutation(child, MUTATION_RATE)
            new_pop.append(child)

        pop = new_pop

    exec_time = time.time() - start
    return best_solution, fitness_history, exec_time

# --------------------------------------------------
# Run Optimization
# --------------------------------------------------
st.subheader("Optimization Output")

if st.button("Run Genetic Algorithm"):
    with st.spinner("Optimizing traffic signal timings..."):
        best_sol, fitness_hist, exec_time = run_ga(MODE)

    col1, col2 = st.columns(2)

    # ---------------- Results ----------------
    with col1:
        st.success("Optimal Traffic Light Timing")
        st.metric("Phase 1 Green Time (seconds)", best_sol[0])
        st.metric("Phase 2 Green Time (seconds)", best_sol[1])
        st.metric("Execution Time (seconds)", f"{exec_time:.4f}")

    # ---------------- Graph 1: Convergence ----------------
    with col2:
        fig1, ax1 = plt.subplots()
        ax1.plot(fitness_hist)
        ax1.set_title("GA Convergence Curve")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness Value")
        st.pyplot(fig1)

    # ---------------- Graph 2: Traffic Light Optimization ----------------
    st.subheader("Optimized Traffic Light Green Time Distribution")

    fig2, ax2 = plt.subplots()
    phases = ["Phase 1", "Phase 2"]
    ax2.bar(phases, best_sol)
    ax2.set_ylabel("Green Time (seconds)")
    ax2.set_title("Optimized Green Signal Timing per Phase")
    st.pyplot(fig2)

st.success("✅ Optimization Completed")
