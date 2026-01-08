import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ==============================
# Load Dataset
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("traffic_dataset.csv")

data = load_data()

flow_rate = data["flow_rate"].values
base_wait = data["waiting_time"].values
base_queue = data["vehicle_count"].values

# ==============================
# Streamlit UI
# ==============================
st.title("🚦 Traffic Light Optimization using NSGA-II")

population_size = st.sidebar.slider("Population Size", 50, 300, 100)
generations = st.sidebar.slider("Generations", 50, 500, 200)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
min_green = st.sidebar.slider("Min Green Time (s)", 5, 20, 10)
max_green = st.sidebar.slider("Max Green Time (s)", 30, 90, 60)

num_phases = 4

# ==============================
# Objective Functions
# ==============================
def evaluate(individual):
    total_wait = 0
    total_queue = 0

    green_effect = sum(individual)

    for i in range(len(flow_rate)):
        wait = max(0, base_wait[i] - green_effect / (flow_rate[i] + 1))
        queue = max(0, base_queue[i] - green_effect / (flow_rate[i] + 1))
        total_wait += wait
        total_queue += queue

    return total_wait, total_queue

# ==============================
# NSGA-II Utilities
# ==============================
def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def fast_non_dominated_sort(pop, scores):
    fronts = [[]]
    domination_count = [0] * len(pop)
    dominated = [[] for _ in pop]

    for i in range(len(pop)):
        for j in range(len(pop)):
            if dominates(scores[i], scores[j]):
                dominated[i].append(j)
            elif dominates(scores[j], scores[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    f = 0
    while fronts[f]:
        next_front = []
        for i in fronts[f]:
            for j in dominated[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        f += 1
        fronts.append(next_front)

    return fronts[:-1]

def crowding_distance(front, scores):
    distance = [0] * len(front)
    for m in range(2):
        sorted_idx = sorted(range(len(front)), key=lambda i: scores[front[i]][m])
        distance[sorted_idx[0]] = distance[sorted_idx[-1]] = float("inf")
        min_val = scores[front[sorted_idx[0]]][m]
        max_val = scores[front[sorted_idx[-1]]][m]
        if max_val - min_val == 0:
            continue
        for i in range(1, len(front) - 1):
            distance[sorted_idx[i]] += (
                scores[front[sorted_idx[i + 1]]][m]
                - scores[front[sorted_idx[i - 1]]][m]
            ) / (max_val - min_val)
    return distance

# ==============================
# Genetic Operators
# ==============================
def create_individual():
    return [random.randint(min_green, max_green) for _ in range(num_phases)]

def crossover(p1, p2):
    point = random.randint(1, num_phases - 1)
    return p1[:point] + p2[point:]

def mutation(ind):
    for i in range(len(ind)):
        if random.random() < mutation_rate:
            ind[i] = random.randint(min_green, max_green)
    return ind

# ==============================
# NSGA-II Algorithm
# ==============================
def nsga2():
    population = [create_individual() for _ in range(population_size)]

    for gen in range(generations):
        offspring = []
        while len(offspring) < population_size:
            p1, p2 = random.sample(population, 2)
            child = crossover(p1, p2)
            child = mutation(child)
            offspring.append(child)

        combined = population + offspring
        scores = [evaluate(ind) for ind in combined]

        fronts = fast_non_dominated_sort(combined, scores)
        new_population = []

        for front in fronts:
            if len(new_population) + len(front) > population_size:
                distances = crowding_distance(front, scores)
                sorted_front = sorted(
                    range(len(front)),
                    key=lambda i: distances[i],
                    reverse=True,
                )
                for i in sorted_front:
                    if len(new_population) < population_size:
                        new_population.append(combined[front[i]])
                break
            else:
                new_population.extend([combined[i] for i in front])

        population = new_population

    final_scores = [evaluate(ind) for ind in population]
    pareto_front = fast_non_dominated_sort(population, final_scores)[0]

    return population, final_scores, pareto_front

# ==============================
# Run NSGA-II
# ==============================
if st.button("▶ Run NSGA-II Optimization"):
    population, scores, pareto_front = nsga2()

    st.subheader("🏆 Pareto-Optimal Traffic Signal Solutions")

    pareto_wait = [scores[i][0] for i in pareto_front]
    pareto_queue = [scores[i][1] for i in pareto_front]

    fig, ax = plt.subplots()
    ax.scatter(
        [s[0] for s in scores],
        [s[1] for s in scores],
        alpha=0.3,
        label="Population"
    )
    ax.scatter(
        pareto_wait,
        pareto_queue,
        color="red",
        label="Pareto Front"
    )
    ax.set_xlabel("Total Waiting Time")
    ax.set_ylabel("Total Queue Length")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🔍 Sample Pareto-Optimal Green Times")
    for idx in pareto_front[:5]:
        st.write(
            f"Green Times: {population[idx]} | "
            f"Waiting: {scores[idx][0]:.2f} | "
            f"Queue: {scores[idx][1]:.2f}"
        )
