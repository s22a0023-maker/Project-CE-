import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# ----------------------------------
# Load Dataset
# ----------------------------------
data = pd.read_csv("traffic_dataset.csv")

vehicle_count = data['vehicle_count'].values
avg_wait = data['avg_wait_time'].values
lanes = len(vehicle_count)

# ----------------------------------
# Fitness Function
# ----------------------------------
def fitness(green_times, alpha=0.8):
    waiting_time = np.sum((vehicle_count * avg_wait) / green_times)
    fairness = np.var(green_times)
    return alpha * waiting_time + (1 - alpha) * fairness

# ----------------------------------
# Genetic Algorithm
# ----------------------------------
def genetic_algorithm(pop_size, generations, mutation_rate, alpha):
    population = np.random.uniform(10, 60, (pop_size, lanes))
    history = []

    for gen in range(generations):
        fitness_scores = np.array([fitness(ind, alpha) for ind in population])
        history.append(fitness_scores.min())

        # Selection
        selected = population[np.argsort(fitness_scores)[:pop_size // 2]]

        # Crossover
        offspring = []
        while len(offspring) < pop_size:
            parents = random.sample(list(selected), 2)
            cp = random.randint(1, lanes - 1)
            child = np.concatenate((parents[0][:cp], parents[1][cp:]))
            offspring.append(child)

        offspring = np.array(offspring)

        # Mutation
        for i in range(pop_size):
            if random.random() < mutation_rate:
                idx = random.randint(0, lanes - 1)
                offspring[i][idx] += random.uniform(-5, 5)
                offspring[i][idx] = np.clip(offspring[i][idx], 10, 60)

        population = offspring

    best = population[np.argmin([fitness(ind, alpha) for ind in population])]
    return best, history

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.title("🚦 Traffic Light Optimization using Genetic Algorithm")

st.sidebar.header("GA Parameters")
population_size = st.sidebar.slider("Population Size", 10, 100, 30)
generations = st.sidebar.slider("Generations", 10, 100, 50)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.3, 0.1)
alpha = st.sidebar.slider("Objective Weight (Efficiency vs Fairness)", 0.0, 1.0, 0.8)

if st.button("Run Optimization"):
    best_solution, history = genetic_algorithm(
        population_size, generations, mutation_rate, alpha
    )

    st.subheader("Optimized Green Time (seconds)")
    for i, t in enumerate(best_solution):
        st.write(f"Lane {i+1}: {t:.2f} seconds")

    st.subheader("Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness Value")
    st.pyplot(fig)
