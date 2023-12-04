import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from path import Path
from metro_line import Metro
import copy
import math

def plot_heatmap_grid(individuals, board, cols = 3):
    num_individuals = len(individuals)

    rows = math.ceil(num_individuals/cols)
    
    # Configuração da figura e subplots
    fig, axes = plt.subplots(rows, cols, figsize=(36, 12*rows))
    fig.suptitle("Grid of Paths", fontsize=16)

    # Loop através dos indivíduos e seus respectivos subplots
    for i in range(num_individuals):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        ind = individuals[i]
        get_plot_path(ind, board, ax, label=str(num_individuals))
        ax.set_title(f'Individual {i + 1}')

    # Remova subplots vazios, se houverem
    for i in range(num_individuals, rows * cols):
        fig.delaxes(axes[i // cols, i % cols])

    plt.savefig('Paths Found.png')
    plt.clf()


def get_plot_path(ind: Path, board: Metro, ax: any, label: str, best=False,):
    board_decode = copy.deepcopy(board)
    pos = board_decode.start
    directions = ind.decode(board_decode)
    for direction in directions:
        pos = ind.get_new_pos(direction, pos)
        if board.get_value(pos) == 3:
            board_decode.matrix[pos[0]][pos[1]] = 5
        else:
            board_decode.matrix[pos[0]][pos[1]] = 4
    if best:
        sns.heatmap(board_decode.matrix, annot=False, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
        plt.savefig(f'Best_result_{label}.png')
        plt.clf()
        return None

    return sns.heatmap(board_decode.matrix, annot=False, cmap='viridis', ax=ax, cbar=False, xticklabels=False, yticklabels=False)
        

def plot_convergence(generation, best_values, mean_values, n_dim = 0, trial=0, save=False, label='board1'):
    sns.set_style("whitegrid")  # Define um estilo para o gráfico
    generations = [i+1 for i in range(generation)]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_values, label="Best - Fitness")
    plt.plot(generations, mean_values, label="Mean - Fitness (population)")
    plt.axhline(y=1, color="red", linestyle="--", label="y=1")  # Adiciona a linha vermelha em y=1
    plt.xlabel("Generation")
    plt.ylabel("Fitness Value")
    plt.title(f"Convergence for Trial {trial}")
    plt.legend()

    folder_path = f'results_{label}_maze'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    figure_path = os.path.join(folder_path, f'convergence_trial_{trial}.png')
    plt.savefig(figure_path)

    plt.clf()


def save_trial_results(list_to_save, label, file_name):
    # Create a comma-separated string of best values
    best_values_str = ', '.join(map(str, list_to_save))
    label = str(label)

    folder_path = f'results_{label}'

    # Create or use the specified folder path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path for the text file (you can change the filename)
    file_path = os.path.join(folder_path, f'{file_name}_results.txt')

    # Write the best values to the text file
    with open(file_path, 'w') as file:
        file.write(best_values_str)

    print(f"Results for '{label}' saved to '{file_path}'")


def save_any_result(text: str, label, file_name):
    # Create a comma-separated string of best values
    label = str(label)

    folder_path = f'results_{label}'

    # Create or use the specified folder path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path for the text file (you can change the filename)
    file_path = os.path.join(folder_path, f'{file_name}_{label}_results.txt')

    # Write the best values to the text file
    with open(file_path, 'w') as file:
        file.write(text)

    print(f"Results for '{label}' saved to '{file_path}'")



def plot_boxplot_trials_single_label(best_values_each_gen, label):
    print(best_values_each_gen)
    label = str(label)
    folder_path = f'results_{label}'

    # Create or use the specified folder path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create a boxplot using Seaborn
    plt.figure(figsize=(40, 30))
    sns.set(style="whitegrid")
    sns.boxplot(data=best_values_each_gen, orient='v', width=0.4)
    plt.xlabel("Generation")
    plt.ylabel("Best Values")
    plt.title(f"Boxplot of Best Values for '{label}_queens' Trial")
    
    figure_path = os.path.join(folder_path, f'boxplot_trials_{label}.png')
    plt.savefig(figure_path)


def read_results(labels: list[str], type: str) -> dict:
    results = {}
    for label in labels:
        folder_name = f"results_{label}"
        file_name = f"{type}_results.txt"
        file_path = os.path.join(folder_name, file_name)

        print(file_path)

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
                fitness_list = [float(x.strip()) for x in content.split(',')]
                results[label] = fitness_list
        else:
            results[label] = None  # No file found for this iteration

    return results


def plot_box_plots_all_labels(values_dict: dict, type: str):
    labels = []
    values = []
    total_runs = len(values_dict)

    for key, value_list in values_dict.items():
        # Extend labels and values with the same key multiple times to match lengths
        labels.extend([key] * len(value_list))
        values.extend(value_list)
    
    df = pd.DataFrame(
        {
            type: values,
            "boards": labels,
        }
    )
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=type, y="boards")
    plt.title(f"Boxplot {type.capitalize()} - {total_runs} Trials")
    
    folder_path = f'results_maze'
    figure_path = os.path.join(folder_path, f'boxplot_all_trials_{type}.png')
    plt.savefig(figure_path)


def plot_full_convergence(label: str = 'board1'):
    df_fitness = pd.read_csv(f'results_{label}/df_fitness.csv', index_col=0)
    df_mean = pd.read_csv(f'results_{label}/df_mean.csv', index_col=0)

    # Calculate the mean and standard deviation for each generation
    mean_fitness = df_fitness.mean(axis=1)
    std_fitness = df_fitness.std(axis=1)

    mean_mean = df_mean.mean(axis=1)
    std_mean = df_mean.std(axis=1)

    # Create a line plot for the mean fitness values
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.lineplot(data=mean_fitness, label="Fitness")
    sns.lineplot(data=mean_mean, label="Mean")

    # Plot the standard deviation as error bars
    plt.fill_between(mean_fitness.index, mean_fitness - std_fitness, mean_fitness + std_fitness, alpha=0.3, color="blue", label="STD Fitness")
    plt.fill_between(mean_mean.index, mean_mean - std_mean, mean_mean + std_mean, alpha=0.3, color="orange", label="STD Mean")


    plt.xlabel("Generation Number")
    plt.ylabel("Fitness Value")
    plt.title(f"Convergence Plot of Genetic Algorithm with Standard Deviation for Maze")
    plt.legend(loc="upper right")
    
    
    folder_path = f'results_{label}_maze'
    figure_path = os.path.join(folder_path, f'full_convergence_maze.png')
    plt.savefig(figure_path)


if __name__ == "__main__":
    labels = ['board1']
    trials_values_fitness = read_results(labels=labels, type="fitness")
    trials_values_mean = read_results(labels=labels, type="mean")
    trials_values_time = read_results(labels=labels, type="time")

    plot_box_plots_all_labels(values_dict=trials_values_fitness, type="fitness")
    plot_box_plots_all_labels(values_dict=trials_values_time, type="time")
    
    for label in labels:
        plot_full_convergence(label=label)
