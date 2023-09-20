import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def plot_convergence(generation, best_values, mean_values, n_dim = 0, trial=0, save=False):
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

    folder_path = f'results_{n_dim}_queens'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    figure_path = os.path.join(folder_path, f'convergence_trial_{trial}.png')
    plt.savefig(figure_path)

    # plt.clf()


def save_trial_results(list_to_save, label, file_name):
    # Create a comma-separated string of best values
    best_values_str = ', '.join(map(str, list_to_save))
    label = str(label)

    folder_path = f'results_{label}_queens'

    # Create or use the specified folder path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path for the text file (you can change the filename)
    file_path = os.path.join(folder_path, f'{file_name}_{label}_results.txt')

    # Write the best values to the text file
    with open(file_path, 'w') as file:
        file.write(best_values_str)

    print(f"Results for '{label}' saved to '{file_path}'")


def plot_boxplot_trials_single_label(best_values_each_gen, label):
    label = str(label)
    folder_path = f'results_{label}_queens'

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
        folder_name = f"results_{label}_queens"
        file_name = f"{type}_{label}_results.txt"
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
        total_runs = len(value_list)
        labels.extend([key] * len(value_list))
        values.extend(value_list)
    
    print(labels)
    df = pd.DataFrame(
        {
            type: values,
            "n_queens": labels,
        }
    )
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=type, y="n_queens")
    plt.title(f"Boxplot {type.capitalize()} - {total_runs} Trials")
    
    folder_path = f'results_n_queens'
    figure_path = os.path.join(folder_path, f'boxplot_all_trials_{type}.png')
    plt.savefig(figure_path)


if __name__ == "__main__":
    labels = ['8', '16', '32', '64', '128']
    trials_values_fitness = read_results(labels=labels, type="fitness")
    # trials_values_mean = read_results(labels=labels, type="mean")
    trials_values_time = read_results(labels=labels, type="time")

    plot_box_plots_all_labels(values_dict=trials_values_fitness, type="fitness")
    plot_box_plots_all_labels(values_dict=trials_values_time, type="time")
    # plot_box_plots_all_labels(values_dict=trials_values_fitness, type="mean")