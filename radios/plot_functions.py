import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def plot_convergence(generation, best_values, n_dim = 0, trial=0, save=False):
    sns.set_style("whitegrid")  # Define um estilo para o gráfico
    generations = [i+1 for i in range(generation)]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_values, label="Convergência")
    plt.axhline(y=1, color="red", linestyle="--", label="y=1")  # Adiciona a linha vermelha em y=1
    plt.xlabel("Geração")
    plt.ylabel("Melhor Valor")
    plt.title(f"Gráfico de Convergência Trial {trial}")
    plt.legend()

    folder_path = f'results'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    figure_path = os.path.join(folder_path, f'convergence_trial_{trial}.png')
    plt.savefig(figure_path)

    # plt.clf()


def save_trial_results(list_to_save, label, file_name):
    # Create a comma-separated string of best values
    best_values_str = ', '.join(map(str, list_to_save))
    label = str(label)

    folder_path = f'results'

    # Create or use the specified folder path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path for the text file (you can change the filename)
    file_path = os.path.join(folder_path, f'{file_name}_{label}_results.txt')

    # Write the best values to the text file
    with open(file_path, 'w') as file:
        file.write(best_values_str)

    print(f"Results for '{label}' saved to '{file_path}'")


def save_any_result(text: str, label, file_name):
    # Create a comma-separated string of best values
    label = str(label)

    folder_path = f'results'

    # Create or use the specified folder path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path for the text file (you can change the filename)
    file_path = os.path.join(folder_path, f'{file_name}_{label}_results.txt')

    # Write the best values to the text file
    with open(file_path, 'w') as file:
        file.write(text)

    print(f"Results for '{label}' saved to '{file_path}'")


def plot_boxplot_trials(best_values_each_gen, label):
    label = str(label)
    folder_path = f'results'

    # Create or use the specified folder path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create a boxplot using Seaborn
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")
    sns.boxplot(data=best_values_each_gen, orient='v', width=0.4)
    plt.xlabel("Generation")
    plt.ylabel("Best Values")
    plt.title(f"Boxplot of Best Values for '{label}_queens' Trial")
    
    figure_path = os.path.join(folder_path, f'boxplot_trials_{label}.png')
    plt.savefig(figure_path)


def plot_full_convergence():
    df_fitness = pd.read_csv(f'results/df_fitness.csv', index_col=0)
    df_mean = pd.read_csv(f'results/df_mean.csv', index_col=0)

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
    plt.title(f"Convergence Plot of Genetic Algorithm with Standard Deviation for the problem of Radios")
    plt.legend(loc="upper right")
    
    
    folder_path = f'results'
    figure_path = os.path.join(folder_path, f'full_convergence.png')
    plt.savefig(figure_path)


if __name__ == "__main__": 
    plot_full_convergence()