import matplotlib.pyplot as plt
import seaborn as sns
import os

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


def save_trial_results(best_values_each_gen, label):
    # Create a comma-separated string of best values
    best_values_str = ', '.join(map(str, best_values_each_gen))
    label = str(label)

    folder_path = f'results'

    # Create or use the specified folder path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path for the text file (you can change the filename)
    file_path = os.path.join(folder_path, f'{label}_results.txt')

    # Write the best values to the text file
    with open(file_path, 'w') as file:
        file.write(best_values_str)

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