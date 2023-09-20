from init_functions import create_initial_population, get_info_from_base, print_population
from evolu_algo import generation_manager, calc_fitness
from plot_functions import plot_convergence, plot_boxplot_trials_single_label, save_trial_results
import time
import numpy as np

if __name__ == "__main__":
    best_values_each_gen = []
    mean_values_each_gen = []
    time_spent_each_gen = []
    params = get_info_from_base("base1")

    init = time.time()
    

    for trial in range(1, params["N_TESTS"]):
        print(f'Starting Trial {trial}')
        init_time_trial = time.time()
        best_values = []
        mean_values = []
        population = create_initial_population("base1")

        for ind in population:
            ind.fitness = calc_fitness(ind)

        # print("Initial Population:")
        # print_population(population=population)
        for gen in range(1, params["GEN"]+1):
            population = generation_manager(population=population, params=params, gen_number=gen)
            best_individual = max(population, key=lambda x: x.fitness)
            mean_fitness = np.mean([ind.fitness for ind in population])
            mean_values.append(mean_fitness)
            best_values.append(best_individual.fitness)

        end_time_trial = time.time()

        time_spent_each_gen.append(end_time_trial-init_time_trial)
        best_values_each_gen.append(best_individual.fitness)
        mean_values_each_gen.append(mean_values)
        plot_convergence(generation=params["GEN"], best_values=best_values, mean_values=mean_values, n_dim=params["DIM"], save=True, trial=trial)

    plot_boxplot_trials_single_label(best_values_each_gen=best_values_each_gen, label=params["DIM"])
    save_trial_results(list_to_save=best_values_each_gen, label=params["DIM"], file_name='fitness')
    save_trial_results(list_to_save=mean_values_each_gen, label=params["DIM"], file_name='mean')
    save_trial_results(list_to_save=time_spent_each_gen, label=params["DIM"], file_name='time')

    end = time.time()

    print(f'Total time spent: {end - init}s')