from init_functions import create_initial_population, get_info_from_base, print_population
from evolu_algo import generation_manager
from plot_functions import plot_convergence, plot_boxplot_trials, save_trial_results

if __name__ == "__main__":
    best_values_each_gen = []
    params = get_info_from_base("base1")

    for trial in range(1, params["N_TESTS"]):
        best_values = []
        population = create_initial_population("base1")

        # print("Initial Population:")
        # print_population(population=population)
        for gen in range(1, params["GEN"]+1):
            population = generation_manager(population=population, params=params, gen_number=gen)
            best_individual = max(population, key=lambda x: x.fitness)
            best_values.append(best_individual.fitness)

        best_values_each_gen.append(best_individual.fitness)
        plot_convergence(generation=params["GEN"], best_values=best_values, n_dim=params["DIM"], save=True, trial=trial)

    plot_boxplot_trials(best_values_each_gen=best_values_each_gen, label=params["DIM"])
    save_trial_results(best_values_each_gen=best_values_each_gen, label=params["DIM"])