from init_functions import create_initial_population, get_info_from_base, insert_values, expand_df
from evolu_algo import generation_manager, calc_fitness
from plot_functions import plot_convergence, save_trial_results, save_any_result
import pandas as pd
import numpy as np
import time
from radio import Radio

# PARALLEL
from joblib import Parallel, delayed
from multiprocessing import Manager

if __name__ == "__main__":
    params = get_info_from_base("base1")

    init = time.time()

    manager = Manager()

    dfss_fitness = []
    dfss_mean = []

    best_values_each_trial = manager.list()
    best_individual_each_trial = manager.list()
    mean_values_each_trial = manager.list()
    time_spent_each_trial = manager.list()

    # INIT TRIAL
    for trial_number in range(params["N_TESTS"]):
        df_fitness = pd.DataFrame()
        df_mean = pd.DataFrame()

        best_values = []
        mean_values = []

        init_time_trial = time.time()
        population = create_initial_population("base1")

        for ind in population:
            ind.fitness = calc_fitness(ind)

        for gen in range(1, params["GEN"]+1):
            population = generation_manager(population=population, params=params, gen_number=gen)
            best_individual = max(population, key=lambda x: x.fitness)
            best_values.append(best_individual.fitness)
            mean_fitness = np.mean([ind.fitness for ind in population])
            mean_values.append(mean_fitness)

            print(f'Best Individual: {best_individual.fitness}')
        
        plot_convergence(generation=params["GEN"], best_values=best_values, n_dim=params["DIM"], save=True, trial=trial_number)
        
        end_time_trial = time.time()

        best_individual_each_trial.append(best_individual)
        time_spent_each_trial.append(end_time_trial-init_time_trial)
        best_values_each_trial.append(best_individual.fitness)
        mean_values_each_trial.append(mean_fitness)

        # GET THE BEST INDIVIDUAL
        best_individual = max(best_individual_each_trial, key=lambda x: x.fitness)

        print(calc_fitness(best_individual))

        # Insert values into df
        df_fitness = insert_values(df_fitness, best_values, trial_number)
        df_mean = insert_values(df_mean, mean_values, trial_number)

        # SAVE TRIAL RESULTS
        df_fitness.to_csv(f'results/fitness_trial_{trial_number}.csv')
        df_mean.to_csv(f'results/mean_trial_{trial_number}.csv')
    
        # GET RESULTS FOR EVERY TRIAL
    
    # END TRIALS

    dfss_fitness = []
    dfss_mean = []

    for trial_number in range(0, params["N_TESTS"]):
        df_fitness_trial = pd.read_csv(f'results/fitness_trial_{trial_number}.csv', index_col=0)
        df_mean_trial = pd.read_csv(f'results/mean_trial_{trial_number}.csv', index_col=0)
        dfss_fitness.append(df_fitness_trial)
        dfss_mean.append(df_mean_trial)

    max_gens = max([df.shape[0] for df in dfss_fitness])
    max_means = max([df.shape[0] for df in dfss_mean])

    # Expand dfs to fill
    all_dfs_fitness = [expand_df(df_fitness, max_gens) for df_fitness in dfss_fitness]
    all_dfs_means = [expand_df(df_means, max_gens) for df_means in dfss_mean]

    df_fitness_final = pd.concat(all_dfs_fitness, axis=1)
    df_means_final = pd.concat(all_dfs_means, axis=1)

    df_fitness_final.to_csv(f"results/df_fitness.csv")
    df_means_final.to_csv(f"results/df_mean.csv")

    best_individual_all: Radio = max(best_individual_each_trial, key=lambda x: x.fitness)

    save_trial_results(list_to_save=best_values_each_trial, label="radios", file_name='fitness')
    save_trial_results(list_to_save=mean_values_each_trial, label="radios", file_name='mean')
    save_trial_results(list_to_save=time_spent_each_trial, label="radios", file_name='time')
    save_any_result(text=best_individual_all.get_info(), label="radios", file_name='best')


    end = time.time()

    print(f'Total time spent: {end - init}s')
