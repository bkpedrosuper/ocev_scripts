from init_functions import create_initial_population, get_info_from_base, expand_df, insert_values
from evolu_algo import generation_manager, calc_fitness
from plot_functions import plot_convergence, plot_boxplot_trials_single_label, save_trial_results
import time
import numpy as np
import pandas as pd

from board import Board

if __name__ == "__main__":


    best_values_each_gen = []
    mean_values_each_gen = []
    time_spent_each_gen = []
    params = get_info_from_base("base1")
    board = Board(params["DIM"])
    
    # dfss_fitness = []
    # dfss_mean = []
    
    
    # init = time.time()

    # def run_trial(trial_number):
    #     df_fitness = pd.DataFrame()
    #     df_mean = pd.DataFrame()

    #     print(f'Starting Trial {trial_number}')
    #     init_time_trial = time.time()
    #     best_values = []
    #     mean_values = []
    #     population = create_initial_population("base1")

    #     for ind in population:
    #         ind.fitness = calc_fitness(ind)
        
    #     gen = 0
    #     best = 0
    #     while best < 1:
    #         population = generation_manager(population=population, params=params, gen_number=gen)
    #         best_individual = max(population, key=lambda x: x.fitness)
    #         mean_fitness = np.mean([ind.fitness for ind in population])
    #         mean_values.append(mean_fitness)
    #         best_values.append(best_individual.fitness)
    #         best = best_individual.fitness
    #         gen+=1

    #     print(f'Result achieved with {gen} generations')
    #     end_time_trial = time.time()

    #     time_spent_each_gen.append(end_time_trial-init_time_trial)
    #     best_values_each_gen.append(best_individual.fitness)
    #     mean_values_each_gen.append(mean_fitness)

    #     # Insert values into df
    #     df_fitness = insert_values(df_fitness, best_values, trial_number)
    #     df_mean = insert_values(df_mean, mean_values, trial_number)

    #     dfss_fitness.append(df_fitness)
    #     dfss_mean.append(df_mean)


    # for trial in range(0, params["N_TESTS"]):
    #     run_trial(trial)

    # max_gens = max([df.shape[0] for df in dfss_fitness])
    # max_means = max([df.shape[0] for df in dfss_mean])

    # print(max_gens)
    # print(max_means)

    # # Expand dfs to fill
    # all_dfs_fitness = [expand_df(df_fitness, max_gens) for df_fitness in dfss_fitness]
    # all_dfs_means = [expand_df(df_means, max_gens) for df_means in dfss_mean]
    
    # df_fitness_final = pd.concat(all_dfs_fitness, axis=1)
    # df_means_final = pd.concat(all_dfs_means, axis=1)

    
    # df_fitness_final.to_csv(f"results_{params['DIM']}_queens/df_fitness.csv")
    # df_means_final.to_csv(f"results_{params['DIM']}_queens/df_mean.csv")

    # end = time.time()

    # print(f'Total time spent: {end - init}s')