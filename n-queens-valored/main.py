from init_functions import create_initial_population, get_info_from_base, expand_df, insert_values, str_individual_plus_board
from evolu_algo import generation_manager, calc_fitness
from plot_functions import plot_convergence, plot_boxplot_trials_single_label, save_trial_results, save_any_result
import time
import numpy as np
import pandas as pd

from board import Board
from queen import Individual

# PARALLEL
from joblib import Parallel, delayed
from multiprocessing import Manager
import sys


if __name__ == "__main__":

    # GET BASE IF IT's passed by params
    base = "base1"
    if len(sys.argv) > 1:
        base = sys.argv[1]

    manager = Manager()

    best_values_each_trial = manager.list()
    best_individual_each_trial: list[Individual] = manager.list()
    mean_values_each_trial = manager.list()
    time_spent_each_trial = manager.list()
    params = get_info_from_base(base)
    print(params["DIM"])
    board = Board(params["DIM"])    
    
    init = time.time()
    print(f'INIT GA with {base} and {params["N_TESTS"]} TRIALS')
    def run_trial(trial_number):
        df_fitness = pd.DataFrame()
        df_mean = pd.DataFrame()

        print(f'Starting Trial {trial_number}')
        init_time_trial = time.time()
        best_values = []
        mean_values = []
        raw_population = create_initial_population(base) # get initial population

        # Calculate the fitness value for the population
        population = []
        for ind in raw_population:
            fitness, colisions = calc_fitness(ind, board)
            new_individual = Individual(queens=ind.queens, fitness=fitness, colisions=colisions)
            population.append(new_individual)
        
        gen = 0
        best = 0
        current_best = 0
        gen_no_increment = 0
        # while best < 1:
        for _ in range(params["GEN"]):
            population = generation_manager(population=population, params=params, gen_number=gen, board=board)
            best_individual = max(population, key=lambda x: x.fitness)
            mean_fitness = np.mean([ind.fitness for ind in population])
            mean_values.append(mean_fitness)
            best_values.append(best_individual.fitness)
            best = best_individual.fitness
            gen+=1

            # for ind in population:
            #     print_individual_plus_board(ind, board)

            if best == current_best:
                gen_no_increment += 1
            else:
                current_best = best
                gen_no_increment = 0
            
            if gen_no_increment == params["PATIENCE"]:
                print(f"Best fitness didn't increment for {params['PATIENCE']} generations")
                break
        
        print(f'Result achieved with {gen} generations')
        end_time_trial = time.time()

        # GET THE BEST INDIVIDUAL
        best_individual = max(population, key=lambda x: x.fitness)
        
        best_individual_each_trial.append(best_individual)
        time_spent_each_trial.append(end_time_trial-init_time_trial)
        best_values_each_trial.append(best_individual.fitness)
        mean_values_each_trial.append(mean_fitness)

        # Insert values into df
        df_fitness = insert_values(df_fitness, best_values, trial_number)
        df_mean = insert_values(df_mean, mean_values, trial_number)

        # SAVE TRIAL RESULTS
        df_fitness.to_csv(f'results_{params["DIM"]}_queens/fitness_trial_{trial_number}.csv')
        df_mean.to_csv(f'results_{params["DIM"]}_queens/mean_trial_{trial_number}.csv')

    
    Parallel(n_jobs=-1)(delayed(run_trial)(trial) for trial in range(0, params["N_TESTS"]))
    # for trial in range(0, params["N_TESTS"]):
    #     run_trial(trial)

    # GET RESULTS FOR EVERY TRIAL
    dfss_fitness = []
    dfss_mean = []
    for trial_number in range(0, params["N_TESTS"]):
        df_fitness_trial = pd.read_csv(f'results_{params["DIM"]}_queens/fitness_trial_{trial_number}.csv', index_col=0)
        df_mean_trial = pd.read_csv(f'results_{params["DIM"]}_queens/mean_trial_{trial_number}.csv', index_col=0)
        dfss_fitness.append(df_fitness_trial)
        dfss_mean.append(df_mean_trial)

    max_gens = max([df.shape[0] for df in dfss_fitness])
    max_means = max([df.shape[0] for df in dfss_mean])


    # Expand dfs to fill
    all_dfs_fitness = [expand_df(df_fitness, max_gens) for df_fitness in dfss_fitness]
    all_dfs_means = [expand_df(df_means, max_gens) for df_means in dfss_mean]


    df_fitness_final = pd.concat(all_dfs_fitness, axis=1)
    df_means_final = pd.concat(all_dfs_means, axis=1)


    df_fitness_final.to_csv(f"results_{params['DIM']}_queens/df_fitness.csv")
    df_means_final.to_csv(f"results_{params['DIM']}_queens/df_mean.csv")

    # GET BEST INDIVIDUAL OVER ALL TRIALS
    best_individual_all = max(best_individual_each_trial, key=lambda x: x.fitness)

    save_trial_results(list_to_save=best_values_each_trial, label=params["DIM"], file_name='fitness')
    save_trial_results(list_to_save=mean_values_each_trial, label=params["DIM"], file_name='mean')
    save_trial_results(list_to_save=time_spent_each_trial, label=params["DIM"], file_name='time')
    save_any_result(text=str_individual_plus_board(ind=best_individual_all, board=board), label=params["DIM"], file_name='best')


    end = time.time()

    print(f'Total time spent: {end - init}s')