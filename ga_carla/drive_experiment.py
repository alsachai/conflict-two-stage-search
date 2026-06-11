import ga_carla.GeneticAlgorithm as GeneticAlgorithm

bounds = [[5, 54],
          [0, 3],
          [0, 2],
          [0, 2],
          [2, 4]]

mutation_prob = 1.0
crossover_prob = 0.0
pop_size       = 4
num_npc        = 20
num_time_slice = 30      # 30 seconds per scenario
conflict_t     = 4       # critical conflict horizon (slices)
conflict_d     = 3       # spatial conflict threshold (meters)
period         = 3       # period length for findConflicts() segmentation
max_gen        = 50      # outer-loop generations
is_exploit     = False


if __name__ == "__main__":
    ge = GeneticAlgorithm.GeneticAlgorithm(
        bounds, mutation_prob, crossover_prob,
        pop_size, num_npc, num_time_slice,
        conflict_t, conflict_d, period,
        max_gen, is_exploit)
    # ge.set_checkpoint('GaCheckpoints/last_gen.obj')   # uncomment to resume
    ge.ga()
