import os
import sys
import copy
import random
import numpy as np
import pickle
from ga_carla.Chromosome import Chromosome
from datetime import datetime
import util
import tools
import generateRestart

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DRIVEFUZZ_SRC = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, "src"))
if _DRIVEFUZZ_SRC not in sys.path:
    sys.path.insert(0, _DRIVEFUZZ_SRC)
import config as _drivefuzz_config
_drivefuzz_config.set_carla_api_path()
import carla

class GeneticAlgorithm:
    def __init__(self, bounds, pm, pc, pop_size, NPC_size, time_size, conflict_t, conflict_d, period, max_gen, is_exploit):
        self.bounds = bounds                # The value ranges of the inner most elements
        self.pm = pm
        self.pc = pc
        self.pop_size = pop_size            # Number of scenarios in the population
        self.NPC_size = NPC_size            # Number of NPC in each scenario
        self.time_size = time_size          # Number of time slides in each NPC
        self.conflict_t = conflict_t
        self.conflict_d = conflict_d
        self.period = period
        self.max_gen = max_gen
        self.pop = []
        self.bests = [0] * max_gen          # Best scenario in each generation
        self.bestIndex = 0
        self.g_best = None                  # Best scenario across all generations
        self.ck_path = None                 # Checkpoint path, if set, GE will start from the checkpoint (population object)
        self.touched_chs = []               # Record which chromosomes have been touched in each generation

        self.isInLis = False                # Set flag for local iterative search (LIS)
        self.minLisGen = 2                  # Min gen to start LIS
        self.numOfGenInLis = 5              # Number of gens in LIS
        self.hasRestarted = False
        self.lastRestartGen = 0
        self.bestYAfterRestart = 0
        self.count_dict = []
        self.is_exploit = is_exploit

        self._map = None

    @property
    def map(self):
        if self._map is None:
            client = carla.Client('localhost', 2000)
            client.set_timeout(20.0)
            world = client.get_world()
            self._map = world.get_map()
            util.print_debug("[GA] lazy-fetched map: {}".format(
                world.get_map().name))
        return self._map

    def set_checkpoint(self, ck_path):
        self.ck_path = ck_path

    def take_checkpoint(self, obj, ck_name):
        if os.path.exists('GaCheckpoints') == False:
            os.mkdir('GaCheckpoints')
        ck_f = open('GaCheckpoints/' + ck_name, 'wb')
        pickle.dump(obj, ck_f)
        ck_f.truncate() 
        ck_f.close()
   
    def setLisFlag(self):
        self.isInLis = True

    def setLisPop(self, singleChs):
        for i in range(self.pop_size):
            self.pop.append(copy.deepcopy(singleChs))

        # Add some entropy
        tempPm = self.pm
        self.pm = 1
        self.mutation_fuzz()
        self.pm = tempPm
        self.g_best, bestIndex = self.find_best()

    def ga(self):

        if self.ck_path != None:
            ck = open(self.ck_path, 'rb')
            self.pop = pickle.load(ck)
            ck.close()
        elif self.isInLis == False:
            self.init_pop()
        best, bestIndex = self.find_best()
        self.g_best = copy.deepcopy(best)

        for i in range(self.max_gen):

            util.print_debug(" \n\n*** " + str(i) + "th generation ***")

            # Make sure we clear touched_chs history book every gen
            self.touched_chs = []
            if self.isInLis == False:
                self.mutation_search()
            else:
                self.mutation_fuzz()
            self.cross()
            self.simulate_touched(i)
            self.select_roulette()                             # Selection scenarios for the next generation

            best, bestIndex = self.find_best()                     # Find the scenario with the best fitness score in current generation 
            self.bests[i] = best                        # Record the scenario with the best fitness score in i th generation

            ########### Update noprogressCounter #########
            noprogress = False  # Flag to indicate whether there is no progress in the last 5 generations
            ave = 0  # Average of the best 5 scenarios
            if i >= self.lastRestartGen + 5:  
                for j in range(i - 5, i):
                    ave +=  self.bests[j].y
                ave /= 5
                if ave >= best.y:
                    self.lastRestartGen = i
                    noprogress = True

            util.print_debug(" ###### Best score of the generation: " + str(best.y) + " ###### ")
            if self.g_best.y < best.y:                  # Record the best fitness score across all generations
                self.g_best = copy.deepcopy(best)

            util.print_debug(" ###### Best score overall: " + str(self.g_best.y) + " ###### ")

            N_generation = self.pop
            N_b = self.g_best                           # Record the scenario with the best score over all generations

            self.take_checkpoint(N_b, 'best_scenario.obj')                       

            self.take_checkpoint(N_generation, 'last_gen.obj')

            now = datetime.now()
            date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
            self.take_checkpoint(N_generation, 'generation-' + str(i) + '-at-' + date_time)

            #################### Start the Restart Process ################### 
            if noprogress == True and not self.isInLis:
                util.print_debug(" ###### Restart Based on Generation: " + str(i) + " ###### ")
                oldCkName = 'GaCheckpoints'
                newPop = generateRestart.generateRestart(oldCkName, 1000, self.bounds, self.conflict_d, self.conflict_t, self.period, self.is_exploit)
                self.pop = copy.deepcopy(newPop)
                self.hasRestarted = True
                best, self.bestIndex = self.find_best()
                self.bestYAfterRestart = best.y
                self.lastRestartGen = i
            #################### End the Restart Process ################### 

            if os.path.exists("GaCheckpoints") == True:
                prePopPool = generateRestart.getAllCheckpoints('GaCheckpoints') 
                simiSum = 0
                for eachChs in self.pop:
                    eachSimilarity =  generateRestart.getSimularityOfScenarioVsPrevPop(eachChs, prePopPool)
                    simiSum += eachSimilarity
                util.print_debug(" ==== Similarity compared with all prior generations: " + str(simiSum/float(self.pop_size)))

            f = open('Progress.log' ,'a')
            f.write(str(i) + " " + str(best.y) + " " + str(self.g_best.y) + " " + str(simiSum/float(self.pop_size)) + " " + date_time + "\n")
            f.close()

            if best.y > self.bestYAfterRestart:
                self.bestYAfterRestart = best.y
                if i > (self.lastRestartGen + self.minLisGen) and self.isInLis == False:
                    ################## Start LIS #################
                    util.print_debug(" \n\n === Start of Local Iterative Search === \n\n")
                    lis = GeneticAlgorithm(self.bounds, 1.0, self.pc, self.pop_size, self.NPC_size, self.time_size, self.conflict_t, self.conflict_d, self.period, self.numOfGenInLis, True)
                    lis.setLisPop(self.g_best)                
                    lis.setLisFlag()
                    lisBestChs = lis.ga()
                    util.print_debug(" --- Best fitness in LIS: " + str(lisBestChs.y))
                    if lisBestChs.y > self.g_best.y:
                        self.pop[bestIndex] = copy.deepcopy(lisBestChs)
                        util.print_debug(" --- Find better scenario in LIS: LIS->" + str(lisBestChs.y) + ", original->" + str(self.g_best.y))
                    else:
                        util.print_debug(" --- LIS does not find any better scenarios")
                    util.print_debug(" \n\n === End of Local Iterative Search === \n\n")
                    ################## End LIS ################ 
        if self.isInLis == False:
            s_f = open('count_dict.obj', 'wb')
            pickle.dump(self.count_dict, s_f)
            s_f.truncate()
            s_f.close()  
            print(self.count_dict)
        return self.g_best

    def init_pop(self):
        i = 0
        regen_attempts = 0
        max_regen_attempts = 50
        while i < self.pop_size:
            chromosome = Chromosome(self.bounds, self.NPC_size, self.time_size,
                                    self.conflict_t, self.conflict_d,
                                    self.period, self.is_exploit)
            chromosome.rand_init()
            chromosome.func()

            n_period = (len(chromosome.period_conflicts)
                        if chromosome.period_conflicts is not None else 0)
            n_potential = (len(chromosome.potential_conflicts)
                           if chromosome.potential_conflicts is not None else 0)

            if chromosome.is_accident:
                util.print_debug(" --- init_pop: chromosome produced an "
                                 "accident (saved to AccidentScenario/), "
                                 "but NOT added to population. Regenerating. ---")
                regen_attempts += 1
            elif n_period == 0 and n_potential == 0:
                util.print_debug(" --- init_pop: regenerate "
                                 "(no conflicts) attempt {}/{} ---"
                                 .format(regen_attempts + 1, max_regen_attempts))
                regen_attempts += 1
            else:
                util.print_debug(" --- init_pop: keep chromosome "
                                 "(period={}, potential={}) ---"
                                 .format(n_period, n_potential))
                self.pop.append(chromosome)
                i += 1
                regen_attempts = 0
                continue

            if regen_attempts >= max_regen_attempts:

                util.print_debug(" --- init_pop: max regen attempts "
                                 "reached, accepting next non-accident "
                                 "chromosome regardless of conflicts ---")
                while True:
                    chromosome = Chromosome(self.bounds, self.NPC_size, self.time_size,
                                            self.conflict_t, self.conflict_d,
                                            self.period, self.is_exploit)
                    chromosome.rand_init()
                    chromosome.func()
                    if not chromosome.is_accident:
                        self.pop.append(chromosome)
                        i += 1
                        regen_attempts = 0
                        break

    def cross(self):

        for i in range(int(self.pop_size / 2.0)):
            if self.pc > random.random():
                i = 0
                j = 0
                while i == j:
                    i = random.randint(0, self.pop_size-1)
                    j = random.randint(0, self.pop_size-1)
                pop_i = self.pop[i]
                pop_j = self.pop[j]


                swap_index = random.randint(0, pop_i.code_x1_length - 1)
                if pop_i.scenario_pos[swap_index][0] == pop_j.scenario[swap_index][0] and pop_i.scenario[swap_index][1] == pop_j.scenario[swap_index][1] and pop_i.scenario[swap_index][2] == pop_j.scenario[swap_index][2]:
                    continue
                
                self.touched_chs.append(self.pop[i])
                self.touched_chs.append(self.pop[j])
                temp = copy.deepcopy(pop_j.scenario[swap_index])
                pop_j.scenario[swap_index] = copy.deepcopy(pop_i.scenario[swap_index])
                pop_i.scenario[swap_index] = temp

                
                temp = copy.deepcopy(pop_j.scenario_pos[swap_index])
                pop_j.scenario_pos[swap_index] = copy.deepcopy(pop_i.scenario_pos[swap_index])
                pop_i.scenario_pos[swap_index] = temp
                

    def isStraight(self, ego_pos, npc_pos, npc_pos_last, ego_time, npc_time):
        if ego_time > npc_time:
            ego_position = self.map.get_waypoint(carla.Location(x=ego_pos[0], y=ego_pos[1], z=ego_pos[2]))
            npc_position = self.map.get_waypoint(carla.Location(x=npc_pos[0], y=npc_pos[1], z=npc_pos[2]))
            if npc_pos_last is not None:
                npc_position_last = self.map.get_waypoint(carla.Location(x=npc_pos_last[0], y=npc_pos_last[1], z=npc_pos_last[2]))
                if npc_position_last.road_id == npc_position.road_id and npc_position_last.lane_id != ego_position.lane_id:
                    return False
            if ego_position.road_id == npc_position.road_id and ego_position.lane_id == npc_position.lane_id:
                return True
            else:
                return False
        else:
            return False
                
    def mutation_fuzz(self):
        i = 0
        while(i<len(self.pop)) :
            eachChs = self.pop[i]
            i += 1
            if len(eachChs.period_conflicts) == 0:
                continue
            if self.pm >= random.random():

                beforeMutation = copy.deepcopy(eachChs)
                min_index = 0
                min_dt = 100000
                for conflict in eachChs.period_conflicts:
                    dt = abs(conflict['npc_time'] - conflict['ego_time'])
                    if min_dt > dt:
                        min_dt = dt
                        min_index = conflict['npc']
                    elif min_dt == dt:
                        if 0.8 < random.random():
                            min_index = conflict['npc']
                if 0.8 < random.random():
                    chosen = None
                    for c in eachChs.period_conflicts:
                        if c['npc'] == min_index:
                            chosen = c
                            break
                    if chosen is None:
                        chosen = random.choice(eachChs.period_conflicts)
                else:
                    chosen = eachChs.period_conflicts[
                        random.randint(0, len(eachChs.period_conflicts) - 1)]
                ego_time = chosen['ego_time']
                npc_index = chosen['npc']
                npc_time = chosen['npc_time']
                ego_pos = eachChs.scenario_pos[0][npc_time] 
                npc_pos = eachChs.scenario_pos[npc_index + 1][npc_time]
                if npc_time - 1 >= 0:
                    npc_pos_last = eachChs.scenario_pos[npc_index + 1][npc_time - 1]
                else:
                    npc_pos_last = None

                if self.isStraight(ego_pos, npc_pos, npc_pos_last, ego_time, npc_time):
                    if ego_time > npc_time:
                        prob = random.randint(0, 1)
                        if prob == 0:  # decelerate
                            count = 0
                            temp = 0
                            for t_s in range(npc_time - self.conflict_t if npc_time - self.conflict_t >= 0 else 0, npc_time):
                                if count == 0:
                                    temp = eachChs.scenario[npc_index][t_s][0]
                                    count += 1
                                v_s = random.uniform(self.bounds[3][0], self.bounds[3][1])
                                
                                if temp - v_s > 0:
                                    temp -= v_s
                                eachChs.scenario[npc_index][t_s][0] = temp
                        else:          # brake
                            count = 0
                            temp = 0
                            for t_s in range(npc_time - self.conflict_t if npc_time - self.conflict_t >= 0 else 0, npc_time):
                                if count == 0:
                                    temp = eachChs.scenario[npc_index][t_s][0]
                                    count += 1
                                v_s = random.uniform(self.bounds[4][0], self.bounds[4][1])
                                if temp - v_s > 0:
                                    temp -= v_s
                                eachChs.scenario[npc_index][t_s][0] = temp
                    
                else:
                    if ego_time > npc_time:
                        prob = random.randint(0, 2)
                        if prob == 0:       # decelerate
                            count = 0
                            temp = 0
                            for t_s in range(npc_time - self.conflict_t if npc_time - self.conflict_t >= 0 else 0, npc_time):
                                if count == 0:
                                    temp = eachChs.scenario[npc_index][t_s][0]
                                    count += 1
                                v_s = random.uniform(self.bounds[3][0], self.bounds[3][1])
                                if temp - v_s > 0:
                                    temp -= v_s
                                eachChs.scenario[npc_index][t_s][0] = temp
                        elif prob == 1:          # brake
                            count = 0
                            temp = 0
                            for t_s in range(npc_time, ego_time):
                                if count == 0:
                                    temp = eachChs.scenario[npc_index][t_s][0]
                                    count += 1
                                v_s = random.uniform(self.bounds[4][0], self.bounds[4][1])
                                if temp - v_s > 0:
                                    temp -= v_s
                                eachChs.scenario[npc_index][t_s][0] = temp
                        else:
                            count = 0
                            temp = 0
                            for t_s in range(npc_time - self.conflict_t if npc_time - self.conflict_t >= 0 else 0, npc_time):
                                if count == 0:
                                    temp = eachChs.scenario[npc_index][t_s][0]
                                    count += 1
                                v_s = random.uniform(self.bounds[3][0], self.bounds[3][1])
                                if temp - v_s > 0:
                                    temp -= v_s
                                eachChs.scenario[npc_index][t_s][0] = temp
                            count = 0
                            temp = 0
                            for t_s in range(npc_time, ego_time):
                                if count == 0:
                                    temp = eachChs.scenario[npc_index][t_s][0]
                                    count += 1
                                v_s = random.uniform(self.bounds[4][0], self.bounds[4][1])
                                if temp - v_s > 0:
                                    temp -= v_s
                                eachChs.scenario[npc_index][t_s][0] = temp
                    elif ego_time < npc_time:
                        count = 0
                        temp = 0
                        for t_s in range(ego_time - self.conflict_t if ego_time - self.conflict_t >= 0 else 0, ego_time): # accelerate
                            if count == 0:
                                temp = eachChs.scenario[npc_index][t_s][0]
                                count += 1
                            v_s = random.uniform(self.bounds[2][0], self.bounds[2][1])
                            temp += v_s
                            eachChs.scenario[npc_index][t_s][0] = temp
                    
                # Record which chromosomes have been touched
                self.touched_chs.append(eachChs)

        
    def mutation_search(self):
        i = 0
        while(i<len(self.pop)) :
            eachChs = self.pop[i]
            changed = False
            i += 1
            if len(eachChs.potential_conflicts) != 0:
                for conflict in eachChs.potential_conflicts:
                    if self.pm >= random.random():
                        ego_time = conflict['ego_time']
                        npc_index = conflict['npc']
                        npc_time = conflict['npc_time']
                        if npc_time > ego_time:
                            for dt in range(0, ego_time+1):
                                eachChs.scenario[npc_index][dt][0] += 0.5
                                changed = True
                        elif npc_time < ego_time:
                            for dt in range(0, ego_time+1):
                                eachChs.scenario[npc_index][dt][0] -= 0.5
                                changed = True
                        
                    
            for actor in range(0, eachChs.code_x1_length):
                if actor not in eachChs.saved_c_npcs and actor not in eachChs.saved_p_npcs:
                    if self.pm >= random.random():
                        changed = True
                        time_index = random.randint(0, eachChs.code_x2_length-1)
                        actionIndex = random.randint(0, 2)
                        if actionIndex == 0:
                            # Change Speed
                            eachChs.scenario[actor][time_index][0] = random.uniform(self.bounds[0][0], self.bounds[0][1])
                        elif actionIndex == 1:
                            # Change direction
                            eachChs.scenario[actor][time_index][1] = random.randrange(self.bounds[1][0], self.bounds[1][1])
                    
            # Record which chromosomes have been touched
            if changed == True:
                self.touched_chs.append(eachChs)


    def simulate_touched(self, gen):
        i = 0
        while(i < len(self.pop)):
            eachChs = self.pop[i]
            i += 1
            if eachChs in self.touched_chs:
                try:
                    eachChs.func(gen, self.isInLis)
                except Exception as e:
                    util.print_debug(" ***** func() crashed: {}; marking "
                                     "chromosome as zero-fitness *****".format(e))
                    eachChs.is_accident = False
                    eachChs.period_conflicts = []
                    eachChs.saved_c_npcs = []
                    eachChs.potential_conflicts = []
                    eachChs.saved_p_npcs = []
                    eachChs.y = 0
                if self.isInLis == False:
                    self.count_dict.append([len(eachChs.potential_conflicts), len(eachChs.period_conflicts)])
            else:
                util.print_debug(" --- The chromosome has not been touched in this generation, skip simulation. ---")
            util.print_debug(" --- In mutation: Current scenario has y = " + str(eachChs.y))


    def select_top2(self):

        util.print_debug(" +++ Before select() +++ ")
        for i in range(0, self.pop_size):
            util.print_debug(" === Fitness result of the scenario is " + str(self.pop[i].y) + " === ")
    
        maxFitness = 0
        v = []
        for i in range(0, self.pop_size):
            if self.pop[i].y > maxFitness:
                maxFitness = self.pop[i].y

        for i in range(0, self.pop_size):
            if self.pop[i].y == maxFitness:
                for j in range(int(self.pop_size / 2.0)):
                    selectedChromosome = Chromosome(self.bounds, self.NPC_size, self.time_size, self.conflict_t, self.conflict_d, self.period, self.is_exploit)
                    selectedChromosome.scenario = self.pop[i].scenario
                    selectedChromosome.y = self.pop[i].y
                    v.append(selectedChromosome)
                break

        max2Fitness = 0
        for i in range(0, self.pop_size):
            if self.pop[i].y > max2Fitness and self.pop[i].y != maxFitness:
                max2Fitness = self.pop[i].y

        for i in range(0, self.pop_size):
            if self.pop[i].y == max2Fitness:
                for j in range(int(self.pop_size / 2.0)):
                    selectedChromosome = Chromosome(self.bounds, self.NPC_size, self.time_size, self.conflict_t, self.conflict_d, self.period, self.is_exploit)
                    selectedChromosome.scenario = self.pop[i].scenario
                    selectedChromosome.y = self.pop[i].y
                    v.append(selectedChromosome)
                break

        self.pop = copy.deepcopy(v)
        util.print_debug(" +++ After select() +++ ")
        for i in range(0, self.pop_size):
            util.print_debug(" === Fitness result of the scenario is " + str(self.pop[i].y) + " === ")


    def select_roulette(self):

        sum_f = 0

        util.print_debug(" +++ Before select() +++ ")
        for i in range(0, self.pop_size):  
            if self.pop[i].y == 0:  # If the fitness score is 0, set it to a small value  y:fitness score
                self.pop[i].y = 0.001
            util.print_debug(" === Fitness result of the scenario is " + str(self.pop[i].y) + " === ")

        min = self.pop[0].y
        for k in range(0, self.pop_size):
            if self.pop[k].y < min:
                min = self.pop[k].y
        if min < 0: 
            for l in range(0, self.pop_size):
                self.pop[l].y = self.pop[l].y + (-1) * min 

        # roulette
        for i in range(0, self.pop_size):
            sum_f += self.pop[i].y
        p = [0] * self.pop_size
        for i in range(0, self.pop_size):
            if sum_f == 0:
                sum_f = 1
            p[i] = self.pop[i].y / sum_f
        q = [0] * self.pop_size
        q[0] = 0
        for i in range(0, self.pop_size):
            s = 0
            for j in range(0, i+1):
                s += p[j]                    
            q[i] = s

        # start roulette
        v = []
        for i in range(0, self.pop_size):
            r = random.random()
            if r < q[0]:
                selectedChromosome = Chromosome(self.bounds, self.NPC_size, self.time_size, self.conflict_t, self.conflict_d, self.period, self.is_exploit)
                selectedChromosome.scenario = self.pop[0].scenario
                selectedChromosome.y = self.pop[0].y
                selectedChromosome.is_accident = False
                selectedChromosome.scenario_pos = self.pop[0].scenario_pos
                selectedChromosome.period_conflicts = self.pop[0].period_conflicts
                selectedChromosome.saved_c_npcs = self.pop[0].saved_c_npcs
                selectedChromosome.potential_conflicts = self.pop[0].potential_conflicts
                selectedChromosome.saved_p_npcs = self.pop[0].saved_p_npcs
                selectedChromosome.routes = self.pop[0].routes
                v.append(selectedChromosome)
            for j in range(1, self.pop_size):
                if q[j - 1] < r <= q[j]:
                    selectedChromosome = Chromosome(self.bounds, self.NPC_size, self.time_size, self.conflict_t, self.conflict_d, self.period, self.is_exploit)
                    selectedChromosome.scenario = self.pop[j].scenario
                    selectedChromosome.y = self.pop[j].y
                    selectedChromosome.is_accident = False
                    selectedChromosome.scenario_pos = self.pop[j].scenario_pos
                    selectedChromosome.period_conflicts = self.pop[j].period_conflicts
                    selectedChromosome.saved_c_npcs = self.pop[j].saved_c_npcs
                    selectedChromosome.potential_conflicts = self.pop[j].potential_conflicts
                    selectedChromosome.saved_p_npcs = self.pop[j].saved_p_npcs
                    selectedChromosome.routes = self.pop[j].routes
                    v.append(selectedChromosome)
        self.pop = copy.deepcopy(v)

        util.print_debug(" +++ After select() +++ ")
        for i in range(0, self.pop_size):
            util.print_debug(" === Fitness result of the scenario is " + str(self.pop[i].y) + " === ")

    def find_best(self):
        best = copy.deepcopy(self.pop[0])
        bestIndex = 0
        for i in range(self.pop_size):
            if best.y < self.pop[i].y:
                best = copy.deepcopy(self.pop[i])
                bestIndex = i
        return best, bestIndex
        
    def __inverse(self, i):
        r = '1'
        if i == '1':	
            r = '0'

        return r

if __name__ == '__main__':
    bounds = [[5, 54], [0, 3], [0, 2], [0, 2], [2, 4]]
    algorithm = GeneticAlgorithm(bounds, 0.4, 0.0, 4, 20, 30, 4, 3, 3, 50, False)
    algorithm.ga()
    pass
