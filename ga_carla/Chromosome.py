import math
import random
import pprint
import pickle
import sys
import math
import os
from datetime import datetime
import numpy as np
import util

ROUTE_ACTIONS = ["S", "L", "R"]
ROUTE_WEIGHTS = [0.6, 0.2, 0.2]
ROUTE_LENGTH = 8


class Chromosome:
    def __init__(self, bounds, NPC_size, time_size, conflict_t, conflict_d, period, is_exploit):
        self.y = 0
        self.scenario = [[[] for i in range(time_size)] for j in range(NPC_size)] # start with npc, scenario_pos starts with ego
        self.scenario_pos = None
        self.period_conflicts = None
        self.saved_c_npcs = None
        self.potential_conflicts = None
        self.saved_p_npcs = None
        self.bounds = bounds
        self.code_x1_length = NPC_size 
        self.code_x2_length = time_size
        self.conflict_t = conflict_t
        self.conflict_d = conflict_d
        self.period = period
        self.is_init = True
        self.timeoutTime = 300 
        self.is_accident = False
        self.is_exploit = is_exploit
        self.routes = [[] for _ in range(NPC_size)]

    def fix_init(self):
        for i in range(self.code_x1_length):        # For every NPC
            for j in range(self.code_x2_length):    # For every time slice
                v = (self.bounds[0][0] + self.bounds[0][1]) / float(2)
                a = 3 
                self.scenario[i][j].append(v)
                self.scenario[i][j].append(a)
        for i in range(self.code_x1_length):
            self.routes[i] = ["S"] * ROUTE_LENGTH

    def rand_init(self):
        self.is_init = True
        for i in range(self.code_x1_length):
            self.routes[i] = random.choices(
                ROUTE_ACTIONS, weights=ROUTE_WEIGHTS, k=ROUTE_LENGTH)
            for j in range(self.code_x2_length):    # For every time slice
                v = random.uniform(self.bounds[0][0], self.bounds[0][1])        # Init velocity
                a = random.randrange(self.bounds[1][0], self.bounds[1][1])      # Init action
                self.scenario[i][j].append(v)
                self.scenario[i][j].append(a)
                self.scenario[i][j].append(0) # x
                self.scenario[i][j].append(0) # y
                self.scenario[i][j].append(0) # z
                if j == 0:
                    self.scenario[i][j].append(0) # roll
                    self.scenario[i][j].append(0) # pitch
                    self.scenario[i][j].append(0) # yaw

    def foo_obj_func(self):
        speedSum = 0
        for npc in self.scenario:
            for nt in npc:
                speedSum += nt[0]
                speedSum += nt[0] * 34
        return speedSum

    def decoding(self):
        fitness_score = 0

        s_f = open('scenario.obj', 'wb')
        pickle.dump(self.scenario, s_f)
        s_f.truncate()
        s_f.close()

        r_f = open('routes.obj', 'wb')
        pickle.dump(self.routes, r_f)
        r_f.truncate()
        r_f.close()

        MAX_ERROR_RETRIES = 3
        error_retries = 0

        for x in range(0, 100):	

            if os.path.isfile('result.obj') == True:
                os.remove("result.obj")

            os.system("python3 simulation_drivefuzz.py scenario.obj result.obj "
                      "ego_path.obj npc_spawn.obj routes.obj {} {} {} Town03".format(
                          self.code_x1_length, self.code_x2_length, self.is_init))
            resultObj = None

            if os.path.isfile('result.obj') == True:
                f_f = open('result.obj', 'rb')
                resultObj = pickle.load(f_f)
                f_f.close()

            if resultObj != None:
                pos_ok = resultObj.get('pos') is not None
                error_set = resultObj.get('error') is not None
                if pos_ok and not error_set:
                    return resultObj 
                error_retries += 1
                err_msg = resultObj.get('error', 'no-pos')
                util.print_debug(" ***** scenario failed (error={}); "
                                 "retry {}/{} ***** ".format(
                                     err_msg, error_retries, MAX_ERROR_RETRIES))
                if error_retries >= MAX_ERROR_RETRIES:
                    util.print_debug(" ***** error retries exhausted, "
                                     "giving up on this scenario ***** ")
                    return resultObj
                continue
            else:
                util.print_debug(" ***** " + str(x) + "th/10 trial: Fail to get fitness, try again ***** ")

        return None

    # Get fitness score of the scenario
    def func(self, gen=None, lisFlag=False):
        resultObj = self.decoding()
        self.is_init = False

        if (resultObj is None or
                resultObj.get('pos') is None or
                resultObj.get('error') is not None):
            err = resultObj.get('error', 'unknown') if resultObj else 'no result'
            util.print_debug(" ***** Scenario failed at simulator level: {}; "
                             "marking chromosome zero-fitness ***** ".format(err))
            self.is_accident = False
            self.period_conflicts = []
            self.saved_c_npcs = []
            self.potential_conflicts = []
            self.saved_p_npcs = []
            self.y = 0
            return

        if resultObj['fault'] == 'ego':      
                util.print_debug(" ***** Found an accident where ego is at fault ***** ")
                self.is_accident = True
                if os.path.exists('AccidentScenario') == False:
                    os.mkdir('AccidentScenario')
                date_time = resultObj['time']
                kind = resultObj.get('collision_kind') or 'unknown'
                ckName = 'AccidentScenario/accident-{}-gen{}-{}'.format(
                    kind, gen, date_time)
                if lisFlag == True:
                    ckName = ckName + "-LIS"
                a_f = open(ckName, 'wb')
                pickle.dump(self, a_f)
                a_f.truncate() 
                a_f.close()
        else:
            self.is_accident = False
            get_pos = resultObj['pos']
            for i in range(self.code_x1_length):        # For every NPC
                for j in range(self.code_x2_length):    # For every time slice
                    self.scenario[i][j][2] = get_pos[i+1][j][0]
                    self.scenario[i][j][3] = get_pos[i+1][j][1]
                    self.scenario[i][j][4] = get_pos[i+1][j][2]
                    if j == 0:
                        self.scenario[i][0][5] = get_pos[i+1][0][3]
                        self.scenario[i][0][6] = get_pos[i+1][0][4]
                        self.scenario[i][0][7] = get_pos[i+1][0][5]
                        
            self.scenario_pos = [[[] for i in range(self.code_x2_length)] for j in range(self.code_x1_length+1)]
            for i in range(self.code_x1_length + 1):        # For every NPC
                for j in range(self.code_x2_length):    # For every time slice
                    self.scenario_pos[i][j].append(get_pos[i][j][0])
                    self.scenario_pos[i][j].append(get_pos[i][j][1])
                    self.scenario_pos[i][j].append(get_pos[i][j][2])
            self.period_conflicts, self.saved_c_npcs = self.findConflicts()
            self.potential_conflicts, self.saved_p_npcs = self.find_potential()
            if self.is_exploit:
                average_c = sum(conflict['score'] for conflict in self.period_conflicts if conflict is not None)
                max_c = 0
                for conflict in self.period_conflicts:
                    if conflict['score'] > max_c:
                        max_c = conflict['score']
                self.y = average_c + max_c
            else:
                number_c = len(self.period_conflicts)
                self.y = number_c
                
        
    def findConflicts(self):
        num_periods = (self.code_x2_length + self.period - 1) // self.period

        period_conflicts = []
        saved_npcs = []

        for period_idx in range(num_periods):
            start_time = period_idx * self.period
            end_time = min(start_time + self.period, self.code_x2_length)
        
            for t in range(start_time, end_time):
                ego_pos = self.scenario_pos[0][t]
                # conflict_found = False
                for dt in range(0, self.conflict_t + 1):
                    if t - dt >= start_time: 
                        past_pos_index = t - dt - start_time
                        past_positions = [self.scenario_pos[m][t - dt] for m in range(1, self.code_x1_length + 1)]
                        distances = np.linalg.norm(np.array(past_positions) - np.array(ego_pos), axis=1)
                        min_distance_idx = np.argmin(distances)

                        if distances[min_distance_idx] < self.conflict_d:
                            period_conflicts.append({
                                "ego_time": t,
                                "npc_time": t - dt,
                                "npc": min_distance_idx,
                                "distance": distances[min_distance_idx],
                                "score": self.conflict_t - dt
                            })
                            conflict_found = True
                            if min_distance_idx not in saved_npcs:
                                saved_npcs.append(min_distance_idx)
                            # break
                    # Check future positions
                    if t + dt < end_time: 
                        future_pos_index = t + dt - start_time
                        future_positions = [self.scenario_pos[m][t + dt] for m in range(1, self.code_x1_length + 1)]
                        distances = np.linalg.norm(np.array(future_positions) - np.array(ego_pos), axis=1)
                        min_distance_idx = np.argmin(distances)

                        if distances[min_distance_idx] < self.conflict_d:
                            period_conflicts.append({
                                "ego_time": t,
                                "npc_time": t + dt,
                                "npc": min_distance_idx, 
                                "distance": distances[min_distance_idx],
                                "score": self.conflict_t - dt
                            })
                            # conflict_found = True
                            if min_distance_idx not in saved_npcs:
                                saved_npcs.append(min_distance_idx)
                            # break

                # if conflict_found:
                #     break
        if len(period_conflicts) == 0: 
            print("No critical conflict!!")
        else:
            print(len(period_conflicts), " critical conflicts found!!")

        return period_conflicts, saved_npcs


    def find_potential(self):
        potential_conflicts = []
        saved_npcs = []
        
        for t in range(self.code_x2_length):
            ego_pos = self.scenario_pos[0][t]
            for dt in range(self.conflict_t+1, self.conflict_t+7):
                if t - dt >= 0:
                    past_positions = [self.scenario_pos[m][t - dt] for m in range(1, self.code_x1_length + 1)]
                    distances = np.linalg.norm(np.array(past_positions) - np.array(ego_pos), axis=1)
                    min_distance_idx = np.argmin(distances)
                    if distances[min_distance_idx] < self.conflict_d:
                        if min_distance_idx not in saved_npcs:
                            potential_conflicts.append({
                                "ego_time": t,
                                "npc_time": t - dt,
                                "npc": min_distance_idx,
                            })
                            saved_npcs.append(min_distance_idx)
                
                if t + dt < self.code_x2_length:
                    future_positions = [self.scenario_pos[m][t + dt] for m in range(1, self.code_x1_length + 1)]
                    distances = np.linalg.norm(np.array(future_positions) - np.array(ego_pos), axis=1)
                    min_distance_idx = np.argmin(distances)
                    if distances[min_distance_idx] < self.conflict_d:
                        if min_distance_idx not in saved_npcs:
                            potential_conflicts.append({
                                "ego_time": t,
                                "npc_time": t + dt,
                                "npc": min_distance_idx,
                            })
                            saved_npcs.append(min_distance_idx)
        if len(potential_conflicts) == 0:
            print("No noncritical conflicts!!")
        else:
            print(len(potential_conflicts), " noncritical conflicts found!!")
            
        return potential_conflicts, saved_npcs
                

# # Test
# if __name__ == '__main__':
#     a = [[10, 30], [0, 2]]
#     chromosome = Chromosome(a, 5, 10)
#     pprint.pprint(chromosome.scenario)

