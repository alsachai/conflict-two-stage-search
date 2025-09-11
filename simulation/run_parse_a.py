import os
import math
import pickle
import shutil

from loguru import logger
from simulation.simulator_a import Simulator

def isNaN(i_):
    return math.isnan(i_)

def rotation(position, heading):
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    x0, y0 = position[0], position[1]
    x1, y1 = x0 * cos_h + y0 * sin_h, y0 * cos_h - x0 * sin_h

    return (x1, y1)

def clear_and_create(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

class Runner(object):

    def __init__(self, scenario_env_json, output_path, total_sim_time, default_record_folder, conflict_t, conflict_d, period, is_exploit, lgsvl_map = 'SanFrancisco_correct', apollo_map = 'SanFrancisco'):
        self.global_id = 0
        self.scenario_env_json = scenario_env_json
        self.scenario_name = os.path.basename(scenario_env_json).split('.')[0]

        self.SCENARIO_FOLDER = 'scenarios'
        self.RESULT_FOLDER = 'results'
        self.RECORD_FOLDER = 'records'
        self.REPLAY_FOLDER = 'replays'

        self.default_record_folder = default_record_folder
        logger.info('Default record path: ' + str(self.default_record_folder))

        self.scenario_path = os.path.join(output_path, 'simulation', self.SCENARIO_FOLDER)
        self.result_path = os.path.join(output_path, 'simulation', self.RESULT_FOLDER)
        self.record_path = os.path.join(output_path, 'simulation', self.RECORD_FOLDER)
        self.replay_path = os.path.join(output_path, 'simulation', self.REPLAY_FOLDER)

        clear_and_create(self.scenario_path)
        clear_and_create(self.result_path)
        clear_and_create(self.record_path)
        clear_and_create(self.replay_path)
        
        self.conflict_t = conflict_t
        self.conflict_d = conflict_d
        self.period = period
        self.is_exploit = is_exploit

        self.sim = Simulator(self.default_record_folder, self.record_path, total_sim_time, conflict_t, conflict_d, period, is_exploit, lgsvl_map, apollo_map, ) # save record to records/scenario_name/scenario_id
        
        self.runner_log = os.path.join(output_path, 'logs/case_states.log')
        if os.path.exists(self.runner_log):
            os.remove(self.runner_log)

    
    def run(self, scenario_data):
        scenario_id = 'scenario_' + str(self.global_id)
        replay_list, sim_result, period_conflicts, saved_c_npcs, potential_conflicts, saved_p_npcs = self._run_scenario(scenario_id, scenario_data)
        if sim_result is None:
            print('sim_result is None, ERROR')
            exit(-1)
        
        sim_fault = sim_result['fault']
        with open(self.runner_log, 'a') as f:
            f.write(str(scenario_id))
            for item in sim_fault:
                f.write(' ')
                f.write(str(item))
            f.write('\n')
        
       
        self.global_id += 1
        logger.info(' === Simulation Result: ' + str(sim_result))
        logger.info(' === Record ' + scenario_id + ' to ' + self.runner_log)
        return float(sim_result['fitness']), scenario_id, replay_list, period_conflicts, saved_c_npcs, potential_conflicts, saved_p_npcs

    def _run_scenario(self, scenario_id, scenario_data):
        """
        run elements:
        save - recording, json config
        save scenario config to scenario_name/scenarios/scenario_id
        save results to scenario_name/results/scenario_id
        """
        scenario_file = os.path.join(self.scenario_path, scenario_id + '.obj')
        with open(scenario_file, 'wb') as s_f:
            pickle.dump(scenario_data, s_f)

        result_file = os.path.join(self.result_path, scenario_id + '.obj')

        if os.path.isfile(result_file):
            os.remove(result_file)

        # replace simulator codes
        resultDic = {}
        try:
            replay_list, resultDic, period_conflicts, saved_c_npcs, potential_conflicts, saved_p_npcs = self.sim.runSimulation(scenario_data, self.scenario_env_json, scenario_id)
        except Exception as e:
            logger.debug(e.message)
            #resultDic['fitness'] = ''
            resultDic['fault'] = ''

        # Send fitness score int object back to ge
        if os.path.isfile(result_file):
            os.system("rm " + result_file)
        with open(result_file, 'wb') as f_f:
            pickle.dump(resultDic, f_f)
            
        replay_file = os.path.join(self.replay_path, scenario_id + '.obj')
        with open(replay_file, 'wb') as s_f:
            pickle.dump(replay_list, s_f)
            
        return replay_list, resultDic, period_conflicts, saved_c_npcs, potential_conflicts, saved_p_npcs
    