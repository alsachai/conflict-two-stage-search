import math
import os
import lgsvl
import time
import json
import numpy as np
import copy

import simulation.utils as util
import simulation.liability as liability
from obswebsocket import obsws, requests

from datetime import datetime
from loguru import logger

EGO_VEHICLE_ID = '2e966a70-4a19-44b5-a5e7-64e00a7bc5de'

class Simulator(object):

    def __init__(self, default_record_folder, target_record_folder, total_sim_time, conflict_t, conflict_d, period, is_exploit, lgsvl_map = 'SanFrancisco_correct', apollo_map = 'SanFrancisco'):
        
        self.default_record_folder = default_record_folder
        self.target_record_folder = target_record_folder
        ################################################################
        self.total_sim_time = total_sim_time
        self.destination = None
        ################################################################
        self.sim = None
        self.data_prime = None
        self.dv = None
        self.lgsvl_map = lgsvl_map
        self.apollo_map = apollo_map
        self.ego = None
        self.mutated_npc_list = [] # The list contains all the npc added
        self.fixed_npc_list = []
        self.yellow_lines = None
        self.cross_lines = None
        self.edge_lines = None

        self.obs = obsws('localhost',4455,'123456')
        self.record_dir = ''
        self.default_dir = os.path.join(self.record_dir,'sample.mkv')
        self.obs.connect()

        self.connect_lgsvl()
        self.load_map(self.lgsvl_map)
        self.isEgoFault = False
        self.isHit = False
        self.maxint = 130
        self.egoFaultDeltaD = 0
        
        self.conflict_t = conflict_t
        self.conflict_d = conflict_d
        self.period = period
        self.is_exploit = is_exploit
        self.replay_list = None

        self.modules = [
            'Localization',
            'Transform',
            'Routing',
            'Prediction',
            'Planning',
            'Control',
            'Storytelling',
        ]
        self.dy_modules = [
            'Recorder',
        ]

    def connect_lgsvl(self):
        try:
            sim = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), 8181) 
            self.sim = sim
        except Exception as e:
            logger.error('Connect LGSVL wrong: ' + '127.0.0.1:8181')
            logger.error(e.message)
        logger.info('Connected LGSVL 127.0.0.1:8181')

    def load_map(self, mapName="SanFrancisco_correct"):
        if self.sim.current_scene == mapName:
           self.sim.reset()
        else:
           self.sim.load(mapName)
        logger.info('Loaded map: ' + mapName)
        
        
    def get_speed(self, vehicle):
        vel = vehicle.state.velocity
        return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    
    
    def load_json(self, json_file):
        self.data_prime = json.load(open(json_file))
        if not self.data_prime.get('environment'):
            self.data_prime['environment'] = dict()
            self.data_prime['environment'] = {
                    'rain': 0,
                    'fog':  0,
                    'wetness':  0,
                    'cloudiness':  0,
                    'damage': 0,
                    'time':0,
                }

    def init_environment(self, json_file):

        self.isEgoFault = False
        self.isHit = False
        self.mutated_npc_list = []
        self.fixed_npc_list = []

        self.load_json(json_file)

        # load ego car
        ego_data = self.data_prime['agents']['ego']
        ego_position = ego_data['position']
        ego_pos_vector = lgsvl.Vector(x=ego_position['x'], y=ego_position['y'], z=ego_position['z'])
        ego_state = lgsvl.AgentState()
        ego_state.transform = self.sim.map_point_on_lane(ego_pos_vector)
        self.ego = self.sim.add_agent(EGO_VEHICLE_ID, lgsvl.AgentType.EGO, ego_state)
        ego_transfrom = ego_state.transform
        self.replay_list[0][0].extend([self.get_speed(self.ego),
                                       ego_transfrom.position.x,
                                       ego_transfrom.position.y,
                                       ego_transfrom.position.z,
                                       ego_transfrom.rotation.x,
                                       ego_transfrom.rotation.y,
                                       ego_transfrom.rotation.z])
        
        ## ego destination
        des_method = ego_data['destination']['method']
        if des_method == 'forward_right':
            des_forward = ego_data['destination']['value']['v1']
            des_right = ego_data['destination']['value']['v2']
            forward = lgsvl.utils.transform_to_forward(ego_state.transform)
            right = lgsvl.utils.transform_to_right(ego_state.transform)
            self.destination = ego_state.position + des_forward * forward + des_right * right
            
            self.replay_list[0][0].extend([forward.x, forward.z])
        elif des_method == 'xyz':
            x = ego_data['destination']['value']['v1']
            y = ego_data['destination']['value']['v2']
            z = ego_data['destination']['value']['v3']
            self.destination = lgsvl.Vector(x, y, z)
        else:
            raise RuntimeError('Unmatched destination method')

        # load mutated npc
        npcs = self.data_prime['agents']['npcs']
        m = 1
        for m_npc in npcs:
            npc_type = m_npc['type']
            npc_goal = m_npc['goal']
            npc_pos_x = m_npc['position']['x']
            npc_pos_y = m_npc['position']['y']
            npc_pos_z = m_npc['position']['z']
            npc_pos = lgsvl.Vector(x=npc_pos_x, y=npc_pos_y, z=npc_pos_z)
            npc_state = lgsvl.AgentState()
            npc_state.transform = self.sim.map_point_on_lane(npc_pos)
            npc = self.sim.add_agent(npc_type, lgsvl.AgentType.NPC, npc_state)
            npc_transform = npc_state.transform
            self.replay_list[m][0].extend([self.get_speed(npc),
                                           npc_transform.position.x,
                                           npc_transform.position.y,
                                           npc_transform.position.z,
                                           npc_transform.rotation.x,
                                           npc_transform.rotation.y,
                                           npc_transform.rotation.z])
            m = m + 1
            if npc_goal == 'fixed':
                self.fixed_npc_list.append(npc)
            elif npc_goal == 'mutated':
                self.mutated_npc_list.append(npc)
            else:
                raise RuntimeError('Wrong npc goal. Only support fixed or mutated.')

        # load environments
        self.sim.weather = lgsvl.WeatherState(
            rain=self.data_prime['environment']['rain'],
            fog=self.data_prime['environment']['fog'],
            wetness=self.data_prime['environment']['wetness'],
            cloudiness=self.data_prime['environment']['cloudiness'],
            damage=self.data_prime['environment']['damage']
        )
        self.sim.set_time_of_day(self.data_prime['environment']['time'])

         # load lines
        # yellow line
        self.yellow_lines = self.data_prime['lines']['yellow_lines']
        self.cross_lines = self.data_prime['lines']['cross_lines']
        self.edge_lines = self.data_prime['lines']['edge_lines']

    def runSimulation(self, scenario_obj, json_file, case_id):
        while(1):
            break_signal = False
                
            #exit_handler()
            now = datetime.now()
            date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
            logger.info(' === Simulation Start:  ['  + date_time + '] ===')
            logger.info(' Initialize Start!')
            
            time_slice_size = len(scenario_obj[0])
            mutated_npc_num = len(scenario_obj)
            
            self.replay_list = [[[] for i in range(int(time_slice_size) + 1)] for j in range(int(mutated_npc_num) + 1)]  # include ego as position 0
                    
                    
            self.load_map()
            self.init_environment(json_file)
            


            assert mutated_npc_num == len(self.mutated_npc_list)

            # simulation info
            simulation_recording = {
                'bbox': {
                    'ego' : self.ego.bounding_box
                },
                'frames': {

                }
            }
            for npc_i in range(mutated_npc_num):
                simulation_recording['bbox']['npc_' + str(npc_i)] = self.mutated_npc_list[npc_i].bounding_box
            
            global collision_info
            global accident_happen
            global time_index
            
            collision_info = None
            accident_happen = False

            def on_collision(agent1, agent2, contact):
                global accident_happen
                global collision_info
                global time_index

                accident_happen = True
                collision_info = {}

                name1 = "STATIC OBSTACLE" if agent1 is None else agent1.name
                name2 = "STATIC OBSTACLE" if agent2 is None else agent2.name
                logger.error(str(name1) + " collided with " + str(name2) + " at " + str(contact))

                agent1_info = [agent1.state, agent1.bounding_box]
                            
                if not agent2:
                    agent2_info = [None, None]
                else:
                    agent2_info = [agent2.state, agent2.bounding_box]
                
                if contact:
                    contact_loc = [contact.x, contact.y, contact.z]
                
                collision_info['time'] = time_index
                collision_info['ego'] = agent1_info
                collision_info['npc'] = agent2_info
                collision_info['contact'] = contact_loc

                self.sim.stop()
            
            # INIT apollo      
            self.ego.connect_bridge(address='127.0.0.1', port=9090) #address, port
            self.ego.on_collision(on_collision)
            
            times = 0
            success = False
            while times < 3:
                try:
                    dv = lgsvl.dreamview.Connection(self.sim, self.ego, os.environ.get("BRIDGE_HOST", "127.0.0.1"))
                    dv.set_hd_map(self.apollo_map)
                    dv.set_vehicle('Lincoln2017MKZ_LGSVL')
                    dv.disable_apollo()
                    dv.setup_apollo(self.destination.x, self.destination.z, self.modules, default_timeout=30)
                    success = True
                    break
                except:
                    logger.warning('Fail to spin up apollo, try again!')
                    times += 1
            if not success:
                raise RuntimeError('Fail to spin up apollo')

            if self.default_record_folder:
                util.disnable_modules(dv, self.dy_modules)
                time.sleep(1)
                util.enable_modules(dv, self.dy_modules)
            
            dv.set_destination(self.destination.x, self.destination.z)
            logger.info(' --- destination: ' + str(self.destination.x) + ',' + str(self.destination.z))
            
            delay_t = 5
            time.sleep(delay_t)

            for npc in self.mutated_npc_list:
                npc.follow_closest_lane(True, 0)

            for npc in self.fixed_npc_list:
                npc.follow_closest_lane(True, 13.4)

            # Frequency of action change of NPCs
            total_sim_time = self.total_sim_time
            action_change_freq = total_sim_time / time_slice_size
            time_index = 0
            
            # record start
            simulation_recording['frames'][time_index] = {
                'ego': self.ego.state
            }

            for npc_i in range(mutated_npc_num):
                simulation_recording['frames'][time_index]['npc_' + str(npc_i)] = self.mutated_npc_list[npc_i].state
            
            print("Start recording.")
            self.obs.call(requests.StartRecord())
            for t in range(0, int(time_slice_size)):
                i = 0
                for npc in self.mutated_npc_list:
                    npc.follow_closest_lane(True, scenario_obj[i][t][0])
                    turn_command = scenario_obj[i][t][1]

                    #<0: no turn; 1: left; 2: right>
                    if turn_command == 1:
                        #direction = "LEFT"
                        npc.change_lane(True)
                        
                    elif turn_command == 2:
                        #direction = "RIGHT"
                        npc.change_lane(False)
                        
                    i += 1        

                for j in range(0, int(action_change_freq) * 10):
                    module_status_mark = True
                    while module_status_mark:
                        module_status_mark = False
                        module_status = dv.get_module_status()
                        for module, status in module_status.items():
                            if (not status) and (module in self.modules):
                                # logger.warning('$$Simulator$$ Module is closed: ' + module + ' ==> restart')
                                # dv.enable_module(module)
                                # time.sleep(0.5)
                                logger.info(module + ' module closed, restart!')
                                break_signal = True
                                module_status_mark = False
                                
                    if break_signal:
                        break
                    
                    time_index += 1

                    self.sim.run(0.1)
                        
                        
                    simulation_recording['frames'][time_index] = {
                        'ego': self.ego.state
                    }

                    for npc_i in range(len(self.mutated_npc_list)):
                        simulation_recording['frames'][time_index]['npc_' + str(npc_i)] = self.mutated_npc_list[npc_i].state
                    
                if break_signal:
                    break
                
                ego_transform = self.ego.state.transform
                forward = lgsvl.utils.transform_to_forward(ego_transform)
                self.replay_list[0][t+1].extend([self.get_speed(self.ego),
                                                 ego_transform.position.x,
                                                 ego_transform.position.y,
                                                 ego_transform.position.z,
                                                 ego_transform.rotation.x,
                                                 ego_transform.rotation.y,
                                                 ego_transform.rotation.z,
                                                 forward.x,
                                                 forward.z])
                for m, npc in enumerate(self.mutated_npc_list, start=1):
                    npc_transform = npc.state.transform
                    self.replay_list[m][t+1].extend([self.get_speed(npc),
                                                     npc_transform.position.x,
                                                     npc_transform.position.y,
                                                     npc_transform.position.z,
                                                     npc_transform.rotation.x,
                                                     npc_transform.rotation.y,
                                                     npc_transform.rotation.z])
                
            
            if break_signal:
                time.sleep(5)
                print("Module down, restart recording.")
                self.obs.call(requests.StopRecord())
                while(True):
                    status = self.obs.call(requests.GetRecordStatus())
                    if not bool(status.getOutputActive()):
                        print("Recording stopped.")
                        break
                    time.sleep(0.05)
                if os.path.exists(self.default_dir):
                    os.remove(self.default_dir)
                else:
                    print('Warning:No default file existed.')
            else:
                time.sleep(5)
                print("Scenario Finished, stop recording.")
                self.obs.call(requests.StopRecord())
                while(True):
                    status = self.obs.call(requests.GetRecordStatus())
                    if not bool(status.getOutputActive()):
                        print("Recording stopped.")
                        break
                    time.sleep(0.05)
                filename = f"scenario_{case_id}.mkv"
                real_dir = os.path.join(self.record_dir,filename)
                if os.path.exists(self.default_dir):
                    os.rename(self.default_dir, real_dir)
                else:
                    print('Warning:No default file existed.')
                break
    
        if self.default_record_folder:
            util.disnable_modules(dv, self.dy_modules)
            time.sleep(0.5)

        # check new folder and move -> save folder
        if self.default_record_folder:
            util.check_rename_record(self.default_record_folder, self.target_record_folder, case_id)

        
        # compute fitness score & check other bugs such as line cross or else
        '''
        

        global collision_info
        global accident_happen
        
        collision_info = None
        accident_happen = False
        
        '''
        # Step 1 obtain time
        simulation_slices = max(simulation_recording['frames'].keys())

        '''
        simulation_recording[time_index] = {
                    'ego': self.ego.transform,
                    'npc': []
                }
        '''
        fault = []
        fitness = -1111
        
        
        period_conflicts, saved_c_npcs = self.findConflicts(self.replay_list, int(time_slice_size), int(mutated_npc_num))
        potential_conflicts, saved_p_npcs = self.find_potential(self.replay_list, int(time_slice_size), int(mutated_npc_num))
        if self.is_exploit:
            if len(period_conflicts) != 0:
                average_c = sum(conflict['score'] for conflict in period_conflicts if conflict is not None) / len(period_conflicts)
            else:
                average_c = 0
            max_c = 0
            for conflict in period_conflicts:
                if conflict['score'] > max_c:
                    max_c = conflict['score']
            fitness = average_c + max_c
        else:
            fitness = len(period_conflicts)
        
        # Step 2 compute distance and check line error and filter npc_fault
        for t in range(simulation_slices):
            simulation_frame = simulation_recording['frames'][t]
            ego_info = {
                'state': simulation_frame['ego'],
                'bbox': simulation_recording['bbox']['ego']
            }            
            # compute distance
            for npc_i in range(len(self.mutated_npc_list)):
                npc_id = 'npc_' + str(npc_i)
                npc_info = {
                    'state': simulation_frame[npc_id],
                    'bbox': simulation_recording['bbox'][npc_id]
                }
            
            # check line
            for yellow_line in self.yellow_lines:
                hit_yellow_line = liability.ego_yellow_line_fault(ego_info, yellow_line)
                if hit_yellow_line:
                    fault.append('hit_yellow_line')
            
            for edge_line in self.edge_lines:
                hit_edge_line = liability.ego_edge_line_fault(ego_info, edge_line)
                if hit_edge_line:
                    fault.append('hit_edge_line')
            
        # Step 3 if collision, check is npc fault
        
        if collision_info is not None:
            ego_info = {
                'state': collision_info['ego'][0],
                'bbox': collision_info['ego'][1]
            }

            npc_info = {
                'state': collision_info['npc'][0],
                'bbox': collision_info['npc'][1]
            }
            
            ego_fault = liability.ego_collision_fault(ego_info, npc_info, self.cross_lines)
            if ego_fault:
                fault.append('ego_fault')
            else:
                fault.append('npc_fault')
                if os.path.exists(real_dir):
                    os.remove(real_dir)
                else:
                    print('Warning:No default file existed.')
        if len(fault) == 0:
            fault.append('normal')
            if os.path.exists(real_dir):
                os.remove(real_dir)
            else:
                print('Warning:No default file existed.')
        
        
        result_dict = {}
        result_dict['fitness'] = fitness
        result_dict['fault'] = fault
        
        logger.info(' === Simulation End === ')

        return self.replay_list, result_dict, period_conflicts, saved_c_npcs, potential_conflicts, saved_p_npcs
    
    
    def findConflicts(self, replay_list, time_len, npc_len):
        num_periods = (time_len + self.period - 1) // self.period

        period_conflicts = []
        saved_npcs = []

        for period_idx in range(num_periods):
            start_time = period_idx * self.period
            end_time = min(start_time + self.period, time_len)
        
            for t in range(start_time, end_time):
                ego_pos = replay_list[0][t][1:4]
                # conflict_found = False
                for dt in range(0, self.conflict_t + 1):
                    # Check past positions
                    if t - dt >= start_time: 
                        past_positions = [replay_list[m][t - dt][1:4] for m in range(1, npc_len + 1)]
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
                        future_positions = [replay_list[m][t + dt][1:4] for m in range(1, npc_len + 1)]
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


    def find_potential(self, replay_list, time_len, npc_len):
        potential_conflicts = []
        saved_npcs = []
        
        for t in range(time_len):
            ego_pos = replay_list[0][t][1:4]
            for dt in range(self.conflict_t+1, self.conflict_t+7):
                if t - dt >= 0:
                    past_positions = [replay_list[m][t - dt][1:4] for m in range(1, npc_len + 1)]
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
                
                if t + dt < time_len:
                    future_positions = [replay_list[m][t + dt][1:4] for m in range(1, npc_len + 1)]
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
      