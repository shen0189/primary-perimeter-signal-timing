import os, sys, copy

import numpy as np
from numpy.core.fromnumeric import mean, sort
from utils.utilize import config
from collections import deque

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

from metric.trafficmetrics import TrafficMetrics

class TrafficSignalController:
    """Abstract base class for all traffic signal controller.

    Build your own traffic signal controller by implementing the follow methods.
    """
    def __init__(self, tsc_id, junc_id, mode, netdata):
        self.id = tsc_id
        self.junc_id = junc_id
        # self.netdata = netdata
        self.node = netdata['node'][self.junc_id]
        self.incoming_edges = netdata['node'][self.junc_id]['incoming']
        self._get_incoming_outgoing_lanes(netdata)
        self._get_incoming_edge_length_max(netdata)
        self.yellow_t = config['yellow_duration']
        self.lower_mode = config['lower_mode']


        ## lane capacity is the lane length divided by the average vehicle length+stopped headway
        self.lane_capacity = np.array([float(netdata['lane'][lane]['length'])/7.5 for lane in self.incoming_lanes])
        
        #for collecting various traffic metrics at the intersection
        #can be extended in trafficmetric.py class to collect new metrics
        # self.metric_args = ['queue', 'delay']
        # self.trafficmetrics = TrafficMetrics(tsc_id, self.incoming_lanes, netdata, self.metric_args, mode)

        # ## 
        # self.cur_phase = []
        # self.prev_phase = []
        # ##
        # self.ep_rewards = []
        # # self.ep_delay = []

        # self.ep_delay_edge = []
        

## get network
    def _get_incoming_outgoing_lanes(self, netdata):
        ''' obtain the incoming and outgoing lanes/edges of the tlc with order
            incoming order: topological order
            outgoing order: number incresing order
        '''

        ## 1.  incoming edge/lane
        incoming_lanes = []
        incoming_edges = []
        outgoing_lanes = {}
        
        # get all movements of each light [lane_in, lane_out, pass_lane]
        moves = traci.trafficlight.getControlledLinks(self.id)
        for move in moves:
            if move:
                inlane, outlane, _ = move[0]
                ## get inlane
                if not incoming_lanes or inlane != incoming_lanes[-1]:
                    incoming_lanes.append(inlane)
                    outgoing_lanes[inlane] = []

                ## get inedge
                edge = inlane.split('_')[0]
                if not incoming_edges or edge != incoming_edges[-1]:
                    incoming_edges.append(edge)

                ## get outlane
                outgoing_lanes[inlane].append(outlane)
            
        self.incoming_edges = incoming_edges
        self.incoming_lanes = incoming_lanes 
        self.outgoing_lanes = outgoing_lanes

            
    
        ## 2. speed limits of in_lane, in_edge
        self.inlane_speed = {} ## inlane
        for inlane in self.incoming_lanes:
            self.inlane_speed[inlane] = netdata['lane'][inlane]['speed']

        self.inedge_speed = {} ## inedge
        for inedge in self.incoming_edges:
            self.inedge_speed[inedge] = netdata['edge'][inedge]['speed']

        ## 3. flow capacity coeficient
        inlane_coefficient = []
        for inlane in self.incoming_lanes:
            if 'l' in netdata['lane'][inlane]['movement'] \
                or 't' in netdata['lane'][inlane]['movement'] :
                inlane_coefficient.append(1.5)
            else:
                inlane_coefficient.append(1)

        self.inlane_coefficient = np.array(inlane_coefficient)[:,np.newaxis].T

    def _get_incoming_edge_length_max(self, netdata):
        ''' get the maximum length of the incoming edge/lane of the junction
        '''
        length = []
        for edge in self.incoming_edges:
            length.append(netdata['edge'][edge]['length'])
        
        self.max_length = max(length)


    #helper functions for rl controllers

    def get_state(self):
        ''' get state of each junction
        '''
        ## get state of inlane and outlane
        ''' OAM State'''
        if self.lower_mode == 'OAM':
            inlane_state = self.get_inlane_state()

            outlane_state = self.get_outlane_state()
            state = np.concatenate([inlane_state, outlane_state], axis=0)


            ''' Max pressure State'''
        elif self.lower_mode == 'MaxPressure':
            state = np.concatenate([self.get_inlane_state(), self.get_outlane_state()], axis=0)
            state = state[0,:] - state[1,:] ## inlane - outlane
            state = np.expand_dims(state, axis=0)

        ''' FixTime '''
        # elif self.lower_mode == 'FixTime':
        #     return


        ## completion of the state
        if state.shape[1]<12:
            complete = np.zeros([state.shape[0], 12-state.shape[1]])
            state = np.concatenate([state, complete], axis=-1)
        return state.T

    def get_inlane_state(self):
        ''' inlane state
        '''

        ''' OAM State'''
        if self.lower_mode == 'OAM':
            inlane_occupancy = []
            inlane_speed = []
            inlane_halting_veh = []
            inlane_veh_num = []
            
            for in_lane in self.incoming_lanes:
                inlane_occupancy.append(traci.lane.getLastStepOccupancy(in_lane))
                inlane_speed.append(traci.lane.getLastStepMeanSpeed(in_lane) / 13.89)
                inlane_halting_veh.append(traci.lane.getLastStepHaltingNumber(in_lane)/50)
                inlane_veh_num.append(traci.lane.getLastStepVehicleNumber(in_lane)/50)
            inlane_state = np.vstack([inlane_occupancy, inlane_speed, inlane_halting_veh,inlane_veh_num])
        
        ''' Max pressure State'''
        if self.lower_mode == 'MaxPressure':
            inlane_halting_veh = []
            for in_lane in self.incoming_lanes:
                hal_veh = traci.lane.getLastStepHaltingNumber(in_lane)
                # hal_veh = traci.lane.getLastStepVehicleNumber(in_lane)
                if hal_veh>50:
                    hal_veh = hal_veh*1.5
                # hal_veh += traci.lane.getLastStepVehicleNumber(in_lane)*1
                inlane_halting_veh.append(hal_veh)

            inlane_halting_veh = np.multiply(inlane_halting_veh, self.inlane_coefficient)
            inlane_state = np.vstack([inlane_halting_veh])

        return inlane_state

    def get_outlane_state(self):
        ''' outlane state
        '''

        ''' MaxPressure State'''
        if self.lower_mode == 'MaxPressure':
            outlane_halting_veh = []
            
            for in_lane in self.incoming_lanes: # for each inlane
                halting_outlane_match = []
                for out_lane in self.outgoing_lanes[in_lane]: # for all outlanes
                    halt_veh = traci.lane.getLastStepHaltingNumber(out_lane)
                    halt_veh = halt_veh/2 if halt_veh>=50 else 0
                    # halting_outlane_match.append(halt_veh)
                    halting_outlane_match.append(halt_veh)

                ## take average
                outlane_halting_veh.append(mean(halting_outlane_match))
                # outlane_halting_veh.append(max(halting_outlane_match))

            # outlane_state = np.concatenate(outlane_occupancy, outlane_speed)
            outlane_state = np.vstack([outlane_halting_veh])
       
        ''' OAM State'''
        if self.lower_mode == 'OAM':
            outlane_occupancy = []
            outlane_speed = []
            
            for in_lane in self.incoming_lanes: # for each inlane
                occu_outlane_match = []
                speed_outlane_match = []
                for out_lane in self.outgoing_lanes[in_lane]: # for all outlanes
                    occu_outlane_match.append(traci.lane.getLastStepOccupancy(out_lane))
                    speed_outlane_match.append(traci.lane.getLastStepMeanSpeed(out_lane))
                
                ## take average
                outlane_occupancy.append(mean(occu_outlane_match))
                outlane_speed.append(mean(speed_outlane_match))          

            # outlane_state = np.concatenate(outlane_occupancy, outlane_speed)
            outlane_state = np.vstack([outlane_occupancy, outlane_speed])

        return outlane_state




if False:
    def run(self):
        data = self.get_subscription_data()
        self.trafficmetrics.update(data)
        # self.cal_delay()

    def cal_delay(self, lane_delay):
        ''' calculate delay of each junction
        '''
        ## lane level
        # delay = 0
        # for in_lane in self.incoming_lanes:
        #     if lane_data:
        #         if lane_data[in_lane][traci.constants.LAST_STEP_VEHICLE_NUMBER] >0:
        #             per_delay = max((self.inlane_speed[in_lane] - lane_data[in_lane][traci.constants.LAST_STEP_MEAN_SPEED]),0)
        #             delay += per_delay * lane_data[in_lane][traci.constants.LAST_STEP_VEHICLE_NUMBER]

        # delay = 0
        # for in_edge in self.incoming_edges:
        #     # if lane_data:
        #     veh_num = traci.edge.getLastStepVehicleNumber(in_edge)
        #     self.veh_num.append(veh_num)
        #     if veh_num >0:
        #         per_delay = max((self.inedge_speed[in_edge] - traci.edge.getLastStepMeanSpeed(in_edge)),0)
        #         delay += per_delay * veh_num
                    
        # self.ep_delay_edge.append(delay)

        ## edge level
        # delay = 0
        # for in_edge in self.incoming_edges:
        #     if edge_data[in_edge][traci.constants.LAST_STEP_VEHICLE_NUMBER] >0:
        #         per_delay = max((self.inedge_speed[in_edge] - edge_data[in_edge][traci.constants.LAST_STEP_MEAN_SPEED]),0)
        #         delay += per_delay * edge_data[in_edge][traci.constants.LAST_STEP_VEHICLE_NUMBER]
        
        # self.ep_delay_edge.append(delay)

        # self.update(data)
        # self.increment_controller()
        delay = np.zeros(config['max_steps'])
        for d in lane_delay.values():
            delay[d[:,0].astype(int)] += d[:,1]
        
        self.ep_rewards = np.reshape(delay, (-1, config['control_interval'])).sum(axis=-1)
        # print(self.ep_rewards)

    def get_metrics(self):
        ''' get all metrics for the current step 
        '''
        metric = {}
        for m in self.metric_args:
            metric[m] = self.trafficmetrics.get_metric(m)

        return metric

    def get_traffic_metrics_history(self):
        return {m:self.trafficmetrics.get_history(m) for m in self.metric_args} 

    def increment_controller(self):
        if self.phase_time == 0:
            ###get new phase and duration
            next_phase = self.next_phase()
            self.conn.trafficlight.setRedYellowGreenState( self.id, next_phase )
            self.phase = next_phase
            self.phase_time = self.next_phase_duration()
        self.phase_time -= 1

    def get_intermediate_phases(self, phase, next_phase):
        if phase == next_phase or phase == self.all_red:
            return []
        else:
            yellow_phase = ''.join([ p if p == 'r' else 'y' for p in phase ])
            return [yellow_phase, self.all_red]

    def next_phase(self):
        raise NotImplementedError("Subclasses should implement this!")
        
    def next_phase_duration(self):
        raise NotImplementedError("Subclasses should implement this!")

    def update(self, data):
        """Implement this function to perform any
           traffic signal class specific control/updates 
        """
        raise NotImplementedError("Subclasses should implement this!")

    def get_subscription_data(self):
        ''' get the subscription data of the junction
            process the vehs into each in_lane
        '''
        #use SUMO subscription to retrieve vehicle info in batches
        #around the traffic signal controller
        tl_data = traci.junction.getContextSubscriptionResults(self.junc_id)
        # print(tl_data)

        ## create empty incoming lanes for use else where
        lane_vehicles = {l:{} for l in self.incoming_lanes}
        
        ## put vehs on the in_lane
        if tl_data:
            for v in tl_data:
                lane = tl_data[v][traci.constants.VAR_LANE_ID]
                if lane in lane_vehicles:
                    # lane_vehicles[lane] = {}
                    lane_vehicles[lane][v] = tl_data[v] 
        return lane_vehicles

    def get_tl_green_phases(self):
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0]
        #get only the green phases
        green_phases = [ p.state for p in logic.getPhases() 
                         if 'y' not in p.state 
                         and ('G' in p.state or 'g' in p.state) ]

        #sort to ensure parity between sims (for RL actions)
        return sorted(green_phases)
    
    def phase_lanes(self, actions,netdata):
        phase_lanes = {a:[] for a in actions}
        for a in actions:
            green_lanes = set()
            red_lanes = set()
            for s in range(len(a)):
                if a[s] == 'g' or a[s] == 'G':
                    green_lanes.add(netdata['inter'][self.id]['tlsindex'][s])
                elif a[s] == 'r':
                    red_lanes.add(netdata['inter'][self.id]['tlsindex'][s])

            ###some movements are on the same lane, removes duplicate lanes
            pure_green = [l for l in green_lanes if l not in red_lanes]
            if len(pure_green) == 0:
                phase_lanes[a] = list(set(green_lanes))
            else:
                phase_lanes[a] = list(set(pure_green))
        return phase_lanes

    def get_normalized_density(self):
        #number of vehicles in each incoming lane divided by the lane's capacity
        return np.array([len(self.data[lane]) for lane in self.incoming_lanes])/self.lane_capacity

    def get_normalized_queue(self):
        lane_queues = []
        for lane in self.incoming_lanes:
            q = 0
            for v in self.data[lane]:
                if self.data[lane][v][traci.constants.VAR_SPEED] < 0.3:
                    q += 1
            lane_queues.append(q)
        return np.array(lane_queues)/self.lane_capacity

    def empty_intersection(self):
        for lane in self.incoming_lanes:
            if len(self.data[lane]) > 0:
                return False
        return True

    def get_reward(self):
        #return negative delay as reward
        delay = self.ep_delay_edge[-1]
        if delay == 0:
            r = 0
        else:
            r = -delay

        self.ep_rewards.append(r)
        return r

    def empty_dtse(n_lanes, dist, cell_size):
        return np.zeros((n_lanes, int(dist/cell_size)+3 ))
    
    def phase_dtse(phase_lanes, lane_to_int, dtse):
        phase_dtse = {}
        for phase in phase_lanes:
            copy_dtse = np.copy(dtse)
            for lane in phase_lanes[phase]:
                copy_dtse[lane_to_int[lane],:] = 1.0
            phase_dtse[phase] = copy_dtse
        return phase_dtse

    def input_to_one_hot(self, phases):
        identity = np.identity(len(phases))                                 
        one_hots = { phases[i]:identity[i,:]  for i in range(len(phases)) }
        return one_hots

    def int_to_input(self, phases):
        return { p:phases[p] for p in range(len(phases)) }

'''
    def get_dtse():
        dtse = np.copy(self._dtse)
        for lane,i in zip(incoming_lanes, range(len(incoming_lanes))):
            for v in self.data[lane]:
                pos = self.data[lane][v][traci.constants.VAR_LANEPOSITION]
                dtse[i, pos:pos+1] = 1.0

        return dtse
'''


'''
right_on_red_phases = []
for phase in green_phases:
    new_phase = []
    for idx in range(len(phase)):
        if self.netdata['inter'][self.id]['tlsindexdir'][idx] == 'r' and phase[idx] == 'r':
            new_phase.append('s')
        else:
            new_phase.append(phase[idx])
    right_on_red_phases.append(''.join(new_phase))
'''

'''
n_g = len(green_phases)
                                                                                            
right_on_red_phases = []
for phase in green_phases:
    new_phase = []
    for idx in range(len(phase)):
        if self.netdata['inter'][self.id]['tlsindexdir'][idx] == 'r' and phase[idx] == 'r':
            new_phase.append('s')
        else:
            new_phase.append(phase[idx])
    right_on_red_phases.append(''.join(new_phase))
                                                                                            
green_phases = [ p for p in right_on_red_phases 
                 if 'y' not in p
                 and ('G' in p or 'g' in p) ]
'''
'''
n_ror = len(ror_phases)
if n_ror != n_g:
    print('==========')
    print(self.id)
    print(green_phases)
    print(ror_phases)
'''
