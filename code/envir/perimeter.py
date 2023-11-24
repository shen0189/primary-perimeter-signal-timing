import sumolib
import traci
import numpy as np
from utils.utilize import config
from collections import deque

# EdgeCross = [33034, 53054]


class Peri_Agent():
    def __init__(self, tsc_peri):
        # signal constraints
        self.min_green = config['min_green']
        self.max_green = config['max_green']
        self.yellow_duration = config['yellow_duration']
        self.cycle_time = config['cycle_time']
        self.downedge_maxveh = config['max_queue']

        # perimiter infos
        self.info = config['Peri_info']
        self.peri_num = len(self.info)
        self.splitmode = config['splitmode']  # 0 = waittime, 1 = waitveh
        self.peri_action_mode = config['peri_action_mode']
        self.flowrate = config['flow_rate']  # 1 veh /cross time
        self.tsc_peri = tsc_peri
        # list to record datas
        self.green_time = np.array([])
        # self.entered_veh = np.array([])

    def get_waitfeature(self):
        """ get waiting features on the perimeters
        """
        waittime, waitveh = {}, {}
        for tlsID in self.info.keys():

            edge_num = self.info[tlsID]['edge']  # get the controlled edge
            waittime[edge_num] = traci.edge.getWaitingTime(
                str(edge_num))  # waiting times on the edge
            waitveh[edge_num] = traci.edge.getLastStepHaltingNumber(
                str(edge_num))  # waiting vehs

        # print(waittime, waitveh)
        self.wait_feature = (waittime, waitveh
                             )  # store all the dicts into the tuple

        # return wait_feature

    def get_greensplit(self, action, steps):
        """ disctribute the vehicles on the boundary with green split
        """
        green_time = np.zeros(len(self.info))

        if self.peri_action_mode == 'centralize':
            ############ consider upstream waiting features
            if False: 
                # First, get wait feature
                self.get_waitfeature()
                
                # Secend, calcualte green split
                ### waitfeature i\
                # s veh number
                wait_feature = self.wait_feature[self.splitmode]  # choose split mode
                print(f"wating vehicles: {wait_feature}")

                wait_value = np.array(list(
                    wait_feature.values()))  # get values of each crossedge

                wait_value = np.maximum(wait_value, self.min_green / self.flowrate)  # minimum green constraint, in case of NaN
                # print(wait_value)
                ratio = wait_value / sum(
                    wait_value)  # split ratio of the permitted vehs
                # print(ratio)
                permit_vehnum = action * ratio  # calculate the permit vehs on each perimeter
                # print(permit_vehnum)
                green_time = (
                    permit_vehnum *
                    self.flowrate).round()  # transfer: veh num --> green time
            
            ############# directly split green time given total inflow vehicles 
            elif False:
                green_time[:] = (action * self.flowrate / len(self.info)).round()
            
            ############# directly split green time given total green time        
            else:
                green_time[:] = round((action / len(self.info)), 0)

        else:  # decentralized
            green_time[:] = np.around(action)

        # print(green_time)
        green_time = np.minimum(green_time, self.max_green)
        green_time = np.maximum(green_time, self.min_green)
        # green_time = np.maximum(green_time, 0)

        red_time = self.cycle_time - green_time - 2* self.yellow_duration

        self.green_time = np.reshape(np.append(self.green_time, green_time), [-1, len(self.info)])
        # print(self.green_time)

        green_time = dict(zip(config['Peri_info'].keys(), green_time))
        red_time = dict(zip(config['Peri_info'].keys(), red_time))

        return green_time, red_time
        # print(f"action of greentime :{green_time}")

    def set_program(self, green_time, red_time):
        for peri_id, peri in dict.items(self.info):
            tsc = peri['tsc']
            logic = tsc.logic

            ## set green and red time
            logic.phases[tsc.green_phase_index].duration = green_time[peri_id]
            logic.phases[tsc.red_phase_index].duration = red_time[peri_id]

            ## set program
            traci.trafficlight.setProgramLogic(peri_id, logic) # set the new program
            # print(traci.trafficlight.getAllProgramLogics(peri_id))


    def switch_phase(self, steps):
        """ check signal to switch
        """
        for tlsID in self.phase_swithtime.keys():
            if steps in self.phase_swithtime[tlsID]:
                next_phase = (np.where(self.phase_swithtime[tlsID] == steps)[0]
                              + 1) % (np.shape(self.phase_swithtime[tlsID])[0])
                traci.trafficlight.setPhase(tlsID, next_phase)

            # logic, = traci.trafficlight.getAllProgramLogics(tlsID) # get the current program
            # # print(logic)

            # # modify new program
            # phase_index = config['Peri_info'][tlsID][1]
            # logic.phases[phase_index].duration = green_time[tlsID] # the controlled phase
            # logic.phases[3-phase_index].duration = self.cycle_time - green_time[tlsID] - 2 * self.yellow_duration # the side phase

            # traci.trafficlight.setProgramLogic(tlsID, logic) # set the new program

            # # print(traci.trafficlight.getAllProgramLogics(tlsID))
        # return

    def get_down_edge_veh(self):
        down_edge_veh = []
        for tlsID in self.info.keys():
            downedge = self.info[tlsID]['down_edge']
            # downedge_veh.append(edge_infos[downedge]['total'])
            down_edge_veh.append(traci.edge.getLastStepHaltingNumber(str(downedge)))
        # print(downedge_veh) 

        # print(f'waiting vehicles on the down edge: {down_edge_veh}')
        return down_edge_veh

    def get_down_edge_occupancy(self):
        ''' obtain the down edge occupancy of the perimeters
        '''
        down_edge_occupancy = []

        for tlsID in self.info.keys():
            downedge = self.info[tlsID]['down_edge']
            down_edge_occupancy.append(traci.edge.getLastStepOccupancy(str(downedge)))
        
        self.down_edge_occupancy = down_edge_occupancy

        # return down_edge_occupancy

    def get_buffer_average_occupancy(self):
        ''' obtain the average occupancy of the buffer links
        '''
        self.buffer_average_occupancy = [0] * self.peri_num 

        for idx, tlsID in enumerate(self.info.keys()):
            buffer_occupancy = []
            buffer_edges = self.info[tlsID]['buffer_edges']
            
            for edge in buffer_edges:
                buffer_occupancy.append(traci.edge.getLastStepOccupancy(str(edge)))
        
            self.buffer_average_occupancy[idx] = np.mean(buffer_occupancy)

    def get_buffer_edge_veh(self):
        ''' obtain the waiting vehicles of the buffer links to calculate penalty
        '''
        buffer_edge_wait_veh = []

        for tlsID in sorted(list(self.info.keys())):
            buffer_edges = self.info[tlsID]['buffer_edges']
            for edge in buffer_edges:
            # downedge_veh.append(edge_infos[downedge]['total'])
                buffer_edge_wait_veh.append(traci.edge.getLastStepHaltingNumber(str(edge)))
        
        ## total wait vehs on the buffer
        buffer_wait_veh_tot = sum(buffer_edge_wait_veh)

        ## calculate the difference between t and t+1
        delta_buffer_wait_veh_tot = buffer_wait_veh_tot- self.buffer_wait_vehs_tot[-1]  # calculate delta: increment of queue in the buffer

        ## record the queue of current time
        self.buffer_wait_vehs_tot = np.append(self.buffer_wait_vehs_tot, buffer_wait_veh_tot) 
        
        # print(f'waiting vehicles on the buffer edge: {buffer_wait_veh_tot}')
        # print(f'increment waiting vehicles on the buffer edge: {delta_buffer_wait_veh_tot}')

        return delta_buffer_wait_veh_tot, buffer_wait_veh_tot