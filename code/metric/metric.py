import sumolib
import traci
from utils.utilize import config, plot_MFD
import numpy as np
from envir.perimeter import Peri_Agent
import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt
from collections import deque
import sys, subprocess, os
from itertools import product
import pandas as pd

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    import sumolib
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

class Metric():
    def __init__(self, control_interval, info_interval, netdata):
        assert control_interval%info_interval==0, 'the info_interval in incorrect'
        self.info_length = int(control_interval/info_interval) + 1
        self.netdata = netdata
        self.peri_controlled_links = [peri['edge'] for peri in config['Peri_info'].values()]
        self.sumo_tools_path = tools

        self.edge_buffer = self._get_buffer_edges()
        self.reset()


    def reset(self):
        ## accu of PN
        self.accu_PN_list_epis = [] # each episode 
        # self.accu_PN_list_interval = deque([0], maxlen=self.info_length) # each control interval
        # self.accu_PN_record = [] # each control interval
        
        ## accu of buffer
        self.accu_buffer_list_epis = [] # each episode 
        # self.accu_buffer_list_interval = deque([0], maxlen=self.info_length) # each control interval
        
        ## throuput
        self.throuput_list_epis = []
        self.throuput_list_interval = deque([0], maxlen=self.info_length-1) # summation: each control interval
        # self.throuput_list_interval = [] 
        
        ## PN mean speed
        self.meanspeed_list_epis = []
        # self.meanspeed_list_interval = deque([0], maxlen=self.info_length) # each control interval
        # self.meanspeed_record = [] # each control interval
        
        ## PN production
        self.production_list_epis = []
        # self.production_list_interval = deque([0], maxlen=self.info_length) # each control interval
        
        ## PN halting vehs
        self.halveh_PN_list_epis = []
        # self.halveh_PN_list_interval = deque([0], maxlen=self.info_length) # each control interval
        
        ## buffer halting vehs
        self.halveh_buffer_list_epis = []
        # self.halveh_buffer_list_interval = deque([0], maxlen=self.info_length) # each control interval
        
        ## entered vehs from perimeter
        self.entered_vehs_list_epis = []
        self.entered_vehs_list_interval = deque([0], maxlen=self.info_length) # each control interval
        # self.entered_vehs_list_interval = []
        
        ## edge
        self.edge_info = {
            i: {
                'departed': 0,
                'arrived': 0,
                'entered': 0,
                'left': 0,
                'total': 0
            }
            for i in config['Edge']
        }

    def update_info_interval(self, index, peri_info, ouputdir):
        ''' Update network metrics for each info interval
        '''
        
        ## init
        accu_onestep_PN, accu_onestep_buffer = 0, 0 # accu
        tot_speed = 0    #production
        halt_vehs_PN, halt_vehs_buffer = 0, 0 # halt vehs
        
        ## get data by subscribe
        # for edgeID in self.netdata['edge']:
        #     if int(edgeID) not in config['Edge_Peri']:
        #         if int(edgeID) in config['Edge_PN']: # for edges in PN
        #             accu_edge = edge_data[edgeID][traci.constants.LAST_STEP_VEHICLE_NUMBER]
        #             accu_onestep_PN += accu_edge
        #             halt_vehs_PN += edge_data[edgeID][traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER]
        #             tot_speed += edge_data[edgeID][traci.constants.LAST_STEP_MEAN_SPEED] * accu_edge
        #         else: # for edges in buffer
        #             accu_onestep_buffer += edge_data[edgeID][traci.constants.LAST_STEP_VEHICLE_NUMBER]
        #             halt_vehs_buffer += edge_data[edgeID][traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER]
        
        ## get data by individual traci 
        for edgeID in self.netdata['edge']:
            if int(edgeID) not in config['Edge_Peri']:
                if int(edgeID) in config['Edge_PN']: # for edges in PN
                    accu_edge = traci.edge.getLastStepVehicleNumber(edgeID)
                    accu_onestep_PN += accu_edge
                    halt_vehs_PN += traci.edge.getLastStepHaltingNumber(edgeID)
                    tot_speed += traci.edge.getLastStepMeanSpeed(edgeID) * accu_edge
                else: # for edges in buffer
                    accu_onestep_buffer += traci.edge.getLastStepVehicleNumber(edgeID)
                    halt_vehs_buffer += traci.edge.getLastStepHaltingNumber(edgeID)
        
        ## calculate network mean speed
        if accu_onestep_PN>0:
            network_mean_speed = tot_speed / accu_onestep_PN
        else:
            network_mean_speed = 0

        ## record
        self.meanspeed_list_epis.append(network_mean_speed)
        self.production_list_epis.append(tot_speed)
        self.accu_PN_list_epis.append(accu_onestep_PN)
        self.accu_buffer_list_epis.append(accu_onestep_buffer)
        self.halveh_PN_list_epis.append(halt_vehs_PN)       
        self.halveh_buffer_list_epis.append(halt_vehs_buffer)      
    
    def get_PN_speed_production(self, info_inter_flag=False):
        ''' obtain PN mean_speed & production 
        --- info_inter_flag = True: return the instant value: last value of info interval
            - [instant] the speed & production of info intervals
        --- info_inter_flag = False: return the average value: last value of control interval
            - [Average] the speed & production of info intervals within the control interval
        '''
        if info_inter_flag:

            return self.meanspeed_list_epis[-1], self.production_list_epis[-1]

    def get_accu(self, info_inter_flag=False):
        ''' get the accumulaion of the PN and Buffer
        --- info_inter_flag = True: return the instant value: last value of info interval
        --- info_inter_flag = False: return the average value: last value of control interval
        '''
        if info_inter_flag:
            return self.accu_PN_list_epis[-1], self.accu_buffer_list_epis[-1]

    def get_halting_vehs(self, info_inter_flag=False):
        ''' obtain halting vehs of PN & buffer 
        --- info_inter_flag = True: return the instant value: last value of info interval
            - [instant] halting vehs of PN & buffer of info intervals
        --- info_inter_flag = False: return the average value: last value of control interval
            - [Average] the halting vehs of PN & buffer of info intervals within the control interval
        '''
        if info_inter_flag:
            return self.halveh_buffer_list_epis[-1], self.halveh_PN_list_epis[-1]
    
    def get_PN_throuput_control_interval(self):
        ''' obtain PN throuput within the control interval
            - [Sum up] the throuput of info intervals
        '''
        return self.throuput_list_epis[-1]

    def get_demand_nextstep(self, cycle_index):
        ''' get the demand of the PN in the future steps
        '''
        demand_nextstep = 0
        for DemandType in config['DemandConfig']:
            if DemandType[
                    'DemandType'] != 'outin':  # demand generated inside the region
                # averge demand of future steps
                demand_temp = list(DemandType['Demand'][0]) + [0]
                demand_nextstep += np.mean(
                    demand_temp[(cycle_index +
                                 1):(cycle_index + 1 +
                                     config['Demand_state_steps'])])
        
        if np.isnan(demand_nextstep).any():
            demand_nextstep = 0

        return demand_nextstep

## process vehs
    def process_output(self, ouputdir, index=0):
        ''' read the output file 'edg.xml', obtain the data of each edge in the last simulation step
        '''
        # 1. add the end tag of the xml file, so that it can be opened
        with open(ouputdir, "rb") as file:
            off = -50
            while True:
                file.seek(off, 2) #seek(off, 2)表示文件指针：从文件末尾(2)开始向前50个字符(-50)
                lines = file.readlines() #读取文件指针范围内所有行
                if len(lines)>=2: #判断是否最后至少有两行，这样保证了最后一行是完整的
                    last_line = lines[-1] #取最后一行
                    break
                #如果off为50时得到的readlines只有一行内容，那么不能保证最后一行是完整的
                #所以off翻倍重新运行，直到readlines不止一行
                off *= 2
            last_line = last_line.decode('utf-8')

        if 'meandata' not in last_line: 
            with open(ouputdir, "a") as file:
                print('</meandata>', file=file)
        
        # print(f'###### {ouputdir} #####')
        tree = ET.ElementTree(file=ouputdir)
        roots = tree.getroot()

        # 2. process data for each edge
        for child in roots[index]:  # for each edge
            edge = int(child.attrib['id'])  # get the edge

            # copy the info
            self.edge_info[edge]['departed'] = int(child.attrib['departed'])
            self.edge_info[edge]['arrived'] = int(child.attrib['arrived'])
            self.edge_info[edge]['entered'] = int(child.attrib['entered'])
            self.edge_info[edge]['left'] = int(child.attrib['left'])
            # self.edge_info[edge]['left'] = int(child.attrib['left'])
            # self.edge_info[edge]['left'] = int(child.attrib['left'])
        
        for root in roots: # each interval 
            throuput = 0
            entered_vehs = 0

            for child in root: # each edge
                edge = int(child.attrib['id'])  # get the edge

                # outflow of PN
                if edge in config['Edge_PN_out']:
                    throuput += int(child.attrib['entered'])

                # arrived flow within PN
                if edge in config['Edge_PN']:
                    throuput += int(child.attrib['arrived'])

                ## Peri control links
                if edge in self.peri_controlled_links:
                    entered_vehs += int(child.attrib['left'])
        
            self.throuput_list_interval.append(throuput)   
            self.entered_vehs_list_interval.append(entered_vehs)

        ## sum the episode
        self.throuput_list_epis = np.array(self.throuput_list_interval[::2]) + np.array(self.throuput_list_interval[1::2])
        self.entered_vehs_list_epis = np.array(self.entered_vehs_list_interval[::2]) + np.array(self.entered_vehs_list_interval[1::2])

    def get_edge_cur_veh_num(self):
        ''' calculate the current number of vehcles on each edge
        '''
        for _, edge in self.edge_info.items():
            # veh_num(t+1) = veh_num(t) + depart(t) + entered(t) - arrived(t) -left(t)
            # total_old = edge['total']
            edge['total'] = edge['total'] + edge['departed'] + edge[
                'entered'] - edge['arrived'] - edge['left']

            # print(f"{_}, {total_old}, {edge['total']} \n")
    
    def get_edge_cur_mean_speed(self):
        ''' obtain the mean speed of the edges in the PN
        '''
        for edgeID, edge in self.edge_info.items():
            edge['mean_speed'] = traci.edge.getLastStepMeanSpeed(str(edgeID))


## process output file
    def xml2csv(self, file):
        tool_path = os.path.join(self.sumo_tools_path, 'xml', 'xml2csv.py')
        tool_path_norm = os.path.normpath(tool_path)
        # print("#######################")
        # print(f"old: {tool_path}")
        # print(f"new: {tool_path_norm}")
        # print("#######################")
        cmd =' '.join([f'python "{tool_path_norm}"', file]) 
        # print(cmd)
        os.system(cmd)

    def process_edge_output(self, file):
        ## set file name as csv
        file=file.split('.')
        file = '.'.join([file[0],'csv'])

        ## read csv
        # df = pd.read_csv(file, sep=';')
        df = pd.read_csv(file, sep=';', usecols=\
            ['edge_id','interval_end','edge_sampledSeconds',\
                'edge_density','edge_speed', 'edge_waitingTime', 'edge_left'])

        ## fill nan
        df = df.fillna(0.03)

        ## get PN edges
        df_PN = df[df['edge_id'].isin(config['Edge_PN'])]
        df_buffer = df[df['edge_id'].isin(self.edge_buffer)]
        df_peri = df[df['edge_id'].isin(self.peri_controlled_links)]

        edge_data = {}
        ''' PN data'''
        ## sampledSeconds
        edge_sampledSeconds = df_PN.groupby('interval_end').apply(lambda x: x[['edge_id', 'edge_sampledSeconds']].set_index('edge_id').T)
        edge_sampledSeconds = edge_sampledSeconds.fillna(1e-5)
        edge_data['sampledSeconds'] = edge_sampledSeconds.to_numpy()

        ## density
        edge_density = df_PN.groupby('interval_end').apply(lambda x: x[['edge_id', 'edge_density']].set_index('edge_id').T)
        edge_density = edge_density.fillna(1e-5)
        edge_data['density'] = edge_density.to_numpy()
        
        ## speed
        edge_speed = df_PN.groupby('interval_end').apply(lambda x: x[['edge_id', 'edge_speed']].set_index('edge_id').T)
        edge_speed = edge_speed.fillna(13.89)
        edge_data['speed'] = edge_speed.to_numpy()
        
        ## waiting time
        edge_waitingTime = df_PN.groupby('interval_end').apply(lambda x: x[['edge_id', 'edge_waitingTime']].set_index('edge_id').T)
        edge_waitingTime = edge_waitingTime.fillna(1e-5)
        edge_data['PN_waiting'] = edge_waitingTime.to_numpy()
            
        ''' buffer data'''
        ##  buffer waitingtime
        buffer_waitingTime = df_buffer.groupby('interval_end').apply(lambda x: x[['edge_id', 'edge_waitingTime']].set_index('edge_id').T)
        buffer_waitingTime = buffer_waitingTime.fillna(1e-5)
        edge_data['buffer_waiting'] = buffer_waitingTime.to_numpy()
        
        ## buffer density
        buffer_density = df_buffer.groupby('interval_end').apply(lambda x: x[['edge_id', 'edge_density']].set_index('edge_id').T)
        buffer_density = buffer_density.fillna(1e-5)
        edge_data['buffer_density'] = buffer_density.to_numpy()

        ''' perimeter data'''
        ## get perimeter entered vehs
        peri_entered_vehs = df_peri.groupby('interval_end').apply(lambda x: x[['edge_id', 'edge_left']].set_index('edge_id').T)
        peri_entered_vehs = peri_entered_vehs.fillna(1e-5)
        edge_data['peri_entered_vehs'] = peri_entered_vehs.to_numpy()

        ## get perimeter waiting times
        # total waiting time on each perimeter links
        peri_waitingTime_tot = df_peri.groupby('interval_end').apply(lambda x: x[['edge_id', 'edge_waitingTime']].set_index('edge_id').T)
        peri_waitingTime_tot = peri_waitingTime_tot.fillna(1e-5)
        edge_data['peri_waiting_tot'] = peri_waitingTime_tot.to_numpy()        

        # average number of vehs on each perimeter links
        peri_veh_num = df_peri.groupby('interval_end').apply(lambda x: x[['edge_id', 'edge_sampledSeconds']].set_index('edge_id').T)
        peri_veh_num = peri_veh_num.fillna(1)
        edge_data['peri_sampledSeconds'] = peri_veh_num.to_numpy()        

        return edge_data

    def process_lane_output(self, file):
        ''' process edge level output with small resolution after whole simulation
            1. network level
               --- 'directly' get the metrics by pandas
            2. node level 
               --- get raw data to further process
        '''
        ## set file name as csv
        file=file.split('.')
        file = '.'.join([file[0],'csv'])

        ## 1.read csv
        # df = pd.read_csv(file, sep=';')
        df = pd.read_csv(file, sep=';', usecols=\
            ['edge_id','interval_begin','edge_waitingTime', 'edge_left', 'edge_sampledSeconds'])

        ## 2. fill na
        # df['lane_id'].fillna(-1, inplace=True)
        # df['lane_queueing_time'].fillna(0, inplace=True)
        df_fill = pd.DataFrame(product(np.arange(0,config['max_steps'],5, dtype=float), self.netdata['edge'].keys()), columns=['interval_begin', 'edge_id'])
        df['edge_id'] = df['edge_id'].astype(str)
        df = df_fill.merge( df, on=['interval_begin', 'edge_id'], how='left')
        df['edge_waitingTime'].fillna(0, inplace=True)
        df['edge_left'].fillna(0, inplace=True)
        df['edge_sampledSeconds'].fillna(0, inplace=True)

        ## 3. process data
        lane_data = {}

        ''' 1.1 network delay for each step (aggregated in 5 secs) '''
        df_time = df.groupby('interval_begin')['edge_waitingTime'].sum()
        lane_data['network_delay_step'] = df_time.to_numpy()

        ''' 1.2 network veh delay for each step (aggregated in 5 secs) '''
        grouped = df.groupby('interval_begin')
        agg_df = grouped.agg({'edge_waitingTime': 'sum', 'edge_sampledSeconds': 'sum'}).reset_index()
        agg_df['edge_sampledSeconds'] = agg_df['edge_sampledSeconds'].where(agg_df['edge_sampledSeconds'] != 0, 1)
        agg_df['step_waiting_time'] = agg_df['edge_waitingTime'] / agg_df['edge_sampledSeconds']

        lane_data['network_perveh_delay_step'] = agg_df['step_waiting_time'].to_numpy()
        lane_data['network_perveh_delay_mean'] = agg_df['edge_waitingTime'].sum() /agg_df['edge_sampledSeconds'].sum() # per second



        ''' 2.1 Node delay for each step'''
        df['ToNode'] = df['edge_id'].apply(lambda x: self.netdata['edge'][x]['incnode']) # add attribute of ToNode
        df_ToNode = df.groupby(['ToNode', 'interval_begin'])['edge_waitingTime'].sum()  # get ToNode queue at each step
        df_ToNode1 = df_ToNode.reset_index().pivot(index='interval_begin', columns='ToNode', values='edge_waitingTime')  # reset df

        # get dict of ToNode queue
        ToNode_queue_dict = {}
        for col in df_ToNode1.columns:
            ToNode_queue_dict[col] = df_ToNode1[col].to_numpy()

        lane_data['ToNode_delay_step'] = ToNode_queue_dict

        ''' 2.2 Node throughput for each step'''
        df_through = df.groupby(['ToNode', 'interval_begin'])['edge_left'].sum()  # get ToNode queue at each step
        df_through1 = df_through.reset_index().pivot(index='interval_begin', columns='ToNode', values='edge_left')  # reset df

        # get dict of ToNode throughput
        ToNode_throughput_dict = {}
        for col in df_through1.columns:
            ToNode_throughput_dict[col] = df_through1[col].to_numpy()

        lane_data['ToNode_throughput_step'] = ToNode_throughput_dict

        ''' 2.3 Node sampletime for each step'''
        df_sampledSeconds = df.groupby(['ToNode', 'interval_begin'])['edge_sampledSeconds'].sum()  # get ToNode queue at each step
        df_sampledSeconds1 = df_sampledSeconds.reset_index().pivot(index='interval_begin', columns='ToNode', values='edge_sampledSeconds')  # reset df

        # get dict of ToNode queue
        ToNode_sampledSeconds_dict = {}
        for col in df_sampledSeconds1.columns:
            ToNode_sampledSeconds_dict[col] = df_sampledSeconds1[col].to_numpy()

        lane_data['ToNode_sampledSeconds_step'] = ToNode_sampledSeconds_dict
        
   
        return lane_data

## helper funcs
    def _get_buffer_edges(self):
        ''' get the list of buffer edges '''
        edge_buffers=[]
        for edgeID in self.netdata['edge']:
            if int(edgeID) not in config['Edge_Peri'] and \
                int(edgeID) not in config['Edge_PN']: # for edges in PN
                edge_buffers.append(int(edgeID))
        
        return edge_buffers

if False:
    def update_control_interval(self):
        ''' Update network metrics for each control interval
        '''
        ## accu -- [average]
        self.accu_PN_list_epis.append(np.mean(self.accu_PN_list_interval))
        self.accu_buffer_list_epis.append(np.mean(self.accu_buffer_list_interval))

        ## throughput -- [sum]
        # self.throuput_list_epis.append(sum(self.throuput_list_interval))

        ## mean speed -- [average]
        self.meanspeed_list_epis.append(np.mean(self.meanspeed_list_interval))
        
        ## production -- [average]
        self.production_list_epis.append(np.mean(self.production_list_interval))

        ## halting vehs buffer and PN -- [average]
        self.halveh_buffer_list_epis.append(np.mean(self.halveh_buffer_list_interval))
        self.halveh_PN_list_epis.append(np.mean(self.halveh_PN_list_interval))

        ## entered vehs -- [sum]
        # self.entered_vehs_list_epis.append(sum(self.entered_vehs_list_interval))

    ## entered vehs
    def calculate_entered_veh_onestep(self, peri_info ):
        ''' obtain entered vehicles from perimeter for each info interval
        '''
        entered_veh = []

        for tlsID in peri_info.keys():
            edge_num = peri_info[tlsID]['edge']  # get the controlled edge
            entered_veh.append(self.edge_info[edge_num]['left'])

        self.entered_vehs_list_interval.append(sum(entered_veh))

    def get_entered_veh_control_interval(self):
        ''' obtain entered vehs within the control interval
            -Sum up the entered vehs of info intervals
        '''
        return(self.entered_vehs_list_epis[-1])
