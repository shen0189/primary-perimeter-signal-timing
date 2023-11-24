import numpy as np
# from numpy.core.defchararray import index
import random
from random import choice
from utils.utilize import config, plot_demand,  get_NodeConfig


class TrafficGenerator():
    def __init__(self):
        self.demand_config = config['DemandConfig']  # the peak demand
        self.demand_interval_sec = config['Demand_interval_sec']
        self.demand_interval_total = config['Demand_interval_total'] 
        self.noise_mean = config['DemandNoise']['noise_mean']
        self.noise_variance = config['DemandNoise']['noise_variance']
        self.NodeConfig = get_NodeConfig(config['edgefile_dir'])
        self.OD_Pairs = {}
        self.route_file_name = config['routefile_dir']
        # self.noise_variance = noise_variance
        # self.demand_multiplier = demand_multiplier
        # self.demand_total_interval = max_steps // demand_interval

    def gen_demand(self, config):
        ''' generate demand for the network
        '''
        # np.random.seed(1) 
        # random.seed(1)

        with open(self.route_file_name, "w") as routes:
            print('<routes>\n\t<vType id="Car0" carFollowModel="IDM" accel="2.6" decel="4.5" tau="1.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="18" departLane="best"/>\n\t\t<vehicles>', file=routes)

            for DemandType in self.demand_config:

                FrRegion_Node = config['Node'][DemandType['FromRegion']]
                ToRegion_Node = config['Node'][DemandType['ToRegion']]

                ### 1. obtain OD pairs
                OD_PairsEdge, OD_PairsNode = self.get_randOD(FrRegion_Node, ToRegion_Node) 
                DemandType['OD_Edge'] = OD_PairsEdge
                DemandType['OD_Node'] = OD_PairsNode

                OD_num = len(OD_PairsEdge) # number of OD pairs for this demand type
                
                # init
                Demand_Inter = np.zeros([OD_num,1])
                DemandType['Demand'] = []
                ### 2. obtain demand of all intervals for each OD_pairs 
                for i, VolumnRange in enumerate(DemandType['VolumnRange']):
                    ### obtain demand profile for each period
                    
                    interval_num = DemandType['Demand_interval'][i] # number of intervals for this demand type

                    Demand_split = self.cal_demand(VolumnRange, interval_num, OD_num) # get the demand for each OD
                    Demand_Inter = np.concatenate ((Demand_Inter, Demand_split), axis=1)

                ### 3. process the demand with noise and demand multiplier
                Demand_Inter = Demand_Inter[:, 1:] * DemandType['Multiplier'] # multiplier
                Demand_Inter = np.round(Demand_Inter * (np.random.normal(self.noise_mean, self.noise_variance, Demand_Inter.shape) + 1)) # [mean, variance, num]
                
                ### 4. record all intervals for each demayd type
                DemandType['Demand_split'] = Demand_Inter
                DemandType['Demand'].append(sum(Demand_Inter))
            

            ### 5. write the demand
            self.write_demand(routes, sum(DemandType['Demand_interval']))

            
            print("\t\t</vehicles>\n</routes>", file=routes)

            plot_demand(config, self.demand_config, self.demand_interval_sec, self.demand_interval_total)

            config['DemandConfig'] = self.demand_config


    def get_randOD(self, FrRegion_Node, ToRegion_Node):
        ''' Obtain OD pairs with DepartEdge and ArriveEdge
        '''
        OD_PairsEdge = []
        OD_PairsNode = []
        for FrNode in FrRegion_Node:
            for ToNode in ToRegion_Node:
                if FrNode != ToNode: 

                    DepartEdge = choice(self.NodeConfig[FrNode]['OutEdge']) # random choose the DepartEdge from the InEdge
                    ArriveEdge = choice(self.NodeConfig[ToNode]['InEdge']) # random choose the DepartEdge from the InEdge

                    if DepartEdge != ArriveEdge:
                        OD_PairsEdge.append([DepartEdge, ArriveEdge])
                        OD_PairsNode.append([FrNode, ToNode])
                  
        return OD_PairsEdge, OD_PairsNode

    def cal_demand(self, VolumnRange, interval_num, OD_num):
        ''' Calculate the demand profile for each OD_pair with demand multipliers
        '''
        # First, calculate overall demand for each interval
        # VolumnRange[start-demand, end-demand, multipliers]
        Demand_interAll = np.linspace (VolumnRange[0], VolumnRange[1], interval_num)
        Demand_interAll = Demand_interAll[:, np.newaxis]

        # Second, generate split ratio randomly
        # split_ratio = random.Generator.dirichlet(np.ones(OD_num), size = 1)
        # nums = np.random.uniform(1, 3, OD_num)
        # split_ratio = nums / sum(nums)
        split_ratio = np.array([1/OD_num]*OD_num)
        split_ratio = split_ratio[:, np.newaxis]

        # Third, obtain demand of each interval for each OD
        Demand_split = (Demand_interAll * split_ratio.T).T
        return  Demand_split

    def write_demand(self, routes, Demand_Inter):
        ''' Write the demand profile into the route files
        '''
        
        timelist = [i * self.demand_interval_sec for i in range(self.demand_interval_total)]

        for time in range(len(timelist)-1):  # each interval
            for DemandType in self.demand_config:  # each demand type
                for i, OD_Edges in enumerate(DemandType['OD_Edge']): # each OD pairs
                    if DemandType['Demand_split'][i][time] != 0: 
                        DepartEdge, ArriveEdge = OD_Edges
                        FrNode, ToNode = DemandType['OD_Node'][i]
                        value = int(DemandType['Demand_split'][i][time])
                        print(f'\t\t\t<flow id="F{FrNode*10000 + ToNode*100 + time}" begin="{timelist[time]}" end="{timelist[time+1]}" vehsPerHour="{value}" type="Car0" departLane="best" departSpeed="random" from="{DepartEdge}" to="{ArriveEdge}"/>', file=routes)
                        # print(f'                <flow i
                        # d="F{Node[row]*10000+Node[(row-2)*4+col]*100+time}" begin="{timelist[time]}" end="{timelist[time+1]}" vehsPerHour="{int(turn_ratio[1] *demand[time])}" type="Car0" departLane="1" departSpeed="random" from="{EdgeIn[row,col]}" to="{EdgeOut[row-2,col]}"/>', file=routes)    
                        # print(f'                <flow id="F{Node[row]*10000+Node[(row-1)*4+col]*100+time}" begin="{timelist[time]}" end="{timelist[time+1]}" vehsPerHour="{int(turn_ratio[2] * demand[time])}" type="Car0" departLane="0" departSpeed="random" from="{EdgeIn[row,col]}" to="{EdgeOut[row-1,col]}"/>', file=routes)
