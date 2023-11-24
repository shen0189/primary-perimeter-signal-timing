import numpy as np


class TrafficGenerator:
    def __init__(self, demand_profile, demand_multiplier, demand_interval, noise_mean, noise_variance, max_steps):
        self.demand_profile = demand_profile  # the peak demand
        self.demand_multiplier = demand_multiplier
        self.demand_interval = demand_interval
        self.demand_total_interval = max_steps // demand_interval
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance





    def gen_demand(self, seed, route_file_name, EdgeIn, EdgeOut, Node, turn_ratio):
        np.random.seed(seed) # set the random seed

        with open(route_file_name, "w") as routes:
            print("""<routes>
                <vType id="Car0" carFollowModel="IDM" accel="2.6" decel="4.5" tau="1.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="18" departLane="best"/>
                <vehicles>""", file=routes)
      
            # generate demand for each period
            demand1 = np.linspace (self.demand_profile['bottom'], self.demand_profile['peak'], self.demand_profile['incre_inter_num'])  # [start, end, num]
            demand2 = np.array([self.demand_profile['peak']] * self.demand_profile['high_inter_num']) 
            demand3 = np.linspace(self.demand_profile['peak'], self.demand_profile['bottom'], self.demand_profile['decre_inter_num'])
            demand4 = np.array([self.demand_profile['bottom']] * self.demand_profile['low_inter_num'])
            demand = np.concatenate((demand1, demand2, demand3, demand4)) * self.demand_multiplier

            # add noise and multiplier
            demand = demand * (np.random.normal(self.noise_mean, self.noise_variance, self.demand_total_interval) + 1) # [mean, variance, num]
            
            nodenum = np.size(EdgeIn)        
            timelist = [i * self.demand_interval for i in range(self.demand_total_interval)]
            #for time in range(len(timelist)-1):  
            for time in range(len(timelist)-1):  
                
                #for in_id in range(nodenum):
                for in_id in range(nodenum):
                    row = int(in_id / 4)
                    col = in_id % 4
                    print(f'                <flow id="F{Node[row]*10000+Node[(row-3)*4+col]*100+time}" begin="{timelist[time]}" end="{timelist[time+1]}" vehsPerHour="{int(turn_ratio[0] * demand[time])}" type="Car0" departLane="3" departSpeed="" from="{EdgeIn[row,col]}" to="{EdgeOut[row-3,col]}"/>', file=routes)
                    print(f'                <flow id="F{Node[row]*10000+Node[(row-2)*4+col]*100+time}" begin="{timelist[time]}" end="{timelist[time+1]}" vehsPerHour="{int(turn_ratio[1] *demand[time])}" type="Car0" departLane="1" departSpeed="random" from="{EdgeIn[row,col]}" to="{EdgeOut[row-2,col]}"/>', file=routes)
                    # print(f'                <flow id="F{Node[row]*10000+Node[(row-2)*4+col]*100+time}" begin="{timelist[time]}" end="{timelist[time+1]}" vehsPerHour="{int(0.4*demand[time])}" type="Car0" departLane="2" departSpeed="random" from="{EdgeIn[row,col]}" to="{EdgeOut[row-2,col]}"/>', file=routes)
                    print(f'                <flow id="F{Node[row]*10000+Node[(row-1)*4+col]*100+time}" begin="{timelist[time]}" end="{timelist[time+1]}" vehsPerHour="{int(turn_ratio[2] * demand[time])}" type="Car0" departLane="0" departSpeed="random" from="{EdgeIn[row,col]}" to="{EdgeOut[row-1,col]}"/>', file=routes)

            print("</vehicles>"
                "</routes>", file=routes)
        return demand

