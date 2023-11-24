from distutils.command.config import config
from matplotlib import pyplot as plt
import numpy as np
import random
import timeit
from time import time
from utils.utilize import  plot_MFD, plot_accu, plot_actions, plot_flow_MFD, plot_flow_progression, plot_phase_mean_time, plot_tsc_delay, plot_peri_waiting, plot_controlled_tls_delay_epis
import traci

# from train_main import Agent_upper


class Trainer():    
    ''' A class to manage the training 
    '''
    def __init__(self, env, agent_lower, agent_upper, sumo_cmd_e, e, n_jobs, config):
        self.env = env
        self.agent_upper = agent_upper
        self.agent_lower = agent_lower
        self.cur_epis = e
        self.sumo_cmd = sumo_cmd_e
        self.n_jobs = n_jobs
        self.config = config
        self.model_dir = self.config['models_path_name'] if config['mode'] == 'train' else config['cache_path_name']
        self.info_interval = self.config['infostep']
        self.cycle_time = self.config['cycle_time']
        self.lower_mode = config['lower_mode']
        self.mode = config['mode']

    def run(self):
        ''' run the training for one epsiode
        '''
        # print(self.agent_upper.accu_crit)

        ## set random seeds
        self.set_seeds()

        ## clear buffer, set output files
        if self.n_jobs > 0:
            self.clear_buffer()
            self.set_output_file()
        else: ## This is to change the single process 0 to 1
            self.n_jobs = 1
        

        self.set_action_type()


        ## Set epsilon
        self.set_epsilon()  
        print(f'############# Episode {self.cur_epis+1}: {self.agent_upper.action_type}, epsilon = {self.agent_upper.epsilon}, \
         {self.agent_lower.action_type}, epsilon = {self.agent_lower.epsilon} ##############')

        ## Load weights of NN
        self.load_weights(self.cur_epis, self.n_jobs)

        ## run the simulation to get memories
        self.explore()

        
        ## 4. after simulation
        edge_data, lane_data = self.process_output()
        upper_metric, lower_metric = self.get_metric(edge_data, lane_data)


        ## 4.1 get upper reward
        upper_metric['upper_reward_epis'] = self.get_upper_reward(upper_metric['flow'], upper_metric['peri_entered_vehs'])
        lower_metric['lower_reward_epis'] = self.get_lower_reward(lower_metric['tsc_metrics'])


        ## 4.3 fill the upper reward and metrics
        upper_metric['upper_penalty_epis'] = self.fill_upper_buffer_reward(upper_metric['upper_reward_epis'],  upper_metric['PN_waiting'])
        
        upper_metric['cul_reward'] = [sum(upper_metric['upper_reward_epis'])]
        upper_metric['cul_penalty'] = [sum(upper_metric['upper_penalty_epis'])]
        upper_metric['cul_obj'] = [sum(upper_metric['cul_reward'] + upper_metric['cul_penalty'])]

        ## 4.4 record
        self.agent_upper.record(upper_metric)
        self.agent_lower.record(lower_metric)

        self.print_result(upper_metric, lower_metric)


        ## 4.5 plot
        self.plot(upper_metric['accu'], upper_metric['flow'], \
            upper_metric['cul_reward'][0], lower_metric['tsc_perveh_delay_mean'], \
                upper_metric['peri_entered_vehs'],\
                     upper_metric['peri_waiting_tot'], upper_metric['peri_waiting_mean'], \
                        lower_metric['tsc_delay_step'],lower_metric['tsc_perveh_delay_step'],lower_metric['tsc_metrics'])

        ## 4.6 process output
        if self.mode == 'train':
            ## test and train
            if self.cur_epis % self.n_jobs == 0:
                ## if it's not the testing /single process
                upper_metric['test'] = True
            else:
                upper_metric['test'] = False
        else:
            upper_metric['test'] = True

        ## expert and explore
        upper_metric['expert'] = True \
            if self.agent_upper.action_type == 'Expert' else False
            
            


        return self.agent_upper.buffer.buffer, self.agent_lower.memory,\
            upper_metric, lower_metric
 
    def explore(self):
        ''' explore within the episode
        '''

        step, done = self.reset()

            ## 3.simulate the episode
        while not done:
            ## output bar for multiple process
            print('\t' * (self.cur_epis % self.n_jobs) * 3 + f"{self.cur_epis+1} current step: {step}", end='\r')

            ## 3.1 get action
            self.get_actions(step)

            ## 3.2. simulation of one control interval
            step, done, _, _ = self.env.simu_run(step)  

            ## 3.3 upper metric update
            if step % self.info_interval == 0:
                self.agent_upper.get_metric_each_interval()

            ## 3.4 memory
            done = self.memorize(step, done)
            # plot_tsc_delay(self.agent_lower.tsc)

            if done == True:
                traci.close()
  

    ### helper funcs
    def set_seeds(self):
        mod_num = random.randint(0,9999)
        random.seed(int(time()*1e10 % mod_num))
        np.random.seed(int(time()*1e10 % mod_num))

    def clear_buffer(self):
        ''' clear buffer of upper/lower agent
        '''
        ## clear buffer, only save new buffers
        self.agent_upper.buffer.clear()
        self.agent_lower.init_memory()

    def set_output_file(self):
        ''' set output file direction with paraller computing
        '''
        ## redefine upper-level edge output file
        agent_upper_outputfile = self.agent_upper.outputfile.split('/')
        agent_upper_outputfile[1] += str(self.cur_epis % self.n_jobs + 1)
        self.agent_upper.outputfile = '/'.join(agent_upper_outputfile)

        ## redefine lower-level edge output file
        agent_lower_outputfile = self.agent_lower.outputfile.split('/')
        agent_lower_outputfile[1] += str(self.cur_epis % self.n_jobs + 1)
        self.agent_lower.outputfile = '/'.join(agent_lower_outputfile)
        # print(self.agent_lower.outputfile)

    def set_action_type(self):
        self.agent_upper.set_action_type(self.n_jobs, self.cur_epis)
        self.agent_lower.set_action_type(self.n_jobs, self.cur_epis)

    def set_epsilon(self):
        self.agent_upper.set_epsilon(self.n_jobs, self.cur_epis)
        self.agent_lower.set_epsilon(self.n_jobs, self.cur_epis)

    def load_weights(self, e, njobs):
        if e//njobs == 0:
            path = self.config['cache_path_name']
        else:
            path = self.model_dir

        if self.agent_upper.peri_mode in ['DQN', 'DDPG', 'C_DQN']:
            self.agent_upper.load_weights(path)

        if self.agent_lower.lower_mode in ['OAM']:
            self.agent_lower.load_weights(path)

    def reset(self):
        ''' reset to start the simulation
        '''

        ######## must generate demand before traci.start ########
        ## 1. reset envir
        self.env.reset(self.sumo_cmd, self.config)

        ## 2. reset upper agent
        self.agent_upper.reset()
        
        ## 3. reset lower agent
        self.agent_lower.reset()

        ## lower agent fixtime plan
        if self.lower_mode == 'FixTime':
            self.agent_lower.set_program()
        ## Set the action type

        ## 4. 
        self.env.agent_lower = self.agent_lower

        ## training
        step, done = 0, False
        return step, done


    def get_actions(self, step):
        ''' get actions of upper and lower
        '''
        ## upper
        if step % self.cycle_time == 0:
            self.agent_upper.implem_action_all(step)

        ## lower
        self.agent_lower.implem_action_all()

    def memorize(self, step, done ):
        ''' save the memory for upper and lower levels after each step
        '''
        ## upper level
        if step % self.cycle_time == 0:
            ## update metric
            # self.agent_upper.metric.update_control_interval()

            ##  end simu  '''Before the memorize '''
            done = self.check_gridlock(done, step)

            self.agent_upper.get_memory(done)

        ## lower level
        # if self.lower_mode != 'FixTime':
        self.agent_lower.store_memory(done)
        return done

    def check_gridlock(self, done,step):
        ''' check gridlock
            Upper: DQN and DDPG need gridlock break done
        '''
        ## break the episode if grid-lock
        # if self.lower_mode == 'OAM':
        if self.agent_upper.peri_mode in ['DQN', 'DDPG']:
            _, PN_halt_vehs = self.agent_upper.metric.get_halting_vehs(info_inter_flag=True)
            PN_halt_vehs = PN_halt_vehs/self.config['PN_halt_vehs_max']
            if  not done and PN_halt_vehs >0.7:
                done = True  # terminate the episode
                print(f'########## Grid lock of episode {self.cur_epis} at {step} ##########')
                # ## end traci
                # traci.close()
        
        return done

    def process_output(self):
        '''  process edge_data and lane_data after each epis
        '''
        ## output for entered vehicles
        # self.agent_upper.metric.process_output(self.agent_upper.outputfile)

        ## output for csv
        self.agent_upper.metric.xml2csv(self.agent_upper.outputfile) ## edge
        self.agent_upper.metric.xml2csv(self.agent_lower.outputfile) ## lane

        ## edge data
        edge_data = self.agent_upper.metric.process_edge_output(self.agent_upper.outputfile)
        
        ## lane data
        lane_data = self.agent_upper.metric.process_lane_output(self.agent_lower.outputfile)


        return edge_data, lane_data
    
    def get_metric(self, edge_data, lane_data):
        ''' get metric of upper / lower level after each epis
        '''
        ## upper
        upper_metric= self.agent_upper.get_metric_each_epis(edge_data)

        ## lower
        lower_metric = self.agent_lower.get_metric_each_epis(lane_data)

        return upper_metric, lower_metric

    def get_upper_reward(self, flow_epis, entered_vehs):
        ''' get upper reward of each step after each epis
        '''
        
        ## get reward
        upper_reward_epis = flow_epis/ self.config['reward_max']
        for i in range(len(upper_reward_epis)):
            if upper_reward_epis[i] >1:
                upper_reward_epis[i] = upper_reward_epis[i]**2

        entered_vehs = entered_vehs/600
        upper_reward_epis += entered_vehs
        return upper_reward_epis

    def get_lower_reward(self, tsc_metrics):
        ''' get lower reward of each agent after each epis
        '''
        if self.agent_lower.lower_mode == 'OAM':
            reward = self.agent_lower.get_reward(tsc_metrics)

        else:
            return None

        return reward

    def print_result(self, upper_metric, lower_metric):
        ''' print result after simulation
        '''
        print(f"\r\n### Episode {self.cur_epis+1} Finish --- total obj upper: {upper_metric['cul_obj'][0]} = {upper_metric['cul_reward'][0]}+{upper_metric['cul_penalty'][0]}; #############\r\n network delay mean: {np.around(lower_metric['network_delay_mean'],2)} sec; tsc delay mean:{np.around(lower_metric['tsc_delay_mean'],2)} sec \r\n network perveh delay mean: {np.around(lower_metric['network_perveh_delay_mean'],4)} sec; tsc perveh delay mean:{np.around(lower_metric['tsc_perveh_delay_mean'],4)} sec \r\n tsc through mean: {np.around(lower_metric['tsc_through_mean'],0)} veh/hr*tsc \r\n")

    def plot(self, accu_epis, flow_epis, cumul_obj_upper, cumul_reward_lower, peri_entered_vehs, peri_waiting_tot, peri_waiting_mean,tsc_delay_step, tsc_perveh_delay_step, tsc_metrics):
        ''' plot after one episode
        ''' 
        ## upper actions
        if self.agent_upper.peri_mode not in [ 'Static', 'MaxPressure']:
            plot_actions(self.config, self.agent_upper.perimeter.green_time, \
                peri_entered_vehs, self.cur_epis,  \
                    self.agent_upper.action_type, self.n_jobs)

        ## MFD
        plot_MFD(self.config, accu_epis, flow_epis, self.cycle_time,\
             self.cur_epis, cumul_obj_upper, self.n_jobs, cumul_reward_lower)
        
        ## plot flow
        # plot_flow_progression(self.config, flow_epis, self.cur_epis , self.n_jobs)

        ## peri waiting time
        # plot_peri_waiting(self.config, peri_waiting_tot, peri_waiting_mean, self.cur_epis, self.n_jobs)

        ## accu, through, buffer queue progress along the training
        # if self.cur_epis % self.n_jobs == 0:
        #     plot_accu(self.config, accu_episode, throughput_episode, \
        #         self.agent_upper.metric.halveh_buffer_list_epis, self.cur_epis)

        ## lower phase time
        if self.lower_mode != 'FixTime':
            pass
            # plot_phase_mean_time(self.config, self.agent_lower.controled_light, \
            #     self.agent_lower.tsc, self.cur_epis, self.n_jobs)
        
        ''' network delay with controlled tsc '''
        plot_controlled_tls_delay_epis(self.config, tsc_delay_step,self.cur_epis, self.n_jobs, 2000, 'tsc_delay')
        plot_controlled_tls_delay_epis(self.config, tsc_perveh_delay_step,self.cur_epis, self.n_jobs, 1,'tsc_perveh_delay')

        ''' each controlled tsc delay'''
        plot_tsc_delay(self.config, tsc_metrics, self.cur_epis, self.n_jobs)

    def fill_upper_buffer_reward(self, upper_reward_epis, PN_waiting_epis): 
        
        upper_penalty_epis = -PN_waiting_epis /1.5e5

        if self.agent_upper.buffer.buffer:
            for idx, (memory, reward) in \
                enumerate(zip(self.agent_upper.buffer.buffer, upper_reward_epis)):
                
                memory[2] = reward
                
                # memory[-1] = penalty if memory[0][0] > 0.35 else 0
                
                ##penalty
                if self.config['network']=='Grid':
                    upper_penalty_epis = -PN_waiting_epis /1.5e5

                    if memory[0][0] <= 0.3:
                        upper_penalty_epis[idx] = 0
                    # elif  0.25<=memory[0][0] and memory[0][0]<= 0.35:
                    #     if reward<0.8:
                    #         upper_penalty_epis[idx] += -1
                    #     else:
                    #         upper_penalty_epis[idx] = 0
                    else:
                        upper_penalty_epis[idx] += -1

                elif self.config['network']=='Bloomsbury':
                    upper_penalty_epis = -PN_waiting_epis /1.5e5

                    if memory[0][0] <= 0.6:
                        upper_penalty_epis[idx] = 0
                    # elif  0.25<=memory[0][0] and memory[0][0]<= 0.35:
                    #     if reward<0.8:
                    #         upper_penalty_epis[idx] += -1
                    #     else:
                    #         upper_penalty_epis[idx] = 0
                    elif memory[0][0] >= 0.6 :
                        upper_penalty_epis[idx] += -1

                memory[-1] = upper_penalty_epis[idx]
            print(min(upper_penalty_epis))


        return upper_penalty_epis