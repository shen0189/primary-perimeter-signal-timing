from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import matplotlib

from utils.utilize import plot_accu_critic

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from agents.loweragents import OAM, FixTime, MaxPressure
from agents.upperagents import DDPG, DQN, Expert, Static, C_DQN, MFD_PI
from utils.networkdata import NetworkData
from utils.utilize import Test, config, plot_lower_reward_epis,  save_config, save_data_train_upper, set_sumo, plot_MFD, plot_actions, plot_critic_loss, plot_throughput, \
 plot_accu, plot_computime, set_train_path, write_log, plot_obj_reward_penalty, set_test_path
import timeit
from time import time
import random
from copy import copy
from utils.genDemandBuffer import TrafficGenerator
# from memory import Memory
# from dqn_agent import DQNAgent
from envir.envir import Simulator
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from envir.manager import Trainer


def simulate_one_episode_train(e, config, netdata, accu_crit = None):
    ''' simulation with one episode  *** Multiple process ***
        *** 1. Parallel: This is the simulation being 'paralleled'
        *** 2. Configure: Many arguments need to be redefined or passed in, because the file will be reloaded
        *** 3. Explore: The Agent and the Env are the 'initial one'--- epsilon need to be recalculated 
        *** 4. Buffer: In MP, the buffer will be cleared-- Only save new memory
        *** 5. Evaluate: -The first episode is for testing --- No exploration
        ***              -In single process --- No exploration
        *** 6. Record: MP --- 'list to be append' only holds current infos --- Need to record outside parallel
                       SP --- 'list to be append' holds all infos along all episodes --- Agent is not reloaded
        *** 7. Plot:   MP --- plot the testing episode
                       SP --- plot the single process one
        *** 8. Retrun: MP --- retrun value of the testing episode  
                       SP --- retrun value of the single process episode   
    '''
    # random.seed(int(time()*1000 % 12345))
    # np.random.seed(int(time()*1000 % 12345))
    # np.random.seed(int(time() * 10000) % 100)
    #################### Prepare for Simulation with MP ##########################

    ## 1. Redefine number of jobs
    n_jobs = config['n_jobs']
    if n_jobs == -1:
        n_jobs = cpu_count()

    sumo_cmd_e = copy(sumo_cmd)

    ## 2. Re-difine path and files
    if n_jobs > 0:
        sumocfg_file_name = sumo_cmd[2].split('/')
        sumocfg_file_name[1] += str(e % n_jobs + 1)  # set the config file
        sumocfg_file_name = '/'.join(sumocfg_file_name)
        sumo_cmd_e[2] = sumocfg_file_name  # set the cmd path

        # error_file_name = sumo_cmd[-1].split('/')
        # error_file_name[1] += str(e % n_jobs + 1)
        # error_file_name = '/'.join(error_file_name)
        # sumo_cmd_e[-1] = error_file_name

        ## queue fi
        queue_file_name = sumo_cmd[-1].split('/')
        queue_file_name[1] += str(e % n_jobs + 1)
        queue_file_name = '/'.join(queue_file_name)
        sumo_cmd_e[-1] = queue_file_name

    # print(sumo_cmd_e)

    ## 3. Re-load the agent and environment
    agent_upper = Agent_upper   
    agent_lower = Agent_lower  
    if accu_crit:
        agent_upper.accu_crit = accu_crit
    # env = Env
    trafficGen = TrafficGenerator()
    if n_jobs>0:
        route_file_name = trafficGen.route_file_name.split('/')
        route_file_name[1] += str(e % n_jobs + 1)
        trafficGen.route_file_name = '/'.join(route_file_name)
    env = Simulator(trafficGen, netdata)

    trainer = Trainer (env, agent_lower, agent_upper, sumo_cmd_e, e, n_jobs, config)
    
    buffer_upper, buffer_lower,\
        upper_metric, cumul_reward_lower,\
            = trainer.run()

    return buffer_upper, buffer_lower, upper_metric, cumul_reward_lower

def train():
    ''' train the RL controller with 'multiple process'
    '''
    ## 1. set the agent and environment
    agent_upper = Agent_upper   
    agent_lower = Agent_lower   
    env = Env

    ## 2. set the number of jobs: -1 = all cores; 0 = single process; 1+ = MP
    n_jobs = config['n_jobs']
    if n_jobs == -1:
        n_jobs = cpu_count()
        print(f'########## Total CPU number: {n_jobs}')

    ## 3. check the network configure files for the MP
    network_dir = os.listdir('network')
    for i in range(1, n_jobs + 1):
        if f'GridBuffer{i}' not in network_dir:
            raise Exception(f'GridBuffer{i} not Exist')

    results = []

    accu_batch = []
    flow_batch = []
    ## 4. training with MP
    for e in range(0, agent_upper.episodes, max(1, n_jobs)):    
        start_time = timeit.default_timer()

        ## record the current number of memory in the buffer
        agent_upper.buffer.count_old = agent_upper.buffer.count
        # agent_lower.buffer_count_old = agent_lower.buffer_count

        ## 4.a train with multiple process
        if n_jobs > 0:
            print(f'\n######### Parallel {n_jobs} ##########\n')

            ## 4.a.1 simulation with MP
            
            pool = Pool(n_jobs)
            simu_return = [pool.apply_async(func=simulate_one_episode_train, args=(e + i, config, netdata, agent_upper.accu_crit)) for i in range(n_jobs)]
            pool.close()
            pool.join()

            ## 4.a.2 process the output of multiple process
            idx = 0
            for pr in simu_return:
                idx += 1
                buffer_upper, buffer_lower, upper_metric, lower_metric = pr.get()
                
                ## upper buffer
                agent_upper.buffer.buffer.extend(buffer_upper)
                agent_upper.buffer.count += len(buffer_upper)

                ## lower buffer
                if agent_lower.lower_mode=='OAM':
                    for key in agent_lower.memory.keys():
                        agent_lower.memory[key].extend(buffer_lower[key])
                        agent_lower.memory[key]= agent_lower.memory[key][-agent_lower.buffer_size:]  # pop out
                    agent_lower.buffer_count = len(agent_lower.memory['reward'])

                ## accu and flow batch for mfd fitting
                accu_batch.extend(upper_metric['accu'])
                flow_batch.extend(upper_metric['flow'])

                ## best episode except for expert actions
                if upper_metric['expert'] == False:
                    agent_upper.get_best_episode(upper_metric)

                ## record accu and throu: the output value of testing episode
                if upper_metric['test'] == True:
                    ## upper
                    agent_upper.record(upper_metric)
                    agent_lower.record(lower_metric)

                ## save the best nn parameters for training
                if upper_metric['test'] == True and config['mode'] =='train' and upper_metric['expert'] == False:
                    agent_upper.save_best_critic(upper_metric['cul_obj'][0])

                ## lower


        ## 4.b train with single process
        else:
            buffer_upper, buffer_lower, upper_metric, lower_metric \
                = simulate_one_episode_train(e, config, netdata, agent_upper.accu_crit)
            accu_batch.extend(upper_metric['accu'])
            flow_batch.extend(upper_metric['flow'])
            
            ## best episode
            agent_upper.get_best_episode(upper_metric)


        ## 5. plots along trainning process
        # upper
        plot_obj_reward_penalty(agent_upper.record_epis['cul_obj'], agent_upper.record_epis['cul_penalty'], agent_upper.record_epis['cul_reward'])
        # plot_throughput(agent_upper.throughput_episode)
        # lower
        plot_lower_reward_epis(agent_lower.record_epis['tsc_perveh_delay_mean'])


        ## 6. process new memory, save memory buffer to the path
        if agent_upper.buffer.buffer:
            agent_upper.buffer.process_new_memory(e)  #process memory
            agent_upper.buffer.save()

        # if agent_lower.buffer.buffer:
        #     agent_lower.buffer.process_new_memory(e)  #process memory
        #     agent_lower.buffer.save()


        ## 7. replay to train the NN of upper level
        if config['mode'] == 'train':
            agent_upper.train(e, n_jobs)
            agent_lower.train(e, n_jobs)

        ## 8. record computational time
        simulation_time = round(timeit.default_timer() - start_time, 1)
        agent_upper.computime_episode.append(simulation_time)
        plot_computime(agent_upper.computime_episode)
        print(f"###### simulation time: {simulation_time}")
        # print(f"###### explore_std: {agent.explore_std}")

        ## 9. reduce exploration
        # agent_upper.epsilon = agent_upper.epsilon * agent_upper.explore_decay
        # agent_upper.epsilon = agent_upper.epsilon * agent_upper.explore_decay

        ## 10. save agent parameters to the path
        agent_upper.save_weights(config['models_path_name'])
        agent_lower.save_weights(config['models_path_name'])

        ## 11. upper fit mfd
        if agent_upper.peri_mode == 'C_DQN' and agent_upper.mode == 'train':
            agent_upper.fit_mfd(accu_batch, flow_batch)
            ncrit, Gneq = agent_upper.cal_ncritic()
            agent_upper.update_ncritic(ncrit)
            plot_accu_critic(agent_upper.accu_crit_list)


        ## 11. save accu and throughput of all the episodes
        save_data_train_upper(agent_upper, agent_lower)

    return results

def test():
    ''' train the RL controller with 'multiple process'
    '''
    ## 1. set the agent and environment
    agent = Agent   
    env = Env
    tester = Test()

    ## 2. set the number of jobs: -1 = all cores; 0 = single process; 1+ = MP
    n_jobs = config['n_jobs']
    if n_jobs == -1:
        n_jobs = cpu_count()
        print(f'########## Total CPU number: {n_jobs}')

    ## 3. check the network configure files for the MP
    network_dir = os.listdir('network')
    for i in range(1, n_jobs + 1):
        if f'GridBuffer{i}' not in network_dir:
            raise Exception(f'GridBuffer{i} not Exist')

    results = {}

    ## for pre-train: the buffer_size is unlimited：
    # if config['expert_episode'] > 0 :
    #     agent.buffer.buffer_size = int(1e5)

    ## 4. training with MP
    for e in range(0, agent.episodes, max(1, n_jobs)):

        ## record the current number of memory in the buffer
        agent.buffer.count_old = agent.buffer.count

        ## 4.a train with multiple process
        if n_jobs > 0:
            print(f'\n######### Parallel {n_jobs} ##########\n')

            ## 4.a.1 simulation with MP
            pool = Pool(n_jobs)
            simu_return = [pool.apply_async(func=simulate_one_episode_test, args=(e + i, config)) for i in range(n_jobs)]
            pool.close()
            pool.join()

            ## 4.a.2 process the output of multiple process
            for pr in simu_return:
                e, cumul_obj, cumul_reward, cumul_penalty, accu_episode, throughput_episode, actions = pr.get()
                tester.record_data(cumul_obj, cumul_reward, cumul_penalty, accu_episode, throughput_episode, actions)

        ## 4.b train with single process
        else:
            e, cumul_obj, cumul_reward, cumul_penalty, accu_episode, throughput_episode, actions = simulate_one_episode_test(e, config)
            tester.record_data(cumul_obj, cumul_reward, cumul_penalty, accu_episode, throughput_episode, actions)
            
        ## 11. save data 
        tester.save_data_test()
    return results

def simulate_one_episode_test(e, config):
    ''' simulation with one episode  *** Multiple process ***
        *** 1. Parallel: This is the simulation being 'paralleled'
        *** 2. Configure: Many arguments need to be redefined or passed in, because the file will be reloaded
        *** 3. Explore: The Agent and the Env are the 'initial one'--- epsilon need to be recalculated 
        *** 4. Buffer: In MP, the buffer will be cleared-- Only save new memory
        *** 5. Evaluate: -The first episode is for testing --- No exploration
        ***              -In single process --- No exploration
        *** 6. Record: MP --- 'list to be append' only holds current infos --- Need to record outside parallel
                       SP --- 'list to be append' holds all infos along all episodes --- Agent is not reloaded
        *** 7. Plot:   MP --- plot the testing episode
                       SP --- plot the single process one
        *** 8. Retrun: MP --- retrun value of the testing episode  
                       SP --- retrun value of the single process episode   
    '''
    # random.seed(int(time()*1000 % 12345))
    # np.random.seed(int(time()*1000 % 12345))
    # np.random.seed(int(time() * 10000) % 100)
    #################### Prepare for Simulation with MP ##########################

    ## 1. Redefine number of jobs
    n_jobs = config['n_jobs']
    if n_jobs == -1:
        n_jobs = cpu_count()

    sumo_cmd_e = copy(sumo_cmd)

    ## 2. Re-difine path and files
    if n_jobs > 0:
        sumocfg_file_name = sumo_cmd[2].split('/')
        sumocfg_file_name[1] += str(e % n_jobs + 1)  # set the config file
        sumocfg_file_name = '/'.join(sumocfg_file_name)
        sumo_cmd_e[2] = sumocfg_file_name  # set the cmd path

        # error_file_name = sumo_cmd[-1].split('/')
        # error_file_name[1] += str(e % n_jobs + 1)
        # error_file_name = '/'.join(error_file_name)
        # sumo_cmd_e[-1] = error_file_name

    ## 3. Re-load the agent and environment
    agent = Agent
    # env = Env
    trafficGen = TrafficGenerator()
    if n_jobs>0:
        route_file_name = trafficGen.route_file_name.split('/')
        route_file_name[1] += str(e % n_jobs + 1)
        trafficGen.route_file_name = '/'.join(route_file_name)
    env = Simulator(trafficGen)

    ## set random seeds
    mod_num = random.randint(0,9999)
    random.seed(int(time()*1e10 % mod_num))
    np.random.seed(int(time()*1e10 % mod_num))
    
    ## 4. for MP: Redefine the output file in the Env, clear the buffer
    if n_jobs > 0:
        ## redefine output file
        env_outputfile = env.outputfile.split('/')
        env_outputfile[1] += str(e % n_jobs + 1)
        env.outputfile = '/'.join(env_outputfile)

        ## clear buffer, only save new buffers
        agent.buffer.clear()
    else: ## This is to change the single process 0 to 1
        n_jobs = 1

    ## 5. Set the action type of the simulation and Re-calculate the epsilon
    agent.set_action_type(n_jobs, e)
    agent.set_epsilon(n_jobs, e)
    print(f'############# Episode {e+1}: {agent.action_type}, epsilon = {agent.epsilon} ##############')

   
    ## 6. Load weights of NN
    agent.load_weights(config['cache_path_name'])

    #################### Simulation for one episode ##########################

    ## 1. reset env: generate demand and set up sumo
    old_state = env.reset(e, sumo_cmd_e, config, agent.env_dim)

    ## 2. initialize episode
    step, cumul_reward, cumul_penalty, done = 0, 0, 0, False
    action_excute_list = []


    ## 3.simulate the episode
    while not done:
        ## output bar for multiple process
        print('\t' * (e % n_jobs) * 3 + f"{e+1} current step: {step}", end='\r')

        ## 3.1 get action
        a, is_expert = agent.get_action_all(old_state)

        ## 3.2. simulation of one cycle: Retrieve new state, reward, and whether the state is terminal
        new_state, reward, penalty, done, step, entered_veh, action = env.simu_run(a, step)  # returned action is transformed

        ## 3.3. record action for each step
        action_excute_list.append(entered_veh)  #record entered vehicles

        ## check the state with NaN
        if np.isnan(new_state).any():
            print(f"####### CYCLE = {env.info_update_index/config['infostep']}" )
            print(f'####### STATE = {new_state}')
            raise Exception('There is NaN in the state')

        ## 3.4. store memory for each step
        # agent.memorize(old_state, a, reward, penalty, done, new_state, entered_veh)
        
        ## 3.5. Update current state
        old_state = new_state

        ## 3.6. Calculate one-step objective (reward+penalty)

        # penalty = agent.buffer.calculate_penalty(penalty_state, a, entered_veh, old_state)
        # print(penalty)
        cumul_reward += reward  # reward along this episode
        cumul_penalty += penalty


    #################### After Simulation for one episode: record & plots ##########################

    ## 4. after simulation
    ## 4.1 get the accu and throughput of one episode from env for  1). plot    2). record
    accu_episode = env.Metric.accu_PN_list_epis
    throughput_episode = env.Metric.throuput_list_epis

    ## 4.2 append reward, accu, throu to the 'all episodes' list
    cumul_obj = cumul_reward + cumul_penalty

    print(f"\r\n###### total reward of episode {e+1}: {cumul_obj}")
    plot_actions(config, env.Perimeter.green_time, action_excute_list, e,  agent.action_type)
    plot_MFD(config, accu_episode, throughput_episode, env.cycle_time, e, config['Peri_mode'], cumul_obj)
    plot_accu(config, accu_episode, throughput_episode, env.Metric.halveh_buffer_list_epis, e)

    return  e, cumul_obj, cumul_reward, cumul_penalty, accu_episode, throughput_episode, env.Perimeter.green_time

## 1. set sumo cmd
sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
# print(sumo_cmd)

## 2. initialize demand generator & netdata
TrafficGen = TrafficGenerator()

nd = NetworkData(config['netfile_dir'], sumo_cmd)
netdata = nd.get_net_data()
tsc, tsc_peri = nd.update_netdata()

## 3. init envir
Env = Simulator(TrafficGen, netdata)

## 4. initialize tsc controller (upper/lower)
# upper
if config['upper_mode'] == 'DDPG':
    # DDPG controller
    Agent_upper = DDPG(tsc_peri, netdata)

elif config['upper_mode'] == 'DQN': 
    # DQN controller
    Agent_upper = DQN(tsc_peri, netdata)
elif config['upper_mode'] == 'Expert' :
    Agent_upper = Expert(tsc_peri, netdata)
elif config['upper_mode'] == 'C_DQN' :
    Agent_upper = C_DQN(tsc_peri, netdata)
elif config['upper_mode'] == 'PI' :
    Agent_upper = MFD_PI(tsc_peri, netdata)
# elif config['upper_mode'] == 'Static' :
else:
    Agent_upper = Static(tsc_peri, netdata)


# lower
if config['lower_mode'] == 'OAM':
    Agent_lower = OAM(tsc, netdata)
elif config['lower_mode'] == 'FixTime':
    Agent_lower = FixTime(tsc, netdata)
elif config['lower_mode'] == 'MaxPressure':
    Agent_lower = MaxPressure(tsc, netdata)






if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    mp.set_start_method('spawn')
    ## train / test the controller
    # if config['mode'] == 'test':
    if False:
        config['plots_path_name'] = set_train_path(config['plots_path_name'], 'plot')
        config['models_path_name'] = set_train_path(config['models_path_name'], 'model')
        config['cache_path_name'] = set_test_path(config['cache_path_name'])

        ## save config file and log file
        save_config(config)
        write_log(config)

        test()
    else:
        config['plots_path_name'], config['models_path_name'] = set_train_path(config['plots_path_name'], 'plot')
        config['cache_path_name'] = set_test_path(config['cache_path_name'])
        
        ## save config file and log file
        save_config(config)
        write_log(config)
        
        train()

