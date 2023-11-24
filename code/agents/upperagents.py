from bisect import bisect
import datetime
from utils.utilize import config, plot_MFD, plot_critic_loss_cur_epis, plot_flow_MFD, plot_last_critic_loss, plot_reward, plot_actions, plot_critic_loss, plot_throughput, \
    plot_q_value, plot_q_value_improve, plot_accu, plot_computime
from utils.memory_buffer import MemoryBuffer, MemoryBuffer_Upper
from nn.critic import Critic, CriticUpper
from nn.actor import Actor
import scipy.optimize as opt
import numpy as np
from matplotlib.cbook import flatten
from keras.backend.common import epsilon
import sys
import os
from re import X
from matplotlib import pyplot as plt

from numpy.core.shape_base import hstack
from metric.metric import Metric
from envir.perimeter import Peri_Agent
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # kill warning about tensorflow
# from train_main import Env
# from tqdm import tqdm
# from utils.stats import gather_stats
# from utils.networks import tfSummary, OrnsteinUhlenbeckProcess

today = datetime.datetime.today()


class UpperAgents:
    """ A class of upper base agent
    """

    def __init__(self, tsc_peri, netdata):
        """ Initialization
        """
        self.outputfile = config['edge_outputfile_dir']
        self.netdata = netdata
        self.PN_road_length_tot, self.PN_edge_length = self._get_PN_road_length()

        # mode
        self.mode = config['mode']
        self.peri_action_mode = config['peri_action_mode']
        self.peri_mode = config['upper_mode']

        # RL agent
        self.act_dim = config['act_dim']
        self.gamma = config['gamma']
        self.lr_C = config['lr_C']
        self.lr_C_decay = config['lr_C_decay']
        self.lr_C_lb = config['lr_C_lb']
        self.episodes = config['total_episodes']
        self.batch_size = config['batch_size']
        self.reuse_time = config['reuse_time']
        self.explore_decay = config['explore_decay']
        self.epsilon = config['epsilon']
        self.a = 0

        # signal
        self.cycle_time = config['cycle_time']
        self.max_green = config['max_green']

        # states
        if self.peri_mode == 'Static':  # Static' # 'DDPG' #'DQN' # 'Expert''
            self.states = []
        elif self.peri_mode == 'Expert':
            self.states = ['accu', 'accu_buffer', 'future_demand']
        elif self.peri_mode == 'PI':
            self.states = ['accu']
        else:  # for RL
            self.states = config['states']
        self.env_dim = self._get_env_dim()

        # record
        self.metric_list = ['cul_obj', 'cul_reward', 'cul_penalty',
                            'upper_reward_epis', 'upper_penalty_epis',
                            'accu', 'speed', 'flow', 'TTD', 'PN_waiting', 'peri_entered_vehs',
                            'density_heter_PN', 'peri_waiting_mean', 'peri_waiting_tot']

        self.record_epis = {}
        for key in self.metric_list:
            self.record_epis[key] = []

        # record best
        self.best_num = 50
        self.best_epis = {}
        for key in self.metric_list:
            self.best_epis[key] = []
        # self.cul_obj, self.cul_reward, self.cul_penalty = [], [], [] # sum of each epis, list
        # self.reward_epis_all, self.penalty_epis_all = [], []
        # self.accu_episode = []
        # self.flow_episode = []
        # self.speed_episode = []
        # self.TTD_episode = []
        # self.PN_waiting_episode = []
        # self.entered_vehs_episode = []

        self.accu_crit_list = []
        self.mfdpara_list = []

        self.qvalue_list, self.qvalue_improve, self.computime_episode = [], [], []

        # buffer
        self.buffer = MemoryBuffer_Upper(config['buffer_size'], config['reward_delay_steps'], config['penalty_delay_steps'],
                                         config['reward_normalization'], config['multi-step'], config['gamma_multi_step'], config['sample_mode'])

        # metric
        self.info_interval = config['infostep']
        self.metric = Metric(self.cycle_time, self.info_interval, self.netdata)

        # perimeter
        self.tsc_peri = tsc_peri
        self.perimeter = Peri_Agent(self.tsc_peri)

        self.accu_crit = None

    def reset(self):
        ''' reset upper agent
        '''
        # perimeter
        self.perimeter = Peri_Agent(self.tsc_peri)
        self.peri_num = len(self.perimeter.info)
        self.info_update_index = 0
        self.perimeter.buffer_wait_vehs_tot = np.zeros(1)

        # metric
        self.metric.reset()

        # record of episod
        self.cumul_reward, self.cumul_penalty = 0, 0
        # self.action_excute_list = []
        # self.reward_epis, self.penalty_epis = [], []

        # init state
        self.old_state = [0] * self.env_dim

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        # critic_target = np.asarray(q_values)
        critic_target = q_values.copy()
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                # including multi-step
                critic_target[i] = rewards[i] + \
                    self.gamma**self.buffer.multi_step * q_values[i]
        return critic_target

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)

    def set_epsilon(self, n_jobs, e):
        ''' set the epsilon of each episode
            --- For [explore]: decay epsilon
        '''
        # if self.action_type == 'Test' or self.action_type == 'Expert':
        #     self.epsilon = 0

        # else:
        if self.action_type == 'Explore':
            self.epsilon = config['epsilon'] * \
                self.explore_decay**(e // n_jobs)
            self.epsilon = max([self.epsilon, config['explore_lb']])
        else:
            self.epsilon = 0

        # print(f'############# Upper level {e+1}: epsilon = {self.epsilon} ##############')

    def set_action_type(self, n_jobs, e):
        ''' set the action type of each episode:
        ---[train mode]:
            ---[Test]: The first episode of each round in multi-process
            ---[Expert]: 1. before 'expert_episode'
                         2. The second episode of each round in multi-process
        ---[test mode]:
            ---[Test]

        '''
        # FOR expert
        if self.peri_mode == 'Expert':
            self.action_type = 'Expert'
            return

        # For static
        if self.peri_mode == 'Static':
            self.action_type = 'Static'
            return

        # For Maxpressure
        if self.peri_mode == 'MaxPressure':
            self.action_type = 'MaxPressure'
            return

        # For PI-based controller
        if self.peri_mode == 'PI':
            self.action_type = 'PI'
            return

        # For RL
        if self.peri_mode in ['DQN', 'DDPG']:
            # training
            if self.mode == 'train':
                if n_jobs > 1 and e % n_jobs == 0:
                    self.action_type = 'Test'
                elif e < n_jobs * config['expert_episode'] or (n_jobs > 1 and e % n_jobs == 1):
                    self.action_type = 'Expert'
                else:
                    self.action_type = 'Explore'

            elif self.mode == 'test':
                self.action_type = 'Test'

        if self.peri_mode == 'C_DQN':
            # training
            if self.mode == 'train':
                if n_jobs > 1 and e % n_jobs == 0:
                    self.action_type = 'Test'
                else:
                    self.action_type = 'Explore'

            elif self.mode == 'test':
                self.action_type = 'Test'

        # print(f'############ Episode {e}: Action type is {self.action_type} #################')

    def implem_action_all(self, step):
        ''' get state, get action, implement action
        '''
        # no action for Static controller
        if self.peri_mode == 'Static' or self.peri_mode == 'MaxPressure':
            return

        # 1. get action
        self.a, is_expert = self.get_action_all(self.old_state)

        # 2.action coordination
        a = self._action_coordinate_transformation(self.max_green, self.a)

        # 3. assign the vehicle to each perimeter and obtain the phase duration
        green_time, red_time = self.perimeter.get_greensplit(a, step)

        # 4. set program
        self.perimeter.set_program(green_time, red_time)

    def get_action_all(self, old_state):
        ''' get action of upper level with different controllers
        --- Expert actions
        --- RL actions with exploration
        --- PI based action (permitted inflow)
        '''
        if self.action_type == 'Expert':
            a = self.get_expert_action(old_state)
            is_expert = True
            if self.peri_action_mode == 'decentralize':
                a = np.array([a] * self.act_dim)
            # print(f'###{e}: testing ####')

        elif self.action_type == 'PI':
            a = self.get_action(old_state)
            is_expert = False
            # if self.peri_action_mode =='decentralize':
            #     a = np.array([a] * self.act_dim)

        else:
            if self.peri_mode == 'C_DQN':
                a, is_expert, is_constraint = self.get_action(old_state)
                # print(f"actions: {a}")

                # 2. exploration
                if self.epsilon > 0 and not is_constraint:
                    a = self.explore(a)

            else:
                a, is_expert = self.get_action(old_state)
                # print(f"actions: {a}")

                # 2. exploration
                if self.epsilon > 0:
                    a = self.explore(a)

        return a, is_expert

    def get_expert_action(self, state):
        ''' expert action of descrete action
        '''
        if state[0] < 0.15:  # very low accumulation in the PN
            action = np.random.choice([0.5, 0.6, 0.7, 0.8])
        elif state[0] > 0.3:  # congested in the PN
            action = np.random.choice([0, 0.1])
            # print('1')
        else:
            action = np.random.choice([0.3, 0.4, 0.5])
        # elif state[-1]+state[-2] > 0.3:
        #     action = np.random.choice([0, 0.1])

        if state[2] > 0.5 and state[1] > 0.2:  # future demand is high and the PN is congested
            action = np.random.choice([0, 0.1])
            # print('2')

        if self.a >= 0.5:  # entered many vehicles on the last interval
            action = np.random.choice([0, 0.1])
            # print('3')

        return action


    def get_memory(self, done):
        ''' collect state, reward, penalty to get memory
        '''

        # new state
        self.new_state = self._get_state()

        # entered veh
        # entered_veh = self.metric.get_entered_veh_control_interval()
        # self.action_excute_list.append(entered_veh)

        # get reward
        reward = self._get_reward()
        # self.reward_epis.append(reward)

        # get penalty
        penalty = self._get_penalty()
        # self.penalty_epis.append(penalty)

        if self.peri_mode != 'Static':
            # self.memorize(self.old_state, self.a, reward, penalty, done, self.new_state, entered_veh)
            self._memorize(self.old_state, self.a, reward,
                          done, self.new_state,  penalty)

        # 3.5. Update current state
        self.old_state = self.new_state

        # 3.6. Calculate one-step objective (reward+penalty)
        # self.cumul_reward += reward  # reward along this episode
        # self.cumul_penalty += penalty

    def record(self, upper_metric):
        ''' record performace of each epis
        '''

        for key in self.metric_list:
            self.record_epis[key].append(list(upper_metric[key]))

    def train(self, cur_epis, n_jobs):
        '''replay to train the NN of upper level
        '''
        if self.peri_mode in ['DQN', 'DDPG', 'C_DQN'] and \
                self.buffer.count_CA >= self.batch_size:

            print(
                f'####### Upper Training with {self.buffer.count_CA} memory ######')

            loss_epis = []
            for k in range(int(self.buffer.count_CA * self.reuse_time // self.batch_size)):

                loss = self.replay()
                loss_epis.append(loss)

            # lr decay
            self.lr_decay()
            # plot critic loss and last_update critic loss
            self.plot_train_loss(loss_epis, cur_epis//max(n_jobs, 1))

    def get_best_episode(self, upper_metric):

        # get index
        insert_idx = bisect(self.best_epis['cul_obj'], upper_metric['cul_obj'])

        if len(upper_metric['flow']) == (config['max_steps'] // config['cycle_time']):
            for key in self.metric_list:
                self.best_epis[key].insert(insert_idx, list(upper_metric[key]))
                self.best_epis[key] = self.best_epis[key][-self.best_num:]

# MFD critical state
    def fit_mfd(self, accu, outflow):
        coef = np.polyfit(accu, outflow, 3)  # 最小二乘标定整体路网
        yn = np.poly1d(coef)  # 拟合的多项式
        self.mfdpara = coef
        self.mfdpara_list.append(coef)

        # x1 = np.arange(0, max(accu)*1.1)
        # y1 = [yn(i) for i in x1]

        # # 基本图对比
        # plt.xlabel('acc(veh)')
        # plt.ylabel('outflow(veh/h)')
        # plt.title('mfd')
        # plt.scatter(accu, outflow, label='data points')
        # plt.plot(x1, y1, label='curve')
        # plt.legend()
        # plt.savefig("mfdcurve.png")

        return coef

    def func(self, x, sign=1):
        a, b, c, d = self.mfdpara * sign
        return a * x ** 3 + b * x ** 2 + c * x + d

    def cal_ncritic(self):
        bnd = [(0, self.accu_crit)]  # bound of accumulation
        res = opt.minimize(fun=self.func, x0=np.array(
            [0]), bounds=bnd, args=(-1,))  # get the critical accumulation

        ncrit, = res.x  # unpack

        Gneq = self.func(ncrit)   # set point

        return ncrit, Gneq

    def update_ncritic(self, ncrit):
        ''' determine the new n_critic for learning
        '''
        self.accu_crit_list.append(self.accu_crit)

        if ncrit == self.accu_crit:
            self.accu_crit += self.accu_step

        else:
            if ncrit <= self.accu_crit - self.accu_step:
                self.accu_crit = self.accu_crit - self.accu_step*0.5  # descend
                self.accu_step = self.accu_step*0.9  # decay step

# helper funcs
    def _get_PN_road_length(self):

        ##
        PN_road_length_tot = 0
        PN_edge_length = []

        for edge in config['Edge_PN']:
            PN_road_length_tot += self.netdata['edge'][str(edge)]['length'] * len(
                self.netdata['edge'][str(edge)]['lanes'])
            PN_edge_length.append(self.netdata['edge'][str(edge)]['length'])

        return PN_road_length_tot, PN_edge_length

    def _get_env_dim(self):
        ''' get the dimension of the input state
        '''
        local_state = ['down_edge_occupancy', 'buffer_aver_occupancy']

        env_dim = 0
        for state in self.states:
            if state in local_state:
                env_dim += len(config['Peri_info'])
            else:
                env_dim += 1

        return env_dim * config['state_steps']

    def _action_coordinate_transformation(self, upper_bound, action):
        """ Conduct the coordinate transform given the bounded action
            From green ratio [0,1] to the green time for each perimeter signal
        """
        # [-1 1]
        # medium =  (upper_bound -lower_bound) / 2 +lower_bound
        # action  = medium + action * (medium - lower_bound)
        # assert (action>=0).all()==True, 'the action is less than 0'
        # assert (action<=1).all()==True, 'the action is greater than 1'
        # [0,1]
        if self.action_type == 'PI':
            ''' Translate the network inflow to the green time
            '''

            action = action / config['flow_rate']

            return action

        if self.perimeter.peri_action_mode == 'centralize':  # centralized actions
            action = action * upper_bound * self.peri_num

        else:  # decentralized actions
            action = action * upper_bound

        # print(f"actual ation: {action}")
        return action

# upper-level metric
    def get_metric_each_epis(self, edge_data):
        ''' get upper-level metric after each episode --- entire simulation/episode '''

        metric = {}
        '''flow'''
        # flow1
        # flow_epis = np.sum((edge_sampledSeconds/self.info_interval * edge_speed*3.6)/ (self.PN_road_length_tot/1000), axis=1)

        # flow2
        flow_epis = np.mean(
            edge_data['speed'] * edge_data['density'] * 3.6, axis=1)  # veh/h
        metric['flow'] = flow_epis
        # flow 3 bad
        # TTD = accu * mean_speed* 100  # nveh*m/sec*sec = veh*m
        # flow_3 = TTD / self.PN_road_length_tot / (100/3600)

        '''accumulation'''
        # accu
        accu_epis = np.sum(
            (edge_data['sampledSeconds'] / self.info_interval), axis=1)
        metric['accu'] = accu_epis

        '''speed'''
        # speed
        weighted_speed = np.multiply(edge_data['speed'], np.array(
            self.PN_edge_length)[:, np.newaxis].T)
        speed_epis = np.sum(weighted_speed, axis=1)/sum(self.PN_edge_length)
        metric['speed'] = speed_epis

        '''TTD'''
        # Total travel distance
        TTD_epis = np.sum(edge_data['sampledSeconds']
                          * edge_data['speed'], axis=1)/1e3  # (km)
        metric['TTD'] = TTD_epis

        '''waiting_time'''
        # PN waiting time
        PN_waiting_epis = np.sum(edge_data['PN_waiting'], axis=1)
        metric['PN_waiting'] = PN_waiting_epis

        '''entered_vehs'''
        peri_entered_vehs = np.sum(edge_data['peri_entered_vehs'], axis=1)
        if len(peri_entered_vehs) < len(metric['flow']):
            len_short = len(metric['flow']) - len(peri_entered_vehs)
            peri_entered_vehs = np.concatenate(
                [peri_entered_vehs, np.zeros(len_short)])
        metric['peri_entered_vehs'] = peri_entered_vehs

        ''' PN density heterogenity'''
        density_PN = edge_data['density']/300
        mean_density = np.mean(density_PN, axis=1)
        mean_density = np.expand_dims(mean_density, axis=1)

        # density rmse
        metric['density_heter_PN'] = np.sqrt(
            np.mean((density_PN-mean_density)**2, axis=1))

        ''' PN+buffer density heterogenity'''
        # density_PN_buffer = np.hstack([edge_data['density'], edge_data['buffer_density']])
        # density_PN_buffer = density_PN_buffer/300
        # mean_density = np.mean(density_PN_buffer, axis=1)
        # mean_density = np.expand_dims(mean_density, axis=1)

        # metric['density_heter_PN_buffer'] = np.sqrt(np.mean((density_PN_buffer-mean_density)**2, axis=1))

        ''' peri waiting time'''
        # average waiting time for each veh within each cycle time
        metric['peri_waiting_mean'] =np.sum(edge_data['peri_waiting_tot'],1) / (np.sum(edge_data['peri_sampledSeconds'],1)/self.cycle_time)

        # tot waiting time for each perimeter
        metric['peri_waiting_tot'] = edge_data['peri_waiting_tot']


        return metric

    def get_metric_each_interval(self):
        ''' update metric for each interval
        '''
        self.metric.update_info_interval(
            self.info_update_index, self.perimeter.info, self.outputfile)
        self.info_update_index += 1

# RL helper funcs
    def _get_reward(self):
        ''' Collect reward for each upper simulation step
        '''
        # 1. production within control interval ( speed * veh )
        # _, production_control_interval  = self.metric.get_PN_speed_production()
        # # print(f'Production = {production_control_interval}')
        # reward = production_control_interval / config['production_control_interval_max']
        reward = 0.0001
        return reward

    def _get_penalty(self):
        ''' Collect reward for each simulation step
        '''
        # halveh_buffer, halveh_PN = self.metric.get_halting_vehs()
        # penalty = -(halveh_buffer/1000)**2

        # penalty = np.clip(penalty, -4, 0)
        penalty = -0.001
        return penalty

    def _get_state(self):
        ''' obtain state of the upper controller
        '''
        state = []

        # get states
        for state_type in self.states:

            # 1.accumulation of PN after normalization
            if state_type == 'accu':
                accu_PN, _ = self.metric.get_accu(info_inter_flag=True)
                accu_PN = accu_PN / config['accu_max']  # normalize
                # self.state_dict['accu'].append(accu_PN)
                state.append(accu_PN)

            # 2. accumulation of buffer after normalization
            elif state_type == 'accu_buffer':
                _, accu_buffer = self.metric.get_accu(info_inter_flag=True)
                accu_buffer = accu_buffer / \
                    config['accu_buffer_max']  # normalize
                # self.state_dict['accu_buffer'].append(accu_buffer)
                state.append(accu_buffer)

            # 3. occupancy of the downlane in the perimeter
            elif state_type == 'down_edge_occupancy':
                self.perimeter.get_down_edge_occupancy()
                # self.state_dict['down_edge_occupancy'].append(
                #     self.Perimeter.down_edge_occupancy)
                state.extend(self.perimeter.down_edge_occupancy)

            # 4. average occupancy of the buffer links
            elif state_type == 'buffer_aver_occupancy':
                self.perimeter.get_buffer_average_occupancy()
                state.extend(self.perimeter.buffer_average_occupancy)
                # self.state_dict['buffer_aver_occupancy'].append(
                #     self.Perimeter.buffer_average_occupancy)

            # 5. demand of next step
            elif state_type == 'future_demand':
                cycle_index = int(self.info_update_index)
                demand_nextstep = self.metric.get_demand_nextstep(cycle_index)
                demand_nextstep = demand_nextstep / config[
                    'Demand_state_max']  # normalization
                # self.state_dict['future_demand'].append(demand_nextstep)
                state.append(demand_nextstep)

            # 6. entered vehicles from perimeter
            elif state_type == 'entered_vehs':
                entered_veh = self.metric.get_entered_veh_control_interval()
                entered_veh = entered_veh / \
                    config['entered_veh_max']  # normalize
                # self.state_dict['entered_vehs'].append(entered_veh)
                state.append(entered_veh)

            # 7. network mean speed
            elif state_type == 'network_mean_speed':
                network_mean_speed, _ = self.metric.get_PN_speed_production(
                    info_inter_flag=True)
                network_mean_speed = network_mean_speed / \
                    config['network_mean_speed_max']  # normalize
                # self.state_dict['network_mean_speed'].append(network_mean_speed)
                state.append(network_mean_speed)

            # 8. network PN halting vehicles
            elif state_type == 'network_halting_vehicles':
                _, PN_halt_vehs = self.metric.get_halting_vehs(
                    info_inter_flag=True)
                PN_halt_vehs = PN_halt_vehs/config['PN_halt_vehs_max']
                # self.state_dict['network_halting_vehicles'].append(PN_halt_vehs)
                state.append(PN_halt_vehs)

            # 9. buffer halting vehicles
            elif state_type == 'buffer_halting_vehicles':
                buffer_halt_vehs, _ = self.metric.get_halting_vehs(
                    info_inter_flag=True)
                # print(f'buffer_halt_vehs = {buffer_halt_vehs}')
                buffer_halt_vehs = buffer_halt_vehs / \
                    config['buffer_halt_vehs_max']
                # self.state_dict['network_halting_vehicles'].append(buffer_halt_vehs)
                state.append(buffer_halt_vehs)

        return state

    def _memorize(self, state, action, reward,  done, new_state, penalty):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state, penalty)




class DQN(UpperAgents):
    ''' A class of Deep Q-network for descrete action
    '''

    def __init__(self, tsc_peri, netdata,  tau=config['tau']):
        super().__init__(tsc_peri, netdata)

        self.action_num = 11
        self.actions = np.around(np.linspace(0, 1, self.action_num), 1)
        self.critic = CriticUpper(self.env_dim, 1, self.lr_C, tau)

        self.critic_udpate_num = 0
        self.target_update_freq = config['target_update_freq']

        self.best_obj = -1e5

    def get_action(self, s, training=False):
        ''' get action using critic eval_network
            1. In simulation: the expert action may be triggered.
            2. In training:  the next-action must be generated by eval_network
        '''
        is_expert = False

        # if training:
        if training or (s[0] < 10 and s[-1]+s[-2] < 10):

            # get q-values for all possible actions
            input_state = np.array([s] * len(self.actions))
            input_action = np.array(self.actions).reshape(-1, 1)
            # One-Hot encode actions
            # input_action = np.eye(self.action_num)

            q_value = self.critic.eval_predict([input_state, input_action])

            # get the action by argmax Q-value
            # If the max Q-value more than one, randomly chose one action among them
            max_q_index, = np.where(q_value.flatten() == q_value.max())
            action = self.actions[np.random.choice(max_q_index)]
            # print(f'optimal action ={action}')

            if training:
                # print(sorted(q_value.flatten()))
                # print(input_state)
                # print(q_value)
                # print(f'### optimal action = {action} ### max q_value = {max(q_value)}')
                pass

        else:
            action = self.get_expert_action(s)
            is_expert = True

        if training:
            return action
        else:
            return action, is_expert

    def get_batch_action(self, states):
        ''' get batch action using target_network for replay
        '''
        actions = []
        # input_action = np.array(self.actions).reshape(-1, 1)
        for state in states:
            # input_state = np.array([state] * len(self.actions))
            # q_value = self.critic.target_predict([input_state, input_action])

            actions.append(self.get_action(state, True))

        return np.array(actions)

    def explore(self, action):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)

        # print(f"action with noise: {action}")

        return action

    def replay(self):
        ''' DQN replay
        '''
        # print('############ before train ##############')
        # W, target_W = self.critic.model.get_weights(), self.critic.target_model.get_weights()
        # print('##Evaluate network')
        # print(W)
        # print('##Target network')
        # print(target_W)

        # Sample experience from buffer
        states, actions, rewards, dones, new_states = self.sample_batch(
            self.batch_size)

        # rewards = rewards * 10

        # obtain new action
        new_actions = self.get_batch_action(new_states).reshape(-1, 1)

        # Predict target q-values using target networks
        # action_index = np.searchsorted(self.actions, new_actions).flatten()
        # new_actions = np.eye(self.action_num)[action_index]
        q_values = self.critic.target_predict([new_states, new_actions])

        # print(self.actor.target_predict(new_states))

        # check nan of Q values
        if np.isnan(q_values).any():
            raise Exception('There is NaN in the q-value')

        # Compute critic target
        critic_target = self.bellman(rewards, q_values, dones)

        # Train both networks on sampled batch, update target networks
        _, loss = self.update_models(states, actions, critic_target)

        return loss
        # print_value  = np.hstack([rewards.reshape(-1,1), q_values, critic_target])

    def update_models(self, states, actions, critic_target):
        """ Update critic networks from sampled experience
            The target network updates are delayed
        """
        # print('STATES')
        # print(states)
        # print('ACITONS')
        # print(actions)
        # print('TARGET')
        # print(critic_target)

        # Train critic
        # action_index = np.searchsorted(self.actions, actions).flatten()
        # actions = np.eye(self.action_num)[action_index]
        loss = self.critic.train_on_batch(states, actions, critic_target)
        self.critic_udpate_num += 1

        # Transfer weights to target networks at rate Tau
        soft_update_idx = False  # indicate the soft_update completion
        if self.critic_udpate_num % self.target_update_freq == 0:
            print('########## soft update')
            self.critic.transfer_weights()
            soft_update_idx = True

        return soft_update_idx, loss

    def plot_train_loss(self, loss_epis, cur_epis):
        ''' plot loss during trainning
        '''
        # 1. critic loss
        plot_critic_loss(self.critic.qloss_list, 'Upper', self.mode)

        # 2. critic loss of last step
        self.critic.last_qloss_list.append(self.critic.qloss_list[-1])
        plot_last_critic_loss(self.critic.last_qloss_list, 'Upper')

        # print(loss_epis)
        # 3. critic loss of each epis
        plot_critic_loss_cur_epis(loss_epis, cur_epis, self.lr_C)

    def save_weights(self, path):
        self.critic.save(path)

    def load_weights(self, path):
        self.critic.load_weights(path)

    def lr_decay(self):
        self.lr_C = self.lr_C * self.lr_C_decay
        self.lr_C = max(self.lr_C, self.lr_C_lb)
        print(f'#######  current critic_lr is {self.lr_C}')

    def save_best_critic(self, obj):
        if obj > self.best_obj:
            self.critic.save(config['models_path_name'], best=True)
            self.best_obj = obj


class C_DQN(UpperAgents):
    ''' A class of Deep Q-network for descrete action
        Constrained DQN
    '''

    def __init__(self, tsc_peri, netdata,  tau=config['tau']):
        super().__init__(tsc_peri, netdata)

        self.action_num = 11
        self.actions = np.around(np.linspace(0, 1, self.action_num), 1)
        self.critic = CriticUpper(self.env_dim, 1, self.lr_C, tau)

        self.accu_crit = config['accu_max'] * 0.1
        self.accu_step = 50
        self.accu_epsilon = 50

        self.critic_udpate_num = 0
        self.target_update_freq = config['target_update_freq']

    def get_action(self, s, training=False):
        ''' get action using critic eval_network
            1. In simulation: the expert action may be triggered.
            2. In training:  the next-action must be generated by eval_network
        '''
        is_expert = False
        is_constraint = False
        # if training:
        if True:

            # get q-values for all possible actions
            input_state = np.array([s] * len(self.actions))
            input_action = np.array(self.actions).reshape(-1, 1)
            # One-Hot encode actions
            # input_action = np.eye(self.action_num)

            q_value = self.critic.eval_predict([input_state, input_action])

            # get the action by argmax Q-value
            # If the max Q-value more than one, randomly chose one action among them
            max_q_index, = np.where(q_value.flatten() == q_value.max())
            action = self.actions[np.random.choice(max_q_index)]
            # print(f'optimal action ={action}')

            if s[0]*config['accu_max'] > self.accu_crit+self.accu_epsilon:
                action = 0
                is_constraint = True

            if training:
                # print(sorted(q_value.flatten()))
                # print(input_state)
                # print(q_value)
                # print(f'### optimal action = {action} ### max q_value = {max(q_value)}')
                pass

        else:
            action = self.get_expert_action(s)
            is_expert = True

        if training:
            return action
        else:
            return action, is_expert, is_constraint

    def get_batch_action(self, states):
        ''' get batch action using target_network for replay
        '''
        actions = []
        # input_action = np.array(self.actions).reshape(-1, 1)
        for state in states:
            # input_state = np.array([state] * len(self.actions))
            # q_value = self.critic.target_predict([input_state, input_action])

            actions.append(self.get_action(state, True))

        return np.array(actions)

    def explore(self, action):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)

        # print(f"action with noise: {action}")

        return action

    def replay(self):
        ''' DQN replay
        '''
        # print('############ before train ##############')
        # W, target_W = self.critic.model.get_weights(), self.critic.target_model.get_weights()
        # print('##Evaluate network')
        # print(W)
        # print('##Target network')
        # print(target_W)

        # Sample experience from buffer
        states, actions, rewards, dones, new_states = self.sample_batch(
            self.batch_size)

        # rewards = rewards * 10

        # obtain new action
        new_actions = self.get_batch_action(new_states).reshape(-1, 1)

        # Predict target q-values using target networks
        # action_index = np.searchsorted(self.actions, new_actions).flatten()
        # new_actions = np.eye(self.action_num)[action_index]
        q_values = self.critic.target_predict([new_states, new_actions])

        # print(self.actor.target_predict(new_states))

        # check nan of Q values
        if np.isnan(q_values).any():
            raise Exception('There is NaN in the q-value')

        # Compute critic target
        critic_target = self.bellman(rewards, q_values, dones)

        # Train both networks on sampled batch, update target networks
        _, loss = self.update_models(states, actions, critic_target)

        return loss
        # print_value  = np.hstack([rewards.reshape(-1,1), q_values, critic_target])
        # print('States      Actions')
        # print(np.hstack([states, actions.reshape(-1,1)]))
        # print('New states')
        # print(new_states)
        # print('rewards      q_values       critic_target')
        # print(print_value)

        # print('############ after train ##############')
        # W, target_W = self.critic.model.get_weights(), self.critic.target_model.get_weights()
        # print('##Evaluate network')
        # print(W)
        # print('##Target network')
        # print(target_W)

        # Decaying learning rate
        # self.lr_C = self.lr_C * self.lr_C_decay

    def update_models(self, states, actions, critic_target):
        """ Update critic networks from sampled experience
            The target network updates are delayed
        """
        # print('STATES')
        # print(states)
        # print('ACITONS')
        # print(actions)
        # print('TARGET')
        # print(critic_target)

        # Train critic
        # action_index = np.searchsorted(self.actions, actions).flatten()
        # actions = np.eye(self.action_num)[action_index]
        loss = self.critic.train_on_batch(states, actions, critic_target)
        self.critic_udpate_num += 1

        # Transfer weights to target networks at rate Tau
        soft_update_idx = False  # indicate the soft_update completion
        if self.critic_udpate_num % self.target_update_freq == 0:
            print('########## soft update')
            self.critic.transfer_weights()
            soft_update_idx = True

        return soft_update_idx, loss

    def plot_train_loss(self, loss_epis, cur_epis):
        ''' plot loss during trainning
        '''
        # 1. critic loss
        plot_critic_loss(self.critic.qloss_list, 'Upper', self.mode)

        # 2. critic loss of last step
        self.critic.last_qloss_list.append(self.critic.qloss_list[-1])
        plot_last_critic_loss(self.critic.last_qloss_list)

        # print(loss_epis)
        # 3. critic loss of each epis
        plot_critic_loss_cur_epis(loss_epis, cur_epis, self.lr_C)

    def save_weights(self, path):
        self.critic.save(path)

    def load_weights(self, path):
        self.critic.load_weights(path)

    def lr_decay(self):
        self.lr_C = self.lr_C * self.lr_C_decay
        self.lr_C = max(self.lr_C, self.lr_C_lb)
        print(f'#######  current critic_lr is {self.lr_C}')


class DDPG(UpperAgents):
    ''' A class of deep deterministic policy gradient algrithm for continuous action
    '''

    def __init__(self, tsc_peri, netdata, tau=config['tau']):
        super().__init__(tsc_peri, netdata)

        self.lr_A = config['lr_A']
        self.critic = CriticUpper(
            self.env_dim, self.act_dim, self.lr_C, tau, action_grad=True)
        self.actor = Actor(self.env_dim, self.act_dim, 1, self.lr_A, tau)

        self.critic_udpate_num = 0
        self.target_update_freq = config['target_update_freq']

    def get_action(self, s, training=False):
        """ Use the actor to predict value
        """

        is_expert = False

        if True:
            action = self.actor.predict(s)[0]

        else:
            action = self.get_expert_action(s)
            is_expert = True

        return action, is_expert

    def get_expert_action(self, state):
        ''' expert action of descrete action
        '''
        if state[0] < 0.15:  # very low accumulation in the PN
            action = np.random.uniform(0.7, 0.9)
        elif state[0] > 0.3:  # congested in the PN
            action = np.random.uniform(0, 0.1)
        elif state[1] > 0.3:
            action = np.random.uniform(0, 0.1)

        elif state[2] > 0.5 and state[1] > 0.2:
            action = np.random.uniform(0, 0.1)

        elif state[3] > 0.5:  # future demand is high and the PN is congested
            action = np.random.uniform(0, 0.1)

        else:
            action = np.random.uniform(0.3, 0.5)

        return action  # np.array([action])

    def explore(self, action):
        # self.act_range)  # gaussian noise
        action = np.clip(np.random.normal(action, self.epsilon), -0.5, 0.5)
        action = action + 0.5

        # print(f"action with noise: {action}")
        if len(action) == 1:
            return float(action)
        else:
            return action

    def update_models(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience, k for delayed updates for the target networks
        """
        # print('STATES')
        # print(states)
        # print('ACITONS')
        # print(actions)
        # print('TARGET')
        # print(critic_target)

        # Train critic
        loss = self.critic.train_on_batch(states, actions, critic_target)

        # Q-Value Gradients under Current Policy
        # actions = self.actor.model.predict(states)
        grads = self.critic.gradients(states, actions)

        # Train actor
        self.actor.train(states, actions, np.array(
            grads).reshape((-1, self.act_dim)))

        self.critic_udpate_num += 1

        # Transfer weights to target networks at rate Tau
        soft_update_idx = False  # indicate the soft_update completion
        if self.critic_udpate_num % self.target_update_freq == 0:
            self.actor.transfer_weights()
            self.critic.transfer_weights()

            soft_update_idx = True

        return soft_update_idx, loss

    def replay(self):
        # Sample experience from buffer
        states, actions, rewards, dones, new_states, _ = self.sample_batch(
            self.batch_size)
        # states = states[:, np.newaxis]
        # actions = actions.reshape(-1)

        # Predict target q-values using target networks
        q_values = self.critic.target_predict(
            [new_states, self.actor.target_predict(new_states)])
        # print(self.actor.target_predict(new_states))

        if np.isnan(q_values).any():
            raise Exception('There is NaN in the q-value')

        # Compute critic target
        critic_target = self.bellman(rewards, q_values, dones)
        # Train both networks on sampled batch, update target networks
        soft_update_idx, loss = self.update_models(
            states, actions, critic_target)

        if soft_update_idx == True:
            ''' check the q_improvement after soft update
            '''
            # get q_value of the same states after updates
            q_values_new = self.critic.target_predict(
                [new_states, self.actor.target_predict(new_states)])
            q_value_improve = np.mean(q_values_new) - np.mean(q_values)

            # print(q_values)
            print(f'old q_values less than 0: {sum( i <0 for  i in q_values)}')
            print(
                f'new q_values less than 0: {sum( i <0 for  i in q_values_new)}')
            print(f'old average q_values : {np.mean(q_values)}')
            print(f'new average q_values : {np.mean(q_values_new)}')
            print(f"q_value improvement:{q_value_improve}")
            self.qvalue_list.append(np.mean(q_values_new))
            self.qvalue_improve.append(np.mean(q_value_improve))
            # plot_q_value_improve(self.qvalue_improve)

        return loss
        # plot_critic_loss(self.critic.qloss_list)

    def plot_train_loss(self, loss_epis, cur_epis):
        ''' plot loss during trainning
        '''
        # 1. critic loss
        plot_critic_loss(self.critic.qloss_list, 'Upper', self.mode)

        # 2. critic loss of last step
        self.critic.last_qloss_list.append(self.critic.qloss_list[-1])
        plot_last_critic_loss(self.critic.last_qloss_list)

        # 3. actor loss (q-values)
        plot_q_value(self.qvalue_list)

        # 4. actor loss improvement
        plot_q_value_improve(self.qvalue_improve)

        # 5. critic loss of each epis
        plot_critic_loss_cur_epis(loss_epis, cur_epis, self.lr_C)

    def save_weights(self, path):
        # formatted_today = today.strftime('%m%d')
        # print(formatted_today)
        # path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path):
        self.critic.load_weights(path)
        self.actor.load_weights(path)

    def lr_decay(self):
        self.lr_C = self.lr_C * self.lr_C_decay
        self.lr_C = max(self.lr_C, self.lr_C_lb)
        print(f'#######  current critic_lr is {self.lr_C}')

        self.lr_A = self.lr_A * self.lr_C_decay
        self.lr_A = max(self.lr_A, self.lr_C_lb)
        print(f'#######  current actor_lr is {self.lr_C}')


class Static(UpperAgents):
    ''' Static with default plans
    '''

    def __init__(self, tsc_peri, netdata):
        super().__init__(tsc_peri, netdata)

        # self.peri_mode = 'Static'

    def save_weights(self, path):
        pass


class Expert(UpperAgents):
    '''  Expert plans
    '''

    def __init__(self, tsc_peri, netdata):
        super().__init__(tsc_peri, netdata)

    def save_weights(self, path):
        pass


class MFD_PI(UpperAgents):
    '''  PI based perimeter control with the use of MFD 
         Reference: Keyven-Ekbatani et.al 2019      
    '''

    def __init__(self, tsc_peri, netdata):
        super().__init__(tsc_peri, netdata)

        # controller settings
        self.accu_crit = config['accu_critic']  # set-point for the controller
        self.K_p = config['K_p']  # proportional gains
        self.K_i = config['K_i']  # integral gains

        self.q_last = 0   # calculated network inflow (the action) of last step
        self.q_record = [] # record of actions
        self.accu_last = 0

        # the maximum inflow of the network each step
        self.q_max = config['flow_rate'] * \
            self.max_green * len(config['Peri_info'])

    def get_action(self, s):
        ''' PI control using keyvan-Ekbatani 2019, Page8, Eq 10
        '''
        # 1. get states
        accu = s[0] * config['accu_max']   # current accu

        # 2.calculate control
        a = self.q_last - self.K_p * \
            (accu - self.accu_last) + self.K_i*(self.accu_crit-accu)

        # 3. inflow constraints
        a = min(a, self.q_max)
        a = max(a, 0)

        # 4. update and record
        self.q_last = a  # update the inflow
        self.q_record.append(a)  # record the actions
        self.accu_last = accu  # update the accu

        return a

    def save_weights(self, path):
        pass
