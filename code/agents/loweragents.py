import os
import pickle


# from code.utils.light_prepare import CAPACITY

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  # kill warning about tensorflow
import sys
import numpy as np

# from tqdm import tqdm
from nn.actor import Actor
from nn.critic import Critic, CriticLower
# from utils.stats import gather_stats
# from utils.networks import tfSummary, OrnsteinUhlenbeckProcess
from utils.memory_buffer import MemoryBuffer, MemoryBuffer_Lower
from utils.utilize import config, plot_MFD, plot_last_critic_loss, plot_reward, plot_actions, plot_critic_loss, plot_throughput, \
plot_q_value, plot_q_value_improve, plot_accu, plot_computime
import timeit
import traci
import datetime

today = datetime.datetime.today()


class LowerAgents:
    """ A class of lower base agent
    """
    def __init__(self, tsc, netdata):
        """ Initialization
        """
        self.outputfile = config['lane_outputfile_dir']

        ## network config
        self.tsc = tsc
        self.netdata = netdata
        self.perimeter_light = list(config['Peri_info'].keys())
        self.lower_mode = config['lower_mode']
        self.upper_mode = config['upper_mode']
        self.mode = config['mode']
        self.controled_light = self._get_control_light_list(self.netdata['tls']) ## get light list

        ## signal config
        self.control_interval = config['control_interval']
        self.yellow_duration = config['yellow_duration']
        self.green_max_duration = config['max_green_duration']
        self.phase_num = 10
        self.inlane_num = 12
                
        ## light config
        if self.lower_mode!='FixTime':
            self.tls_config = config['tls_config_name']
            self._get_signal_configure() ## get signals 
            self.batch_phase_matrix, self.batch_phase_masks, self.batch_signals\
                = self._get_batch_signal_matrix()

        ##
        self.reward_norm = config['lower_reward_max']
       
        # record
        self.metric_list = ['network_delay_step', 'network_perveh_delay_step',\
            'tsc_delay_step', 'tsc_perveh_delay_step','tsc_perveh_delay_mean', 'tsc_through_step']

        self.record_epis = {}
        for key in self.metric_list:
            self.record_epis[key] = []

        ''' memory '''
        self.init_memory()


        # self.buffer = MemoryBuffer_Lower(config['buffer_size_lower'], 0, 0,\
        #     config['reward_normalization'], 1, self.gamma, config['sample_mode'])

    def reset(self):
        ''' reset of the lower agents
        '''
        ## 1. init tsc metric
        if self.lower_mode!='FixTime':
            for tsc in self.tsc.values(): 
                ## phase
                tsc.cur_phase = []
                tsc.prev_phase = []

                ## get the index of first phase
                row_sum = np.sum(tsc.matrix, axis=1)
                tsc.cur_phase_idx = np.where(row_sum != 0)[0][0]
                
                ## reward
                tsc.ep_rewards = []
                tsc.ep_delay_edge = []
                tsc.veh_num = []

                ## phase time
                tsc.cur_phase_time = 0
                tsc.phase_time_list = []
                tsc.phase_list = []

        ## 2. init state
        self.old_state = self.get_state()


        ## 3. 
        self.cumul_reward = 0

        ## 4. 
        self.action_index = 0

    def store_memory(self, done):
        ''' collect state, reward, penalty to store memory
        '''

        ## 1. get new state
        self.new_state = self.get_state()


        ## 2. memorize
        if self.lower_mode == 'OAM':
            self._memorize(self.old_state, self.action_index, self.new_state)
  
        ## 3. Update current state
        self.old_state = self.new_state

    def set_action_type(self, n_jobs, e):
        ''' set the action type of each episode:
        ---[train mode]:
            ---[Test]: The first episode of each round in multi-process
            ---[Expert]: 1. before 'expert_episode'
                         2. The second episode of each round in multi-process
        ---[test mode]:
            ---[Test]
        
        '''

        ## For RL
        if self.mode == 'train':
            if n_jobs>1 and e % n_jobs == 0:
                self.action_type = 'Test'
            else:
                self.action_type = 'Explore'
        elif self.mode == 'test':
            self.action_type = 'Test'

    def set_epsilon(self, n_jobs, e):
        ''' set epsilon for each epis
        '''
        

        if self.lower_mode == 'OAM' and self.action_type =='Explore':
            self.epsilon = config['epsilon_lower'] * self.explore_decay**(e // n_jobs)
            self.epsilon = max([self.epsilon, config['explore_lb']])
        else:
            self.epsilon = 0



        # print(f'############# Lower level {e+1}: epsilon = {self.epsilon} ##############')
        

    def train(self, cur_epis, n_jobs):
        '''replay to train the NN of lower level
        ''' 
        if self.lower_mode in ['OAM'] and \
            self.buffer_count >= self.batch_size:

            print(f'####### Lower Training with {self.buffer_count} memory ######')
            loss_epis = self.TD_learning()

            self._plot_train_loss(loss_epis,cur_epis)

    def init_memory(self):
        self.memory = {'state': [], 'action': [], 'reward': [], 'next_state': []}
        self.buffer_count = 0

    def _plot_train_loss(self, loss_epis, cur_epis):
        ''' plot loss during trainning
        '''
        # 1. critic loss
        plot_critic_loss(self.critic.qloss_list, 'lower', self.mode)

        # 2. critic loss of last step
        self.critic.last_qloss_list.append(self.critic.qloss_list[-1])
        plot_last_critic_loss(self.critic.last_qloss_list, 'lower')

        # print(loss_epis)
        # 3. critic loss of each epis
        # plot_critic_loss_cur_epis(loss_epis, cur_epis, self.lr_C)
        

    def save_weights(self, path):
        if self.lower_mode in ['OAM']:

            self.critic.save(path)

    def load_weights(self, path):
        if self.lower_mode in ['OAM']:

            self.critic.load_weights(path)


## for init class of lower level
    def _get_batch_signal_matrix(self):
        ''' get the batch signal matrix, batch phase mask of all the controlled lights
        '''
        batch_phase_matrix = []
        batch_phase_mask = []
        batch_signals = []
        for tl_id in self.controled_light:
            ''' capacity embedded'''
            matrix = self.tsc[tl_id].matrix *self.tsc[tl_id].capacity 
            mask = self.tsc[tl_id].mask 
            signal = self.tsc[tl_id].signals 

            batch_phase_matrix.append(matrix)
            batch_phase_mask.append(mask)
            batch_signals.append(signal)
            # batch_capacity.append(capacity)



        return np.array(batch_phase_matrix),  np.array(batch_phase_mask), np.array(batch_signals)

    def _get_control_light_list(self,inters):
        ''' get the list of the controled light of lower lever
            [to fix the order]
        '''
        controled_light_list = []
        for inter in inters.values():
            if inter['tl_id'] not in self.perimeter_light:
                controled_light_list.append(inter['tl_id'])

        if self.upper_mode in ['OAM', 'MaxPressure']:
            controled_light_list.extend(self.perimeter_light)
        
        return controled_light_list

    def _get_signal_configure(self):
        ''' import phase action matrix and signal settings
        '''
        ## load signal configurations
        with open(self.tls_config, 'rb') as f:
            MATRIX, SIGNAL, CAPACITY = pickle.load(f)
            

        for t_id, t_value in self.tsc.items():

            ## set matrix and signal
            t_value.matrix = MATRIX[t_id]
            t_value.signals = SIGNAL[t_id] 
            t_value.capacity = CAPACITY[t_id] 

            ## completion of the matrix
            if t_value.matrix.shape[0]<self.phase_num:
                complete = np.zeros([self.phase_num-t_value.matrix.shape[0], t_value.matrix.shape[1], ])
                t_value.matrix = np.concatenate([t_value.matrix, complete], axis=0)
            
            ## completion of the capacity
            if t_value.capacity.shape[0]<self.phase_num:
                complete = np.ones([self.phase_num-t_value.capacity.shape[0], t_value.capacity.shape[1], ])
                t_value.capacity = np.concatenate([t_value.capacity, complete], axis=0)

            
            ## completion of the signal
            if len(t_value.signals)<self.phase_num:
                t_value.signals.extend(['']*(self.phase_num-len(t_value.signals)))
            
            
            ## generate mask 
            mask = [0] * self.phase_num
            for idx, signal in enumerate(t_value.signals):
                if signal:
                    mask[idx] =1 
            t_value.mask = mask

## for action
    def implem_action_all(self):

        if self.lower_mode == 'FixTime':
            self.get_action()

        if self.lower_mode != 'FixTime':
            # ## get states
            # state = self.get_state()

            ## get actions
            actions, max_phase_idx = self._get_action_batch(self.old_state)

            ## update actions
            self._udpate_actions(actions, max_phase_idx)
        
        # return actions
      
    def _get_action_batch(self,state, training = False ):
        ''' get actions in batch with signal str
        '''

        ''' get phase value of all controlled junctions '''
        phase_value = self._get_phase_value(state)
        max_phase_idx = np.argmax(phase_value, axis=1)  # get optimal phase   
        max_phase_idx = dict(zip(self.controled_light, max_phase_idx))


        if training:
            return max_phase_idx
        else:
            ''' 
            -- OAM: explore actions
            -- MP: keep the actions: no frequent switches
            ''' 
            if self.lower_mode == 'MaxPressure':
                ## keep constraints
                max_phase_idx = self._keep_constraint(phase_value,max_phase_idx)

            if self.lower_mode == 'OAM':
                ## explore
                max_phase_idx = self._explore(max_phase_idx)

            ''' max green constraints '''
            max_phase_idx = self._max_green_constraints(max_phase_idx)
            self.action_index = max_phase_idx
            
            ''' get the signal str '''
            actions = {}
            for tl_id in self.controled_light:
                actions[tl_id] = self.tsc[tl_id].signals[max_phase_idx[tl_id]]

            
            return actions, max_phase_idx

    def _max_green_constraints(self, max_phase_idx):
        ''' check the maximum green time constraint for each junction
        '''

        for tl_id in self.controled_light:
            tsc = self.tsc[tl_id]
            if tsc.cur_phase_time >= self.green_max_duration and\
                tsc.cur_phase_idx == max_phase_idx[tl_id]:
                max_phase_idx[tl_id] = np.random.choice(np.nonzero(tsc.mask)[0])

        return max_phase_idx

    def _udpate_actions(self, actions, max_phase_idx):
        ''' rotate the prev phase and cur phase
        '''
        for tl_id, action in actions.items():
            tsc = self.tsc[tl_id]

            # phase
            tsc.prev_phase = tsc.cur_phase
            tsc.cur_phase = action
            tsc.cur_phase_idx = max_phase_idx[tl_id]

            # time
            if tsc.cur_phase == tsc.prev_phase: ## keep
                tsc.cur_phase_time += self.control_interval
            else: ## switch
                ## record
                tsc.phase_time_list.append(tsc.cur_phase_time) # phase time
                tsc.phase_list.append(tsc.prev_phase) # phase 
                
                ## update
                tsc.cur_phase_time = self.control_interval-self.yellow_duration

## metric
    def get_metric_each_epis(self, lane_data):
        ''' calculate metric of lower-level of each episode, the data is aggregated in 5 sec
            1. network level （all the nodes in the network）
               --1.1
            2. individual node level (each tsc)
            3. controller node level (controlled nodes in the network)
        '''
        step = 5
        
        lower_metric = {}

        ''' 1.1 network delay at each step (all the nodes), aggregated in 5 secs'''
        lower_metric['network_delay_step'] = lane_data['network_delay_step']

        ''' 1.2 network delay mean over the simulation horizon (network delay per second) '''
        lower_metric['network_delay_mean'] = np.mean(lower_metric['network_delay_step'])/step

        ''' 1.3 network perveh delay at each step, aggregated in 5 secs'''
        lower_metric['network_perveh_delay_step'] = lane_data['network_perveh_delay_step']

        ''' 1.4 network perveh delay mean over the simulation horizon (perveh delay per second) '''
        lower_metric['network_perveh_delay_mean'] = lane_data['network_perveh_delay_mean']


        ''' 2. individual controlled tsc level'''
        ## init dict for tsc metrics
        tsc_metrics = {key: [] for key in self.controled_light}

        tsc_through_step_node=[]
        tsc_delay_step_node = []
        tsc_sampleSeconds_step_node = []

        for tl_id in self.controled_light:
            
            tsc_metric = {}
            ''' 2.1 "step delay" at each step of each tsc (aggregated in 5 secs)'''
            tsc_metric['delay_step'] = lane_data['ToNode_delay_step'][self.tsc[tl_id].junc_id]
            tsc_delay_step_node.append(tsc_metric['delay_step'])

            ''' 2.2 "step throughput" at each step of each tsc (aggregated in 5 secs)'''
            tsc_metric['throughput_step'] = lane_data['ToNode_throughput_step'][self.tsc[tl_id].junc_id]
            tsc_through_step_node.append(tsc_metric['throughput_step'])

            ''' 2.3 "step sampleSeconds" at each step of each tsc (aggregated in 5 secs)'''
            tsc_metric['sampleSeconds_step'] = lane_data['ToNode_sampledSeconds_step'][self.tsc[tl_id].junc_id]
            tsc_sampleSeconds_step_node.append(tsc_metric['sampleSeconds_step'])

            ''' 2.4 " step perveh delay" at each step of each tsc (in each second)'''
            tsc_metric['sampleSeconds_step'][tsc_metric['sampleSeconds_step']==0]=1 # exclude 0 for divide
            tsc_metric['perveh_delay_step'] = np.nan_to_num(np.divide(tsc_metric['delay_step'], tsc_metric['sampleSeconds_step']))

            tsc_metrics[tl_id] = tsc_metric
        
        lower_metric['tsc_metrics'] = tsc_metrics


        ''' 3. controlled tsc level'''
        ''' 3.1 "step delay" at each step of all controlled tsc,(in each second) '''
        lower_metric['tsc_delay_step'] = np.sum(tsc_delay_step_node, 0)/step

        ''' 3.2 "mean delay" of all steps of all controlled tsc, (in each second) '''
        lower_metric['tsc_delay_mean'] = np.mean( lower_metric['tsc_delay_step'])

        ''' 3.3 "step throughput" of each step of all tsc, (aggregated in 5 secs)'''
        lower_metric['tsc_through_step'] = np.sum(tsc_through_step_node, 0)
        
        ''' 3.4 "mean throughput" of all steps of each tsc, veh/hour/tsc'''
        lower_metric['tsc_through_mean'] = np.mean(lower_metric['tsc_through_step'])/step*3600/len(self.controled_light)

        ''' 3.5 "step per_veh delay" at each step of all controlled tsc, (veh_delay per second) '''
        tsc_sampleSeconds_step = np.sum(tsc_sampleSeconds_step_node, 0)
        tsc_sampleSeconds_step[tsc_sampleSeconds_step==0] =1 # exclude 0 for divide
        lower_metric['tsc_perveh_delay_step'] = np.nan_to_num(np.divide(np.sum(tsc_delay_step_node, 0), tsc_sampleSeconds_step))
        
        ''' 3.6 "mean per_veh delay" of all steps of all controlled tsc (veh_delay per second)'''
        lower_metric['tsc_perveh_delay_mean'] = np.sum(tsc_delay_step_node)/np.sum(tsc_sampleSeconds_step_node)

        return lower_metric

    def record(self, lower_metric):
        ''' record performace of each epis
        '''
        for key in self.metric_list:
            self.record_epis[key].append(lower_metric[key])

class OAM(LowerAgents):
    ''' A class of Deep Q-network for descrete action
    '''


    def __init__(self, tsc, netdata, tau=config['tau']):
        super().__init__(tsc, netdata)
        ## feature


        """RL parameters"""
        self.lr_C = config['lr_C_lower']
        self.batch_size = config['batch_size']
        self.buffer_size = int(config['buffer_size_lower'] )
        self.reuse_time = config['reuse_time_lower']
        self.gamma = config['gamma_lower']
        self.epsilon = config['epsilon_lower']
        self.explore_decay = config['explore_decay_lower']

        """feature selections"""
        self.feature_list = ['inlane_density', 'inlane_speed', 'inlane_queue', 'inlane_veh_num', \
            'outlane_density', 'outlane_speed', \
                'cur_phase_idx', 'phase_inlane_mask', 'phase_mask']
        
        
        ''' network building'''
        self.critic = CriticLower(self.lr_C, tau)
        self.critic_udpate_num = 0
        self.max_phase = 10
        self.target_update_freq = config['target_update_freq']

        # ''' buffer'''
        # self.buffer.buffer_tls = dict.fromkeys(self.controled_light,[])
        # print(self.buffer.buffer_tls)
        



    def bellman(self, rewards,  phase_value, phase_value_target, max_q_index, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        # critic_target = np.asarray(q_values)
        critic_target = phase_value.copy()
        for reward, action_idx, target, target_network_value, done in \
            zip(rewards, max_q_index, critic_target, phase_value_target, dones):
            if done:
                target[action_idx] = reward
            else:
                target[action_idx] = reward + self.gamma * target_network_value[action_idx]

        return critic_target

    def update_models(self, states, phase_matrix_batch, critic_target):
        """ Update critic networks from sampled experience
            The target network updates are delayed
        """

        ## Train critic
        loss = self.critic.train_on_batch(states, phase_matrix_batch, critic_target)
        self.critic_udpate_num += 1

        ### Transfer weights to target networks at rate Tau
        soft_update_idx = False  # indicate the soft_update completion
        if self.critic_udpate_num % self.target_update_freq == 0 :
            # print('########## soft update')
            self.critic.transfer_weights()
            soft_update_idx = True

        return soft_update_idx, loss

    def _memorize(self, last_states, actions, states):
        self.memory['state'].extend([last_states[tl_id] for tl_id in self.controled_light])
        self.memory['action'].extend([actions[tl_id] for tl_id in self.controled_light])
        self.memory['next_state'].extend([states[tl_id] for tl_id in self.controled_light])
        # self.memory['reward'].extend([rewards[tl_id] for tl_id in self.controled_light])

    def get_state(self):
        ''' get state of each tls for OAM
        '''
        state = {}

        # for tl_idx, tl_id in enumerate(self.controled_light):
        for tsc in self.tsc.values(): 
            lane_state = tsc.get_state()

            state[tsc.id] = {
                'inlane_density': lane_state[:,0],
                'inlane_speed': lane_state[:,1] ,
                'inlane_queue':  lane_state[:,2],
                'inlane_veh_num':  lane_state[:,3],
                'outlane_density': lane_state[:,4],
                'outlane_speed': lane_state[:,5],
                'phase_inlane_mask': tsc.matrix,
                'cur_phase_idx': tsc.cur_phase_idx,
                'phase_mask': tsc.mask
                }
        return state

    def get_reward(self, tsc_metrics):
        ''' obtain reward after each epis for OAM
            The original data resolution is 5 secs
            Need to be grouped by the lower-level resolution    
        '''
        k = int(self.control_interval / 5)

        reward = []

        for tsc_metric in tsc_metrics.values():
            delay = tsc_metric['delay_step']
            delay_agg = [sum(delay[i:i + k]) for i in range(0, len(delay), k)]
            reward.extend(delay_agg)

        # memorize 
        self.memory['reward'].extend(reward)



        return reward



## funcs for generating actions within OAM class
    def _explore(self, max_phase_idx):
        ''' epsilon greedy
        '''
        # for tl_idx, _ in enumerate(self.controled_light):
        #     if np.random.random() < self.epsilon:
        #         max_q_index[tl_idx] = np.random.choice(np.nonzero(self.batch_phase_masks[tl_idx,:])[0])

        for tl_id in self.controled_light:
            if np.random.random() < self.epsilon:
                max_phase_idx[tl_id] = np.random.choice(np.nonzero(self.tsc[tl_id].mask)[0])

        return max_phase_idx

    def _get_phase_value(self, state):
        ''' get phase value through OAM network 
        '''
        phase_value = self.critic.eval_predict(self._create_action_inputs(state))

        return phase_value  

    def _create_action_inputs(self, state):
        ''' get batch state inputs for the OAM model
        '''
        batch_state = {f: [] for f in self.feature_list}

        for a in self.controled_light:
            for f in self.feature_list:
                batch_state[f].append(state[a][f])
        for f in self.feature_list:
            batch_state[f] = np.array(batch_state[f], dtype=float)
            if f in ['inlane_density', 'inlane_speed', 'inlane_queue', 'inlane_veh_num']:
                batch_state[f] = batch_state[f][:,:, np.newaxis]
            if f in ['cur_phase_idx']:
                batch_state[f] = batch_state[f][:,np.newaxis]

        return batch_state


## replay
    def _create_buffer(self):
        batch_action = self.memory['action']
        batch_reward = self.memory['reward']
        batch_state = {f: [s[f] for s in self.memory['state']] for f in self.feature_list}
        batch_next_state = {f: [s[f] for s in self.memory['next_state']] for f in self.feature_list}

        for f in self.feature_list:
            batch_state[f] = np.array(batch_state[f], dtype=float)
            batch_next_state[f] = np.array(batch_next_state[f], dtype=float)

            # add axis
            if f in ['inlane_density', 'inlane_speed', 'inlane_queue', 'inlane_veh_num']:
                batch_state[f] = batch_state[f][:,:, np.newaxis]
                batch_next_state[f] = batch_next_state[f][:,:, np.newaxis]
            if f in ['cur_phase_idx']:
                batch_state[f] = batch_state[f][:,np.newaxis]
                batch_next_state[f] = batch_next_state[f][:,np.newaxis]

        batch_action = np.array(batch_action, dtype=int).reshape(-1, 1)
        batch_reward = np.array(batch_reward, dtype=float).reshape(-1, 1)

        print('Buffer size: ', len(batch_reward))

        batch_reward = self._reward_normalize(batch_reward)

        return batch_state, batch_action, batch_reward, batch_next_state

    def _buffer_sampler(self, state, action, reward, next_state, target):
        num_exp = len(reward)
        num_sample = min(self.batch_size, num_exp)

        batch_state = {}
        batch_next_state = {}

        index = np.random.choice(list(range(num_exp)), num_sample)

        for f in state:
            batch_state[f] = state[f][index]
            batch_next_state[f] = next_state[f][index]

        batch_action = action[index]
        batch_reward = reward[index]
        batch_target = target[index]

        return batch_state, batch_action, batch_reward, batch_next_state, batch_target

    def _reward_normalize(self, batch_reward):
        return np.divide(batch_reward, 50*self.control_interval)

    def TD_learning(self):
        state, action, reward, next_state = self._create_buffer()
        num_exp = len(reward)
        num_sample = min(self.batch_size, num_exp)

        one_hot_action = np.eye(self.max_phase)[action].reshape(-1, self.max_phase)
        state['phase_mask'] *= one_hot_action   # make training loss 0 for not-actions' q value

        value_loss_list = []
        for n in range(num_exp * self.reuse_time // num_sample):
            ############################## Update 1-step value function ##################################
            if n % self.target_update_freq == 0:
                # print('########## soft update')
                self.critic.transfer_weights()
                target = self.critic.target_predict(next_state)
                target = reward + self.gamma * target.max(axis=1).reshape(-1, 1)
                target = np.tile(target, (1, self.max_phase))
                # make training loss 0 for not-actions' q value
                target *= one_hot_action
                target += (one_hot_action - 1) * 1e5

            batch_state, batch_action, batch_reward, batch_next_state, batch_target = self._buffer_sampler(state, one_hot_action, reward, next_state, target)

            loss = self.critic.train_on_batch(batch_state, batch_target, n)

            # loss = np.sqrt(loss)
            value_loss_list.append(loss)
            # if n % 10 == 0:
            print(f'Epoch: {n} Sample: {len(batch_reward)} Loss: {format(loss, ".3f")}' , end='\r')

        print(' ' * 24 + f'\rAvg loss: {np.mean(value_loss_list)}')

        return value_loss_list





class MaxPressure(LowerAgents):
    def __init__(self, tsc, netdata):
        super().__init__(tsc, netdata)

    def get_state(self):
        batch_state = []
        for tl_id in self.controled_light:
            state = self.tsc[tl_id].get_state()
            batch_state.append(state)
             
        return np.array(batch_state)

    def _get_phase_value(self, state):
        ''' get phase value through OAM network 
        '''
        ## phase value
        phase_value = np.matmul(self.batch_phase_matrix, state).squeeze()

        ## mask the unavailable phases
        phase_value [self.batch_phase_masks==0] =-np.inf

        return phase_value

    def _keep_constraint(self, phase_value, max_q_index ):
        ''' keep the last phase if there is long queue
        ''' 

        for i, tl_id in enumerate(self.controled_light):
            old_action = self.tsc[tl_id].cur_phase_idx
            if phase_value[i, old_action] >= 3: # still long queue
                max_q_index[tl_id] = old_action
        
        return max_q_index
                

class FixTime(LowerAgents):

    def __init__(self, tsc, netdata):
        super().__init__(tsc, netdata)
        self.program_id = '0'  if config['network'] == 'Grid' else 2
        # '2' -- acturated, max_dur = 90 sec
        # '3' -- acturated, max_dur = 45 sec


    def set_program(self):
        for tl_id in self.controled_light:
            # phase
            # print(traci.trafficlight.getProgram(tl_id))
            traci.trafficlight.setProgram(tl_id, self.program_id)

    def get_state(self):             
        return 
    
    def get_action(self):
        pass
        # for tl_id in self.controled_light:
        #     # phase
        #     print(traci.trafficlight.getProgram(tl_id))
        

    def save_weights(self, path):
        pass



if False:
    def set_subscribe(self):
        ''' subscribe to get data 
        '''
        # for lane_id in self.lanes_id_list:
        #     lane_length = self.netdata['lane'][lane_id]['length']
        #     traci.lane.subscribeContext(lane_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, lane_length,
        #                                 [traci.constants.VAR_LANEPOSITION, 
        #                                     traci.constants.VAR_SPEED,
        #                                    traci.constants.VAR_LANE_ID]) 
        # for edge_id in self.edges_id_list:
        #     edge_length = self.netdata['edge'][edge_id]['length']
        #     traci.edge.subscribeContext(edge_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, edge_length,
        #                                 [traci.constants.VAR_LANEPOSITION, 
        #                                     traci.constants.VAR_SPEED,
        #                                    traci.constants.VAR_LANE_ID]) 

        # create subscription 
        # MUST IN THE SAME PORT
        # for tsc in self.tsc.values():
        #     #create subscription for this traffic signal junction to gather
        #     #vehicle information efficiently
        #     traci.junction.subscribeContext(tsc.junc_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, tsc.max_length,
        #                                     [traci.constants.VAR_LANEPOSITION, 
        #                                     traci.constants.VAR_SPEED, 
        #                                     traci.constants.VAR_LANE_ID])
        
        for lane in (self.netdata['lane']).keys():
            traci.lane.subscribe(lane, [traci.constants.LAST_STEP_MEAN_SPEED, 
                                        traci.constants.LAST_STEP_VEHICLE_NUMBER])

    def get_reward(self, lane_delay):
        # batch_reward = []
        # for tl_id in self.controled_light:
            
        #     batch_reward.append(self.tsc[tl_id].get_reward()/self.reward_norm)

        # return sum(batch_reward), batch_reward


        for t_id, t_value in self.tsc.items():
            incoming_dalay = {}

            ## get lane delays
            for lane_id, delay in lane_delay.items():
                incoming_lane = lane_id.split('_')[0]
                if incoming_lane in t_value.incoming_edges:
                    incoming_dalay[lane_id] = delay
            
            self.tsc[t_id].cal_delay(incoming_dalay)

    def process_data(self):
        ''' process data for each tsc
        '''

        ## junction value retrive
        for tsc in self.tsc.values():
            # if tsc.id not in self.perimeter_light:
            tsc.run()

        ## edge value retrive
        # data = traci.lane.getAllSubscriptionResults()

        # return data

    def _get_action(self, state, signal_matrix):
        ''' get action with evaluate network in batch
        '''
        phase_value_batch = self.critic.eval_predict([state, signal_matrix])
        return phase_value_batch

    def get_action_batch(self,state, phase_matrix, phase_mask, training = False ):

        ## phase value
        phase_value_batch = np.matmul(phase_matrix, state).squeeze()

        ## mask the unavailable phases
        phase_value_batch[phase_mask==0] =-np.inf

            
        ## argmax
        max_q_index = list(np.argmax(phase_value_batch, axis=1))

        ## keep constraints
        if self.action_index:
            for i, (phase_value, old_action) in \
                enumerate(zip(phase_value_batch, self.action_index)):

                if phase_value[old_action] >= 3:
                    max_q_index[i]= old_action

        ## max green constraints
        max_q_index = self.max_green_constraints(max_q_index)

        ## record
        self.action_index = max_q_index
        
        ## get the signal
        actions = self.batch_signals[ np.arange(len(self.batch_signals)), max_q_index]
        # print(actions)
        return actions

    def memorize(self, done):
        ''' save memory for each tls at each timestep
        '''
        for tl_id, state_new, state_old, action_idx in \
            zip(self.controled_light, self.new_state, self.old_state, self.action_index):
        
            # self.buffer.memorize(state_old, action_idx, reward, 0, done, state_new, self.tsc[tl_id].matrix, self.tsc[tl_id].mask)
            reward = 0
            penalty = 0

            experience = [np.array(state_old), action_idx, reward, penalty, done, np.array(state_new)]
            self.buffer.buffer_tls[tl_id].append(experience)

    def get_metrics(self, lane_data, edge_data):
        ''' get metric for each tsc
        '''
        
        for tsc in self.tsc.values():
            # if tsc.id not in self.perimeter_light:
            # tsc.cur_metric = tsc.get_metrics()
            tsc.cal_delay(lane_data, edge_data)
            # print(tsc.cur_metric['delay']/(tsc.ep_delay_edge[-1]+1e-4), tsc.cur_metric['delay']/(tsc.ep_delay_edge[-1]+1e-4))
    
    def calculate_network_mean_delay(self):
        ''' PN+Buffer mean_delay
        '''

        tot_veh_num = 0
        tot_delay = 0
        for tl_id in self.controled_light:
            if tl_id not in self.perimeter_light:

                tot_veh_num += sum(self.tsc[tl_id].veh_num)
                tot_delay += sum(self.tsc[tl_id].ep_delay_edge)
        
        network_mean_delay = -(tot_delay/tot_veh_num) 

        return network_mean_delay

    def replay(self):
        ''' OAM replay
        '''
        # print('############ before train ##############')
        # W, target_W = self.critic.model.get_weights(), self.critic.target_model.get_weights()
        # print('##Evaluate network')
        # print(W)
        # print('##Target network')
        # print(target_W)       
        
        ## Sample experience from buffer
        states, actions, rewards, dones, new_states, phase_matrix_batch, phase_mask_batch = self.buffer.sample_batch(self.batch_size)

        ## obtain new action
        max_q_index = self.get_action_batch(new_states, phase_matrix_batch, phase_mask_batch, True )

        ## Predict target q-values using target networks
        phase_value_target = self.critic.target_predict([new_states, phase_matrix_batch])

        ## Predict original q_values
        phase_value_old = self.critic.target_predict([new_states, phase_matrix_batch])

        #print(self.actor.target_predict(new_states))
        
        ## check nan of Q values
        if np.isnan(phase_value_target).any():
            raise Exception('There is NaN in the q-value')

        ## Compute critic target
        critic_target = self.bellman(rewards, phase_value_old, phase_value_target, max_q_index, dones)
        
        ## Train both networks on sampled batch, update target networks
        _, loss = self.update_models(states, phase_matrix_batch, critic_target)
        
        return loss
