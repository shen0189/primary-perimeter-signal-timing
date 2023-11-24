import random
import numpy as np

from collections import deque, Counter
# from .sumtree import SumTree
from utils.utilize import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import math


class MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size, reward_delay_steps, penalty_delay_steps,\
         reward_norm_bool, multi_step, gamma, sample_mode, with_per=False):
        """ Initialization
        """
        self.with_per = with_per
        self.buffer_size = buffer_size
        self.reward_delay_steps = reward_delay_steps
        self.penalty_delay_steps = penalty_delay_steps
        self.reward_norm_bool = reward_norm_bool
        self.multi_step = multi_step
        self.gamma = gamma
        self.sample_mode = sample_mode

        # self.cur_epis_num = 0 # number of memory in the current episode
        self.count_old = 0  # record total memory number at the end of the last episode
        self.count_CA = 0  # memory in buffer_CA which has conducted CA
        # CA buffer
        self.buffer_CA = []

    def credit_assignment(self, delay_steps, idx, reward, buffer_idx, distri, idx_in_episode, new_memory):
        ''' Credit assignment for each reward and penalty
        '''
        ## 1. find available steps for assignment
        avai_assign_step = min(delay_steps, idx_in_episode)  # steps available for assignment

        ## 2. generate distribution
        skew_numbers = np.random.gamma(distri, 1, 1000)  # generate 1000 skewed random numbers
        hist, bin_edges = np.histogram(max(skew_numbers) - skew_numbers, avai_assign_step, density=True)
        ratio = hist * np.diff(bin_edges)

        ## 3. split
        split = reward * ratio

        ## 4. assignment
        index = 0
        for j in range(idx - avai_assign_step + 1, idx + 1):  # j -- index of memory
            # self.buffer_CA[j][buffer_idx] += split[index] # penalty
            new_memory[j][buffer_idx] += split[index]  # penalty
            index += 1

    def credit_assignment_batch(self, new_memory):
        ''' assign reward to the previous steps for each batch
        '''

        if self.reward_delay_steps != 0 and self.penalty_delay_steps != 0:
            ''' batch credit assignment of the reward and penalty
            '''

            d_batch = [memory[3] for memory in new_memory]  # list of Dones

            # for i in range(self.count_CA, len(self.buffer_CA)):
            for i, memory in enumerate(new_memory):
                ''' conduct CA for each memory in the new memory
                '''

                ## get the last terminate state before each piece of new memory
                try:
                    d_slice = d_batch[:i]
                    d_slice.reverse()
                    true_index = i - d_slice.index(True)
                except:
                    true_index = 0

                ## get the index of the current memory within this episode
                idx_in_episode = i - true_index + 1

                ### 1. reward credit assignment
                reward = memory[2]
                memory[2] = 0

                self.credit_assignment(self.reward_delay_steps, i, reward, 2, 15, idx_in_episode, new_memory)

                ### 2. penalty credit assignment
                penalty = memory[5]
                memory[5] = 0

                self.credit_assignment(self.penalty_delay_steps, i, penalty, 5, 1.8, idx_in_episode, new_memory)

            ## 3. update count_CA
            # self.count_CA = len(self.buffer_CA)
        return new_memory

    def priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        return (error + self.epsilon)**self.alpha

    def size(self):
        """ Current Buffer Occupation
        """
        return self.count

    def process_new_memory(self, e):
        """ Process the new memory
            1. copy the new memory
            2. calculate penalty
            3. credit assignment
            4. multi-step rewawrd
            5. reward property
        """

        ## 1. process the new memory
        new_memory_number, new_memory = self.get_new_memory()  # deepcopy the new memory
        # penalty_list, new_memory = self.calculate_penalty_batch(new_memory)  # calculate penlaty without credit assignment
        new_memory = self.credit_assignment_batch(new_memory)  # conduct CA of the new memory
        new_memory = self.multistep_reward(new_memory)

        ## 2. copy the new memory to buffer_CA
        self.buffer_CA.extend(deepcopy(new_memory))
        self.buffer_CA = self.buffer_CA[-self.buffer_size:]
        self.count_CA = len(self.buffer_CA)

        ## 3. update property
        self.update_reward_property()

        ## plot
        if self.mode == 'Upper':
            self.upper_buffer_plot(e)

            # self.CA_plot(new_memory_number, penalty_list, e)

    def multistep_reward(self, new_memory):
        ''' calculate multi-step reward for TD target   
        '''
        ## 1. calculate one-step objective for all memory
        # self.buffer_CA[:,6] = self.buffer_CA[:,2] + self.buffer_CA[:,5]
        new_memory = np.array(new_memory)
        objective_batch = (new_memory[:, 2] + new_memory[:, 5]).reshape(-1, 1)
        new_memory = np.concatenate([new_memory, objective_batch], axis=1)

        ## 2. multi_step objective calculation
        if self.multi_step > 1:

            for i in range(len(new_memory)):

                ## 2.1 calculate multi-step objective
                new_memory_reward = 0
                for k in range(self.multi_step):
                    # exclude the current step, if multi-step ==3, then look ahead 2 steps exclude itself
                    # k -- the next step memory for adding
                    new_memory_reward += self.gamma**k * new_memory[i + k][-1]
                    if new_memory[i + k][3]:
                        break

                new_memory[i][-1] = new_memory_reward
                ## 2.2 redefine the next-state: state after multi-steps
                new_memory[i][4] = new_memory[i + k][4]

        return new_memory

    def update_reward_property(self):
        ''' Calculate reward property: min, max, mean, std
        '''
        object_one_step = np.array([i[2] + i[5] for i in self.buffer_CA])
        object_multi_step = np.array([i[-1] for i in self.buffer_CA])

        penalty_all = np.array([i[5] for i in self.buffer_CA])
        reward_all = np.array([i[2] for i in self.buffer_CA])

        # reward 
        self.reward_max = max(reward_all)
        self.reward_min = min(reward_all)
        self.reward_mean = np.mean(reward_all)
        self.reward_std = np.std(reward_all)

        print(f'\n##### {self.mode} REWARD #####')
        print(f'max reward:{self.reward_max}')
        print(f'min reward:{self.reward_min}')
        print(f'average reward:{self.reward_mean}')
        print(f'std reward:{self.reward_std}')

        # penalty 
        self.penalty_max = max(penalty_all)
        self.penalty_min = min(penalty_all)
        self.penalty_mean = np.mean(penalty_all)
        self.penalty_std = np.std(penalty_all)

        print(f'\n##### {self.mode} PENALTY #####')
        print(f'max penalty:{self.penalty_max}')
        print(f'min penalty:{self.penalty_min}')
        print(f'average penalty:{self.penalty_mean}')
        print(f'std penalty:{self.penalty_std}')

        # object one step
        self.object_one_step_max = max(object_one_step)
        self.object_one_step_min = min(object_one_step)
        self.object_one_step_mean = np.mean(object_one_step)
        self.object_one_step_std = np.std(object_one_step)
        
        print(f'\n##### {self.mode} ONE-STEP-OBJECTIVE #####')
        print(f'max object_one_step:{self.object_one_step_max}')
        print(f'min object_one_step:{self.object_one_step_min}')
        print(f'average object_one_step:{self.object_one_step_mean}')
        print(f'std object_one_step:{self.object_one_step_std}')

        # object multi-step
        self.object_multi_step_max = max(object_multi_step)
        self.object_multi_step_min = min(object_multi_step)
        self.object_multi_step_mean = np.mean(object_multi_step)
        self.object_multi_step_std = np.std(object_multi_step)

        if self.multi_step >1:
            print(f'\n##### {self.mode} MULTI-STEP-OBJECTIVE #####')
            print(f'max object_multi_step:{self.object_multi_step_max}')
            print(f'min object_multi_step:{self.object_multi_step_min}')
            print(f'average object_multi_step:{self.object_multi_step_mean}')
            print(f'std object_multi_step:{self.object_multi_step_std}')

    def reward_normalization(self, obj_batch):
        ''' Conduct normalization for one-step objective value
        '''
        obj_batch = (obj_batch - self.object_one_step_min) / (self.object_one_step_max - self.object_one_step_min)
        # plt.hist(r_batch)
        # plt.show()
        return obj_batch

    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER)
        """

        # random.seed(1)
        # np.random.seed(1)

        batch = []

        # Sample using prorities
        if (self.with_per):
            T = self.buffer.total() // batch_size
            for i in range(batch_size):
                a, b = T * i, T * (i + 1)
                skew_numbers = random.uniform(a, b)
                idx, error, data = self.buffer.get(skew_numbers)
                batch.append((*data, idx))
            idx = np.array([i[5] for i in batch])

        # Sample randomly from Buffer
        elif self.count_CA < batch_size:
            idx = None
            batch = random.sample(self.buffer_CA, self.count_CA)
        else:
            idx = None
            batch = self.sample_batch_by_action(batch_size)
            # batch = random.sample(self.buffer_CA, batch_size)

        # Return a batch of experience
        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        # r_batch = np.array([i[2] + i[5] for i in batch])


        r_batch = np.array([i[-1] for i in batch])

        ## normalization
        if self.reward_norm_bool == True:
            r_batch = self.reward_normalization(r_batch)

        d_batch = np.array([i[3] for i in batch])
        new_s_batch = np.array([i[4] for i in batch])

        if self.mode == 'Upper':
            return s_batch, a_batch, r_batch, d_batch, new_s_batch
        
        else:
            phase_matrix_batch = np.array([i[6] for i in batch])
            phase_mask_batch = np.array([i[7] for i in batch])
            return s_batch, a_batch, r_batch, d_batch, new_s_batch, phase_matrix_batch, phase_mask_batch

    def sample_batch_by_action(self, batch_size):
        ''' Sample the batch based on the distribution of actions
        '''
        if self.sample_mode == 'balance':
            actions = np.around(np.linspace(0, 1, 11), 1) # actions to classify
            a_all = [ int(i[1]*10)/10 for i in self.buffer_CA] # get all action in the buffer

            batch = []
            for action in actions:
                idx = np.where(np.array(a_all) == action)[0] # all index of one acton
                sample_index = random.sample(list(idx), min(len(idx), batch_size // len(actions)))
                batch.extend (np.array(self.buffer_CA)[sample_index].tolist())


        elif self.sample_mode == 'random':
            batch = random.sample(self.buffer_CA, batch_size)

        return batch

    def get_new_memory(self):
        ''' get the new raw experience in the buffer
        '''
        new_memory_number = self.count - self.count_old  # number of new memory
        new_memory = deepcopy(self.buffer[-new_memory_number:])  # deepcopy the new memory

        # new_memory_number = len(self.buffer) - len(self.buffer_CA)
        # self.buffer_CA.extend(deepcopy(self.buffer[-new_memory_number:]))

        return new_memory_number, new_memory

    def upper_buffer_plot(self, e):
        ''' plot of upper buffer memory 
        '''
        self.action_reward_distri_plot()
        self.plot_accu_reward()
        if e >= 0:
            self.reward_distri_plot()
            self.action_distri_plot(e)


########### Plot ################

    def update(self, idx, new_error):
        """ Update priority for idx (PER)
        """
        self.buffer.update(idx, self.priority(new_error))

    def clear(self):
        """ Clear buffer / Sum Tree
        """
        if (self.with_per): self.buffer = SumTree(buffer_size)
        else: self.buffer.clear()
        self.count = 0


    def reward_distri_plot(self):
        ''' plot the reward distribution in the memory
        '''
        reward_all = [b[2] for b in self.buffer_CA]
        penalty_all = [b[5] for b in self.buffer_CA]
        obj_all_origin = [b[2] + b[5] for b in self.buffer_CA]
        obj_all_normal = self.reward_normalization(obj_all_origin)

        ## plot histgram of original objective
        plt.hist(obj_all_origin)
        plt.xlabel('objective value')
        plt.ylabel('number')
        plt.title('objective_one_step distribution')
        plt.savefig(f"{config['plots_path_name']}metric/object_onestep_dist.png")
        plt.close()

        ## plot histgram of objective after normalization
        plt.hist(obj_all_normal)
        plt.xlabel('objective normalized')
        plt.ylabel('number')
        plt.xlim((int(self.object_one_step_min), math.ceil(self.object_one_step_max))) 
        plt.title('objective distribution after normalization')
        plt.savefig(f"{config['plots_path_name']}metric/object_onestep_dist_norm.png")
        plt.close()

        ## plot histgram of reward
        plt.hist(reward_all)
        plt.xlabel('reward')
        plt.ylabel('number')
        plt.title('reward distribution')
        plt.savefig(f"{config['plots_path_name']}metric/reward_onestep_dist.png")
        plt.close()

        ## plot histgram of penalty
        plt.hist(penalty_all)
        plt.xlabel('penalty')
        plt.ylabel('number')
        plt.title('penalty distribution')
        plt.savefig(f"{config['plots_path_name']}metric/penalty_onestep_dist.png")
        plt.close()

        ''' multi-step objective distribution
        '''
        if self.multi_step >1:
            obj_all_multistep = [b[-1] for b in self.buffer_CA]
            plt.hist(obj_all_multistep)
            plt.xlabel('multi-step obj')
            plt.ylabel('number')
            plt.title('multi-step obj dist')
            plt.xlim((int(self.object_multi_step_min), math.ceil(self.object_multi_step_max))) 
            plt.savefig(f"{config['plots_path_name']}metric/obj_multistep_dist.png")
            plt.close()

    def action_distri_plot(self, e):

        # a_all = np.around([i[1] for i in self.buffer_CA], 1)
        a_all = [i[1] for i in self.buffer_CA]

        if config['peri_action_mode'] == 'decentralize':
            # DDPG controller
            pass

        elif config['peri_action_mode'] == 'centralize':
            counter = Counter(a_all)
            sort_counter_key = sorted(list(counter.keys()))
            sort_counter_value = [counter[k] for k in sort_counter_key]
            plt.clf()
            plt.bar(x=list(map(str, sort_counter_key)), height=sort_counter_value)
            plt.xlabel('action')
            plt.ylabel('number')
            plt.title('actions distribution')
            plt.savefig(f"{config['plots_path_name']}metric/action_distri.png")
            plt.close()

    def action_reward_distri_plot(self):
        ''' plot histgram of the reward of each action
        '''
        if config['peri_action_mode'] == 'centralize':
            ''' single-step reward dist plot
            '''
            r_all_origin = [b[2] + b[5] for b in self.buffer_CA]  # raw reward
            a_all = [i[1] for i in self.buffer_CA]  # all actions in the memory
            actions = set(a_all)

            if self.reward_norm_bool == True:
                # plot the object distri after normalized of each action
                r_all_normal = list(self.reward_normalization(r_all_origin))  # reward after normalization

                # [action, reward] pairs
                a_r_normal = np.vstack([a_all, r_all_normal])

                for a in actions:
                    # res = (a_r_normal[:, a_r_normal[0] == a])
                    plt.hist(a_r_normal[1, a_r_normal[0] == a])  
                    plt.xlabel(f'reward normalized')
                    plt.ylabel('number')
                    plt.ylim((0, self.count_CA/8)) 
                    plt.title(f'reward distribution after normalization of action = {a}')
                    plt.savefig(f"{config['plots_path_name']}metric/obj_dist_norm_onestep_{a}.png")
                    plt.close()

            else:
                # plot the original object distri of each action
                a_r_origin = np.vstack([a_all, r_all_origin])

                for a in actions:
                    plt.hist(a_r_origin[1, a_r_origin[0] == a])  
                    plt.xlabel(f'raw objective')
                    plt.ylabel('number')
                    plt.xlim((int(self.object_one_step_min)-1, math.ceil(self.object_one_step_max))) 
                    plt.ylim((0,self.count_CA/8)) 
                    plt.title(f'original objective distribution of action = {a}')
                    plt.savefig(f"{config['plots_path_name']}metric/obj_dist_raw_onestep_{a}.png")
                    plt.close()

            ''' multi-step reward dist plot
            '''
            if self.multi_step>1:
                obj_all_multi_step = [b[-1] for b in self.buffer_CA]
                a_obj_multi_step = np.vstack([a_all, obj_all_multi_step])

                for a in actions:
                    plt.hist(a_obj_multi_step[1, a_obj_multi_step[0] == a])  
                    plt.xlabel(f'multi-step raw objective')
                    plt.ylabel('number')
                    plt.xlim((int(self.object_multi_step_min), math.ceil(self.object_multi_step_max))) 
                    plt.ylim((0,self.count_CA/8)) 
                    plt.title(f'raw multi-step obj dist of action = {a}')
                    plt.savefig(f"{config['plots_path_name']}metric/obj_dist_raw_multistep_{a}.png")
                    plt.close()

    def reward_plot(self, config, num, e):
        '''  plot the reward/ penalty of each episode 
        '''
        r_batch, p_batch, r_all_batch = [], [], []
        for memory in list(self.buffer_CA)[-num:]:
            r_batch.append(memory[2])
            p_batch.append(memory[5])
            r_all_batch.append(memory[2] + memory[5])

            # r_batch = np.array([i[2] for i in batch])
            # p_batch = np.array([i[5] for i in batch])
            # r_all_batch = np.array([i[2]+i[5] for i in batch])

        if not p_batch:
            return

        plt.xlabel('memory')
        plt.ylabel('reward')
        plt.title('reward/penalty')
        plt.plot(range(num), r_batch, 'bo-', label=f"reward+")
        plt.plot(range(num), p_batch, 'g>-', label=f"penalty-")
        plt.plot(range(num), r_all_batch, 'r+-', label=f"reward_all")
        plt.legend()
        # plt.ylim((-1,1))
        plt.savefig(f"{config['plots_path_name']}e{e+1}_reward_details.png")
        plt.close()

    def CA_plot(self, new_memory_number, penalty_list, e):
        ''' plot the reward/penalty comparison in terms of credit assignment
        '''
        reward_original, reward_CA, penalty_original, penalty_CA, actions = [], [], [], [], []
        # get the reward/penalty
        for i in range(-new_memory_number, 0, 1):
            reward_original.append(self.buffer[i][2])
            reward_CA.append(self.buffer_CA[i][2])

            penalty_original.append(penalty_list[i])
            penalty_CA.append(self.buffer_CA[i][5])

            actions.append(self.buffer_CA[i][1])

        ## plot subfigures
        plt.subplot(2, 1, 1)
        plt.ylabel('reward')
        plt.title(f'reward/penalty credit assignment of (episode{e+1})')
        plt.plot(range(new_memory_number), reward_original, 'o-', label=f"original reward")
        plt.plot(range(new_memory_number), reward_CA, 'g>-', label=f"CA reward")
        plt.plot(range(new_memory_number), actions, 'k>-', label=f"actions")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.ylabel('penalty')
        # plt.title(f'penalty credit assignment of (episode{e+1})')
        plt.plot(range(new_memory_number), penalty_original, 'o-', label=f"original penalty")
        plt.plot(range(new_memory_number), penalty_CA, 'g>-', label=f"CA penalty")
        plt.plot(range(new_memory_number), actions, 'k>-', label=f"actions")
        plt.legend()

        plt.savefig(f"{config['plots_path_name']}credit_assignment.png")
        plt.close()

    def plot_accu_reward(self):
        ''' Plot reward v.s accu of each simulation
        '''
        reward = [b[2]+b[5] for b in self.buffer_CA[-self.buffer_size:]]  # raw reward
        accu = [b[0][0] * config['accu_max']  for b in self.buffer_CA[-self.buffer_size:]] 
        # unit reform
        plt.xlabel('accu(veh)')
        plt.ylabel('one-step obj')
        plt.title(f'one-step obj v.s accu')
        plt.scatter(accu, reward)
        plt.xlim((0., config['accu_max']))
        plt.ylim((-3., 3))
        # plt.plot(x1, y1, label='整体路网基本图')
        # plt.plot(x2, y2, label='子路网基本图')
        # plt.legend()
        # plt.show()
        plt.savefig(
            f"{config['plots_path_name']}metric/accu_onestep-obj.png")
        plt.close()

class MemoryBuffer_Upper(MemoryBuffer):
    def __init__(self, buffer_size, reward_delay_steps, penalty_delay_steps,\
         reward_norm_bool, multi_step, gamma, sample_mode, with_per=False):
        super().__init__(buffer_size, reward_delay_steps, penalty_delay_steps,\
         reward_norm_bool, multi_step, gamma, sample_mode, with_per=False)

        self.mode = 'Upper'
        try:
            ## load existing memory
            self.buffer = np.load(f"{config['savefile_dir']}cache/memory_{config['Peri_mode']}.npy", allow_pickle=True)
            # self.buffer = self.buffer[-self.buffer_size:]
            self.count = np.shape(self.buffer)[0]
            self.buffer = list(self.buffer)
            # self.buffer_CA = deepcopy(self.buffer)
            self.process_new_memory(-1)
            print(f'####### Upper Agent Load memory: Success {self.count} exist memory ######')

        except:
            print('####### No existed memory of Upper Agent ######')
            # Standard Buffer
            self.buffer = []
            self.count = 0

    def save(self):
        np.save(f"{config['models_path_name']}memory_upper_{config['upper_mode']}.npy", self.buffer)
        print(f'###### Upper Memory save: Success {self.count} ######')

    def memorize(self, state, action, reward,  done, new_state, penalty):
        """ Save an experience to memory, optionally with its TD-Error
        """

        ## initialize default experience 
        # [state, action, reward, done, new_state, penalty, matrix]
        experience = [state, action, reward, done, new_state, penalty]  
        # self.cur_epis_num += 1

        self.buffer.append(experience)
        self.count += 1

        
class MemoryBuffer_Lower(MemoryBuffer):
    def __init__(self, buffer_size, reward_delay_steps, penalty_delay_steps,\
         reward_norm_bool, multi_step, gamma, sample_mode, with_per=False):
        super().__init__(buffer_size, reward_delay_steps, penalty_delay_steps,\
         reward_norm_bool, multi_step, gamma, sample_mode, with_per=False)
        
        self.mode = 'Lower'

        print('####### No existed memory of Lower Agent ######')
        self.buffer = []
        self.count = 0

    def save(self):
        np.save(f"{config['models_path_name']}memory_lower_{config['lower_mode']}.npy", self.buffer[-self.buffer_size:])
        print(f'\n########### Lower Memory save: Success {min(self.count, self.buffer_size) } ###########')

    def memorize(self, state, action, reward, penalty, done, new_state, phase_matrix, phase_mask):
        """ Save an experience to memory, optionally with its TD-Error
        """

        ## initialize default experience 
        # [state, action, reward, done, new_state, penalty, matrix]
        experience = [state, action, reward, done, new_state, penalty, phase_matrix, phase_mask]  
        # self.cur_epis_num += 1

        self.buffer.append(experience)
        self.count += 1


















if False:
    def calculate_penalty_batch(self, new_memory):
        ''' calculate penalty of the batch given original data 
        '''
        penalty_list = []
        for memory in new_memory:
            # get penalty
            penalty_state = memory[5]
            action = memory[1]
            entered_veh = memory[6]
            state = memory[0]
            penalty = self.calculate_penalty(penalty_state, action, entered_veh, state)

            # record
            memory[5] = penalty
            penalty_list.append(penalty)

        return penalty_list, new_memory

        # return penalty

    def calculate_penalty(self, penalty_state, action, entered_veh, state):
        ''' calculate penalty of the single memory given original data 
        '''

        penalty = 0
        if self.penalty_type == 'queue':
            penalty = -penalty_state[-1] / 1000

        if self.penalty_type == 'delta_queue':
            # penalty = - ((penalty_state[-1]/300)**4 - (penalty_state[-2]/300)**4)
            # penalty = -((penalty_state[-1] / 300)**3 - (penalty_state[-2] / 300)**3) / 3
            # penalty = ((penalty_state[-2] - penalty_state[-1])/100) * (penalty_state[-1]/200) / 4

            ## 1. buffer delay:  only action > 0.2 can generate delay on the buffer
            if action >= 0.1:
                penalty = ((penalty_state[-2] - penalty_state[-1])/80)* (penalty_state[-1]/600)/4
                penalty = np.clip(penalty, -2, 0.2 )
            else: 
                penalty= 0
            
            ## 2. excessive green light on the buffer for action >0.5
            if action >= 0.5:
                if action * self.max_green > 3 * entered_veh:
                    penalty -= 0.1
        
        ## 3. encourage shorter green light
        # penalty -= action / 10
        # penalty = - action/2
        penalty = -(state[-1] + state[-2])*2
        return penalty
