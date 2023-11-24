from utils.utilize import config
from keras.constraints import NonNeg
from keras.layers import Input, Dense, concatenate, LSTM, Reshape, BatchNormalization, Lambda, Flatten, Concatenate, Multiply, Add
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.initializers import RandomUniform, Constant
from keras.backend import manual_variable_initialization
import keras.backend as K
import tensorflow as tf
import numpy as np
import os
from keras.callbacks import LearningRateScheduler
# from keras.backend.cntk_backend import squeeze
from keras.legacy.layers import Merge
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # kill warning about tensorflow
# tf.logging.set_verbosity(tf.logging.ERROR)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# manual_variable_initialization(True)


class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, lr, tau):
        # Dimensions and Hyperparams
        self.tau, self.lr = tau, lr

    def target_predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.target_model.predict(inp)

    def eval_predict(self, inp):
        """ Predict Q-Values using the evaluate network
        """
        return self.model.predict(inp)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def print_weight(self):
        print('############# critic weights ##########')
        weights = self.model.get_weights()
        print(weights)


class CriticUpper(Critic):
    def __init__(self, inp_dim, out_dim, lr, tau, action_grad=False):
        super().__init__(lr, tau)
        # Build models and target models
        self.env_dim = inp_dim
        self.act_dim = out_dim

        self.model = self.network()
        self.target_model = self.network()

        # compile
        self.model.compile(Adam(self.lr), 'mse')
        self.target_model.compile(Adam(self.lr), 'mse')

        # init
        self.qloss_list = []
        self.last_qloss_list = []

        if action_grad == True:
            # Function to compute Q-value gradients (Actor Optimization)
            self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(
                self.model.output, [self.model.input[1]]))
        # print('Critic network',self.model.summary())
        # print('Critic network',self.target_model.summary())

    def network(self):
        """ Assemble Critic network to predict q-values
        """
        # state = Input((self.env_dim))
        state = Input(shape=(self.env_dim,))
        action = Input(shape=(self.act_dim,))
        # x = Dense(256, activation='relu', kernel_initializer=Constant(0.01))(state)
        # # x = concatenate([Flatten()(x), action])
        # x = concatenate([x, action])
        x = concatenate([state, action])
        # x = Concatenate()([state, action])
        x = Dense(32, activation='relu', kernel_initializer=RandomUniform())(x)
        # x = Dense(64, activation='tanh', kernel_initializer=Constant(0.01))(x) #RandomUniform()
        # x = Dense(128, activation='tanh', kernel_initializer=Constant(-0.03))(x) #RandomUniform()
        # x = Dense(256, activation='tanh', kernel_initializer=Constant(0.01))(x) #RandomUniform()
        x = Dense(16, activation='relu', kernel_initializer=RandomUniform())(x)
        out = Dense(1, activation='linear',
                    kernel_initializer=RandomUniform())(x)
        # out = Dense(1, activation='linear', kernel_constraint=NonNeg(), bias_constraint=NonNeg(), kernel_initializer=Constant(1))(x)
        # out = Dense(1, activation='linear', kernel_constraint=NonNeg(), bias_constraint=NonNeg(), kernel_initializer=RandomUniform(minval=1e-4))(x)

        return Model([state, action], out)

    def save(self, path, best=False):
        # self.model.save_weights(path + 'critic_eval.h5')
        # self.target_model.save_weights(path + 'critic_target.h5')

        # self.model.save(path+ 'critic_eval.h5')
        if best == False:
            self.model.save(path + f"critic_eval_{config['network']}.h5")
            self.target_model.save(
                path + f"critic_target_{config['network']}.h5")

        if best == True:
            self.model.save(path + f"critic_eval_{config['network']}_best.h5")
            self.target_model.save(
                path + f"critic_target_{config['network']}_best.h5")

        # print('############# parameters saved ##########')
        # weights = self.model.get_weights()
        # print(weights)
        # self.model.save(path+ 'critic.h5')
        # print(self.model.get_weights())

    def load_weights(self, path):
        # eval_path = path + 'critic_eval.h5'
        eval_path = path + f"critic_eval_{config['network']}.h5"
        # target_path = path + "critic_target.h5"
        target_path = path + f"critic_target_{config['network']}.h5"

        if os.path.exists(eval_path):
            print("### load start")
            self.model = load_model(eval_path)
            print('load upper level critic eval network SUCCESS')
        else:
            print('load upper level critic eval network Fail')

        if os.path.exists(target_path):
            self.target_model = load_model(target_path)
            print('load upper level critic target network SUCCESS')
        else:
            print('load upper level critic target network Fail')

        # if os.path.exists(path + 'my_model.h5'):
        #     self.my_model = load_model(path + 'my_model.h5')
        #     print('load try my_model network SUCCESS')
        # else:
        #     print('load try my_model network Fail')

    def gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        return self.action_grads([states, actions])

    def train_on_batch(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        # return self.model.train_on_batch([states, actions], critic_target)
        # print(np.hstack([states, actions]))
        # print([states, actions])
        q_loss = self.model.train_on_batch([states, actions], critic_target)
        self.qloss_list.append(q_loss)
        print(f"critic loss: {q_loss}")
        return q_loss


class CriticLower(Critic):
    def __init__(self, lr, tau, action_grad=False):
        super().__init__(lr, tau)
        # Build models and target models
        # self.phase_num = phase_num
        # self.feature_num = feature_num
        # self.inlane_num = inlane_num
        
        ## network params
        self.cell_num = 1
        self.max_phase = 10
        self.action_lane = [1,1,1, 1,1,1, 1,1,1, 1,1,1]
        self.switch = True

        self.model = self.network()
        self.target_model = self.network()

        # compile
        # self.model.compile(Adam(self.lr), 'mse')
        # self.target_model.compile(Adam(self.lr), 'mse')

        # init
        self.qloss_list = []
        self.last_qloss_list = []



        # print('OAM network',self.model.summary())
        # print('Critic network',self.target_model.summary())

    def network(self):
        """ Assemble Critic network to predict q-values
        """
        # lane_state = Input(batch_shape=(None, self.inlane_num,
        #                    self.feature_num), name='inlane_feature')
        # phase_matrix = Input(batch_shape=(
        #     None, self.phase_num, self.inlane_num), name='phase_action_matrix')
        # lane_embeding = Dense(8, activation='relu', kernel_initializer=RandomUniform(
        # ), name='lane_emb_hidden')(lane_state)
        # lane_embeding = Dense(1, activation='linear', kernel_initializer=RandomUniform(
        # ), name='lane_emb_output')(lane_embeding)
        # lane_embeding = Lambda(lambda x: K.squeeze(
        #     x, -1), name='lane_emb-squeeze')(lane_embeding)  # batch * lane
        # phase_lane_value = Multiply(name='phase_lane_value')(
        #     [phase_matrix, lane_embeding])  # batch * phase *lane
        # phase_value = Lambda(lambda x: K.sum(
        #     x, axis=-1), name='phase_value')(phase_lane_value)  # batch * phase

        # model = Model([lane_state, phase_matrix], phase_value)
        # print(model.summary())
        # return model


        cell_veh_num = Input(
            batch_shape=[None, 12, self.cell_num], name="inlane_density")
        cell_veh_speed = Input(
            batch_shape=[None, 12, self.cell_num], name="inlane_speed")
        cell_veh_halting = Input(
            batch_shape=[None, 12, self.cell_num], name="inlane_queue")
        cell_veh_dens = Input(
            batch_shape=[None, 12, self.cell_num], name="inlane_veh_num")

        outlane_dens = Input(batch_shape=[None, 12], name="outlane_density")
        outlane_speed = Input(batch_shape=[None, 12], name="outlane_speed")

        cur_phase = Input(batch_shape=[None, 1], name="cur_phase_idx")
        phase_inlane_mask = Input(
            batch_shape=[None, self.max_phase, 12], name="phase_inlane_mask")
        phase_mask = Input(
            batch_shape=[None, self.max_phase], name="phase_mask")

        ### cell features ###
        cell_features = Lambda(lambda x: K.stack(
            x, axis=-1))([cell_veh_num, cell_veh_speed, cell_veh_halting, cell_veh_dens])
        cell_emb = Dense(1,  activation='linear')(cell_features)
        cell_emb = Lambda(lambda x: K.squeeze(x, axis=-1))(cell_emb)

        ### lane_features ###
        lane_features = Lambda(lambda x: K.stack(
            x, axis=-1))([outlane_dens, outlane_speed])
        lane_emb = Concatenate(axis=2)([cell_emb, lane_features])

        lane_stop_emb = Dense(self.max_phase, activation='relu')(lane_emb)
        lane_stop_value = Dense(1, activation='linear')(lane_stop_emb)

        lane_switch_emb = Dense(self.max_phase, activation='relu')(lane_emb)
        lane_switch_delta_value = Dense(
            1, activation='linear')(lane_switch_emb)
        lane_switch_delta_value = Lambda(K.abs)(lane_switch_delta_value)
        lane_switch_value = Add()([lane_switch_delta_value, lane_stop_value])

        lane_keep_emb = Dense(self.max_phase, activation='relu')(lane_emb)
        lane_keep_delta_value = Dense(1, activation='linear')(lane_keep_emb)
        lane_keep_delta_value = Lambda(K.abs)(lane_keep_delta_value)
        lane_keep_value = Add()([lane_keep_delta_value, lane_switch_value])

        ### lane-phase combination ###
        cur_phase_vec = Lambda(lambda x: K.squeeze(
            K.one_hot(K.cast(x, dtype='int32'), self.max_phase), axis=1))(cur_phase)
        phase_value = Lambda(self._lane_option_combination)(
            [lane_keep_value, lane_switch_value, lane_stop_value, phase_inlane_mask, cur_phase_vec])
        phase_value = Lambda(
            lambda v: v[0] * v[1] - 1e5 * (1 - v[1]))([phase_value, phase_mask])

        inputs = [cell_veh_num, cell_veh_speed, cell_veh_halting, cell_veh_dens,
                  outlane_dens, outlane_speed,
                  cur_phase, phase_inlane_mask, phase_mask]

        q_value = Model(inputs=inputs, outputs=phase_value)
        q_value.compile(optimizer=Adam(), loss='mse')   # invalid, just for procedure

        return q_value

    def _lane_option_combination(self, x):
        ''' Lane_combination with action-option '''
        q_lane_1, q_lane_05, q_lane_0, phase_inlane_mask, cur_phase_vec = x

        q_lane_1 = K.squeeze(q_lane_1, axis=-1)
        q_lane_05 = K.squeeze(q_lane_05, axis=-1)
        q_lane_0 = K.squeeze(q_lane_0, axis=-1)

        #### phase combination ####
        value_phase_keep = []
        value_phase_switch = []

        for p in range(self.max_phase):
            phase_vector = K.cast(phase_inlane_mask[:, p, :], dtype='float32')
            action_lane = K.cast(self.action_lane, dtype='float32')

            phase_q_lane_keep = q_lane_1 * phase_vector + \
                q_lane_0 * (1 - phase_vector)
            phase_q_lane_switch = q_lane_05 * \
                phase_vector + q_lane_0 * (1 - phase_vector)

            value_phase_keep.append(
                K.mean(phase_q_lane_keep * action_lane, axis=1))
            value_phase_switch.append(
                K.mean(phase_q_lane_switch * action_lane, axis=1))

        value_phase_keep = K.stack(value_phase_keep, axis=1)
        value_phase_switch = K.stack(value_phase_switch, axis=1)

        if self.switch == True:
            phase_value = cur_phase_vec * value_phase_keep + \
                (1 - cur_phase_vec) * value_phase_switch
        else:
            phase_value = value_phase_keep  # + 0 * value_phase_switch * cur_phase_vec

        return phase_value

    def train_on_batch(self, states, critic_target, n):
        """ Train the critic network on batch of sampled experience
        """
        # lr decay
        lr = lr=self.lr * (0.99) ** n
        K.set_value(self.model.optimizer.lr, lr)

        # training
        q_loss = self.model.train_on_batch(
            states, critic_target)
        self.qloss_list.append(q_loss)
        # print(f"critic loss: {q_loss}")

        # Print the current learning rate
        # current_lr = K.get_value(self.model.optimizer.lr)
        # print('Epoch:', n, 'Learning rate:', current_lr)


        return q_loss

    def save(self, path):
        ''' save the lower-level weights
        '''
        
        self.model.save_weights(path + 'lower_critic_eval.h5')

        self.target_model.save_weights(path + 'lower_critic_target.h5')

        # self.model.save(path + 'lower_critic_eval.h5')
        # self.target_model.save(path + 'lower_critic_target.h5')

        # print('############# parameters saved ##########')
        # weights = self.model.get_weights()
        # print(weights)
        # self.model.save(path+ 'critic.h5')
        # print(self.model.get_weights())

    def load_weights(self, path):
        ''' load critic weights of lower level: eval & target networks
        '''
        path_eval = os.path.join(path, 'lower_critic_eval.h5')
        path_target = os.path.join(path, 'lower_critic_target.h5')

        if os.path.exists(path_eval):
            self.model.load_weights(path_eval)
            print('Lower: critic eval network LOAD SUCCESS')
        else:
            print('Lower: critic eval network LOAD FAIL')


        if os.path.exists(path_target):
            self.target_model.load_weights(path_target)
            print('Lower: critic target network LOAD SUCCESS')
        else:
            print('Lower: critic target network LOAD FAIL')

