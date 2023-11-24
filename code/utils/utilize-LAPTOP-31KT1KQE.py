import matplotlib.pyplot as plt
import numpy as np
from sumolib import checkBinary
import os
import sys
import pickle
import xml.etree.cElementTree as ET
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import sys
sys.path.append("../network")


def import_train_configuration():
    """
    Read the config file regarding the training and import its content
    """

    config = {}

    # [test purpose]
    config['test_purpose'] = 'Continue to train with less penalty'
    config['network'] = 'Grid' # 'Grid' 'Bloomsbury'

    # [Perimeter mode]
    ''' DQN and Static: Must be centralize
        DDPG: Can be centralized & decentralized
    '''
    config['mode'] = 'test' # 'test' 'train'
    config['upper_mode'] = 'MaxPressure'  # 'Static' # 'DDPG' #'DQN' # 'Expert' # 'MaxPressure' # 'C_DQN' # 'PI'
    config['lower_mode'] = 'MaxPressure'  # 'FixTime'  #'OAM' # 'MaxPressure'
    config['peri_action_mode'] = 'centralize' # 'decentralize' 'centralize'

    # [PI controller setting]
    if config['upper_mode'] == 'PI':
        if config['network'] == 'Grid':
            config['accu_critic'] = 200
            config['K_p'] = 20
            config['K_i'] = 20

        elif config['network'] == 'Bloomsbury':
            config['accu_critic'] = 800
            config['K_p'] = 20
            config['K_i'] = 20

    # [state]
    if config['network'] == 'Grid':
        config['states'] = [
            'accu', 'accu_buffer', 'future_demand',  'network_mean_speed',     ## general
            'network_halting_vehicles', 'buffer_halting_vehicles',             ## halting
            # 'down_edge_occupancy','buffer_aver_occupancy'                      ## buffer specific
            ]
    if config['network'] == 'Bloomsbury':
        config['states'] = [
            'accu','network_mean_speed',  'future_demand',      ## general
            'network_halting_vehicles']
    # 'accu', accu_buffer,'future_demand', 'entered_vehs','network_mean_speed', 'network_halting_vehicles', 'buffer_halting_vehicles'
    #  'down_edge_occupancy','buffer_aver_occupancy', 


    # [normalization]
    if True:
        config['reward_max'] = 750
        config['entered_veh_max'] = 300
        config['accu_max'] = 3500 if config['network'] == 'Grid' else 6000
        config['accu_buffer_max'] = 3500
        config['max_queue'] = 200
        config['Demand_state_max'] = 2000 if config['network'] == 'Grid' else 5500
        config['network_mean_speed_max'] = 15
        config['PN_halt_vehs_max'] = 3000 if config['network'] == 'Grid' else 5000
        config['buffer_halt_vehs_max'] = 3000
        config['production_control_interval_max'] = 3000
        config['lower_reward_max'] = 2000

    # [network_config]
    if config['network'] == 'Grid':
        config['Node'] = {
            'NodePN': [6, 7, 8, 11, 12, 13, 16, 17, 18],
            'NodePeri': [29, 30, 31, 32]
        }
        config['Edge'] = list(range(0, 96)) 
        config['Edge_Peri'] = sorted([80, 92, 84, 93, 87, 94, 89, 95])
        config['Edge_PN'] = sorted([
            18, 19, 21, 22, 23, 25, 27, 34, 36, 37, 39, 38, 40, 41, 43, 42, 45, 52,
            54, 57, 56, 58, 60, 61
        ])
        config['Edge_PN_out'] = sorted([16, 17, 20, 24, 26, 44, 62, 63, 59, 55, 53,
                                 35])  # the outflows of PN
    elif config['network'] =='Bloomsbury':
        config['Node'] = {
            # 'NodePN': [ 14,13,12,11,9,22,21,20,19,18,17,33,34,35,38],
            'NodePN': [ 13,12,22,21,20,19,18,17,34],
            'NodePeri': [50,51,52,53,54,55]
            # 'NodePeri': [50]
        }
        # read edge data
        df = pd.read_csv("network\Bloomsbury\edge_bloom.csv", sep=';', usecols=['edge_id','edge_PN'])
        
        # config['Edge'] = sorted(df['edge_id'].to_numpy)
        config['Edge'] = sorted(df['edge_id'].values.tolist())
        config['Edge_PN'] = sorted(df[df['edge_PN'] == 1]['edge_id'].values.tolist())
        config['Edge_Peri'] = sorted(df[df['edge_PN'] == 0]['edge_id'].values.tolist())


    # [perimeter_config]
    if config['network'] == 'Grid': # 'Grid':
        config['EdgeCross'] = np.array([92, 93, 94, 95])
        config['Peri_info'] = {
            'P0': {
                'edge': 92,
                'down_edge': 81,
                'buffer_edges': [81, 7, 5],
                'phase_info':
                ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            'P1': {
                'edge': 93,
                'down_edge': 83,
                'buffer_edges': [83, 47, 46],
                'phase_info':
                ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            'P2': {
                'edge': 94,
                'down_edge': 86,
                'buffer_edges': [86, 72, 74],
                'phase_info':
                ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            'P3': {
                'edge': 95,
                'down_edge': 90,
                'buffer_edges': [90, 32, 33],
                'phase_info':
                ['control_phase', 'yellow_phase', '', 'yellow_phase']
            }
        }
    elif config['network'] == 'Bloomsbury': 
        config['Peri_info'] = {
            '15': {
                'edge': 1201,
                'down_edge': 1050,
                'buffer_edges': [1050,1046, 1048],
                'phase_info':
                ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '3': {
                'edge': 1202,
                'down_edge': 1010,
                'buffer_edges': [1010,1036,1035,1034],
                'phase_info':
                ['control_phase', 'yellow_phase','', 'yellow_phase']
            },
            '8': {
                'edge': 1205,
                'down_edge': 1023,
                'buffer_edges': [1023,1027,1028],
                'phase_info':
                ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '32': {
                'edge': 1207,
                'down_edge': 1113,
                'buffer_edges': [1113,1117,1115],
                'phase_info':
                ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '43': {
                'edge': 1209,
                'down_edge': 1150,
                'buffer_edges': [1150,1123,1124,1125],
                'phase_info':
                [ 'control_phase', 'yellow_phase','', 'yellow_phase']
            },
            '39': {
                'edge': 1211,
                'down_edge': 1140,
                'buffer_edges': [1136,1135,1136],
                'phase_info':
                ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
        }

    # [simulation]
    if True:    
        # -1 allcores ## 0 single process ## 1+ multi-process
        config['n_jobs'] = 5
        config['expert_episode'] = 0
        config['gui'] = False  # True  #False  #
        config['infostep'] = 100 # every 10 secs to record info
        config['control_interval'] = 20# 10 sec for lower level
        config['max_steps'] = 10800  # 10800 # 1000 # 6000
        # config['green_duration'] = green_duration
        config['min_green'] = 5.
        config['yellow_duration'] = 5.
        config['cycle_time'] = 100.
        config['max_green_duration'] = 80
        config['max_green'] = config[
            'cycle_time'] - 2 * config['yellow_duration'] - config['min_green']
        config['flow_rate'] = 0.9  # secs per vehicles
        # config['act_range'] = config['max_green'] // config['flow_rate'] * config[
        #     'EdgeCross'].shape[0]
        config['splitmode'] = 1  # 0 = waittime, 1 = waitveh

    # [demand]
    if True:
        config['DemandConfig'] = [
            {
                'DemandType': 'outin',
                'FromRegion': 'NodePeri',
                'ToRegion': 'NodePN',
                'VolumnRange': [[5000, 5000], [5000, 5000], [5000, 5000], [0, 0]],
                # 'Demand_interval': [20, 20, 15, 5],
                'Demand_interval': [36, 36, 27, 9],
                'Multiplier': 1  # 0.1 #0.5 #1.0
            },
            {
                'DemandType': 'inin',
                'FromRegion': 'NodePN',
                'ToRegion': 'NodePN',
                'VolumnRange': [[400, 400], [400, 1500], [1500, 1000], [0, 0]],
                'Demand_interval': [36, 18, 45, 9],
                'Multiplier': 2#1.5  # 0.1 #0.5 #1
            },
            {
                'DemandType': 'inout',
                'FromRegion': 'NodePN',
                'ToRegion': 'NodePeri',
                'VolumnRange': [[800, 800], [800, 1000], [1000, 0], [0, 0]],
                'Demand_interval': [27, 18, 54, 9],
                'Multiplier':2#1.5  # 0.1 #0.5 #1
            }
        ]
        config['DemandNoise'] = {'noise_mean': 0, 'noise_variance': 0.1}
        config['Demand_interval_sec'] = 100  # 100sec for each interval
        config['Demand_interval_total'] = 108
        # future steps of demand in the states
        config['Demand_state_steps'] = 3

    # [agent_upper]
    if True:
        config['reward_delay_steps'] = 0
        config['penalty_delay_steps'] = 0
        config['multi-step'] = 3
        config['rl_type'] = 'off-policy'  # 'off-policy' 'on-policy'

        if config['rl_type'] == 'off-policy':  # off-policy
            if config['multi-step'] == 1:  # real off-policy
                config['buffer_size'] = int(5000)
            else:  # relatively shorter
                config['buffer_size'] = int(5000)
        else:  # on-policy
            config['buffer_size'] = int(108 * max([config['n_jobs'], 1]))

        config['penalty_type'] = 'None'  # 'queue'  # 'delta_queue'
        config['epsilon'] = 0.9
        config['explore_decay'] = 0.9
        config['explore_lb'] = 0.01
        config['reward_normalization'] = False  # True
        config['sample_mode'] = 'random'  # balance # random
        config['state_steps'] = 1  # number of stacked steps for the states
        config['gamma_multi_step'] = 0.9
        config['gamma'] = 0.9
        config['lr_C'] = 5e-4
        config['lr_C_decay'] =.98
        config['lr_C_lb'] = 1e-4

        config['lr_A'] = 1e-4
        config['tau'] = 1e-2
        config['batch_size'] = 128 #64  # 128
        config['total_episodes'] = 50 #1000 # 500
        config['online'] = False  # True #False
        # config['input_dim'] = len(config['states']) * config['state_steps']
        if config['peri_action_mode'] == 'centralize':
            config['act_dim'] = 1
        else:
            config['act_dim'] = len(config['Peri_info'])

        # config['act_range'] = config['max_green'] * len(config['Peri_info'])
        config['reuse_time'] = 2
        # config['action_interval'] = np.array(action_interval)
        # config['initial_std'] = 0.9
        config['target_update_freq'] = 5  # >=1
    
    # [agent_lower]
    if True:
        config['lr_C_lower'] = 1e-4
        config['epsilon_lower'] = 0.5
        config['explore_decay_lower'] = 0.8
        config['buffer_size_lower'] = int(25*max(config['n_jobs'],1)*(config['max_steps']/config['control_interval'])*2)  # junc*n_jobs*intervals*epis
        config['gamma_lower'] = 0.85

    # [dir]
    config['savefile_dir'] = "output/"
    config['models_path_name'] = "output\\model"
    config['plots_path_name'] = 'output/plots'
    config['cache_path_name'] = 'output/cache'

    if config['network'] == 'Grid':
        config['sumocfg_file_name'] = 'network/GridBuffer/GridBuffer.sumocfg'
        config['edgefile_dir'] = "network/GridBuffer/GridBuffer.edg.xml"
        config['netfile_dir'] = "network/GridBuffer/GridBuffer.net.xml"
        config['routefile_dir'] = "network/GridBuffer/GridBuffer.rou.xml"
        config['outputfile_dir'] = "network/GridBuffer/EdgeMesurements.xml"
        config['queuefile_dir'] = "network/GridBuffer/queue.xml"
        
        # tls configure 
        if config['lower_mode'] == 'OAM':  # 'FixTime'  #'OAM' # 'MaxPressure'
            config['tls_config_name'] = './code/tls_new1.pkl'
        elif config['lower_mode'] == 'MaxPressure':
            config['tls_config_name'] = './code/tls_new07.pkl'
    
    if config['network'] == 'Bloomsbury':
        config['sumocfg_file_name'] = 'network/Bloomsbury/Bloomsbury.sumocfg'
        config['netfile_dir'] = "network/Bloomsbury/Bloomsbury.net.xml"
        config['outputfile_dir'] = "network/Bloomsbury/EdgeMesurements.xml"
        config['edgefile_dir'] = "network/Bloomsbury/Bloomsbury.edg.xml"
        config['routefile_dir'] = "network/Bloomsbury/Bloomsbury.rou.xml"
        config['queuefile_dir'] = "network/Bloomsbury/queue.xml"

                # tls configure 
        if config['lower_mode'] == 'OAM':  # 'FixTime'  #'OAM' # 'MaxPressure'
            config['tls_config_name'] = './code/tls_bloom_1.pkl'
        elif config['lower_mode'] == 'MaxPressure':
            config['tls_config_name'] = './code/tls_bloom_05.pkl'




    return config

def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the svisual mode
    if gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [
        sumoBinary,
        "-c",
        os.path.join(sumocfg_file_name),
        "--no-step-log", "true",
        "--verbose", "true",
        "--no-warnings","true",
        # "--no-internal-links","true",
        # "--waiting-time-memory",
        # str(max_steps),
        # "--error-log","network/GridBuffer/error.txt"
        "--time-to-teleport", "120",
        "--queue-output", config['queuefile_dir']
    ]

    return sumo_cmd

def get_NodeConfig(edgefile_dir):
    ''' read the edge file 'edg.xml', obtain the indege and outedge of each node
    '''
    tree = ET.ElementTree(file=edgefile_dir)
    root = tree.getroot()
    NodeConfig = {}
    for child in root:
        edge_id = int(child.attrib['id'])
        From_Node = int(child.attrib['from'])
        To_Node = int(child.attrib['to'])

        # add From_Node
        if From_Node not in NodeConfig:
            NodeConfig[From_Node] = {'InEdge': [], 'OutEdge': []}

        NodeConfig[From_Node]['OutEdge'].append(edge_id)

        # add From_Node
        if To_Node not in NodeConfig:
            NodeConfig[To_Node] = {'InEdge': [], 'OutEdge': []}

        NodeConfig[To_Node]['InEdge'].append(edge_id)

    return NodeConfig

############### Path ###################
def set_train_path(path_name, type):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    path_name = os.path.join(os.getcwd(), path_name, '')
    os.makedirs(os.path.dirname(path_name), exist_ok=True)

    dir_content = os.listdir(path_name)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    path_name = os.path.join(path_name, f'{type}_' + new_version, '')
    os.makedirs(os.path.dirname(path_name), exist_ok=True)

    if type == 'plot':
        ## test
        path_name_test = os.path.join(path_name, 'test', '')
        os.makedirs(os.path.dirname(path_name_test), exist_ok=True)
        
        ## critic
        path_name_critic = os.path.join(path_name, 'critic', '')
        os.makedirs(os.path.dirname(path_name_critic), exist_ok=True)

        ## metric
        path_name_metric = os.path.join(path_name, 'metric', '')
        os.makedirs(os.path.dirname(path_name_metric), exist_ok=True)

        ## explore
        path_name_explore = os.path.join(path_name, 'explore', '')
        os.makedirs(os.path.dirname(path_name_explore), exist_ok=True)
        
        ## model
        path_name_model = os.path.join(path_name, 'model', '')
        os.makedirs(os.path.dirname(path_name_model), exist_ok=True)

    return path_name, path_name_model

def set_test_path(path_name):
    ''' set the path to load the well-trained model
    '''
    path_name = os.path.join(os.getcwd(), path_name, '')
    return path_name

################# PLOT #######################
def plot_MFD(config, accu, flow, aggregate_time, e, reward, n_jobs, reward_lower):
    ''' Plot MFD of each simulation
    '''
    # unit reform
    # throughput = np.array(throughput) / aggregate_time * 3600
    plt.xlabel('acc(veh)')
    plt.ylabel('outflow(veh/h)')
    plt.title(f'MFD (episode{e+1})')
    plt.scatter(accu, flow)
    plt.xlim((0., config['accu_max']))
    plt.ylim(0., 1000)
    # plt.plot(x1, y1, label='整体路网基本图')
    # plt.plot(x2, y2, label='子路网基本图')
    # plt.legend()
    # plt.show()
    if e % n_jobs == 0:
        plt.savefig(
            f"{config['plots_path_name']}test\e{np.around((e)/n_jobs+1, 0)}_MFD_{np.around(reward, 2)}_{np.around(reward_lower, 2)}.png")

    else:
        plt.savefig(
            f"{config['plots_path_name']}explore\e{e+1}_MFD_{np.around(reward, 2)}_{np.around(reward_lower, 2)}.png")
    plt.close()

def plot_flow_MFD(config, accu, flow, e, n_jobs, traci = False):
    # accu = np.array(accu)
    # mean_speed = np.array(mean_speed)
    # TTD = accu * mean_speed* aggregate_time  # nveh*m/sec*sec = veh*m
    # flow = TTD / PN_road_length_tot / (aggregate_time/3600)
    # density = accu/(PN_road_length_tot/1000)

    plt.xlabel('density(veh/km)')
    plt.ylabel('flow (veh/h)')
    plt.title(f'Flow-density MFD')
    plt.scatter(accu, flow)
    plt.xlim((0., config['accu_max']))
    plt.ylim(0., 400)
    if traci:
        if e % n_jobs == 0:
            plt.savefig(
                f"{config['plots_path_name']}test\e{np.around((e)/n_jobs+1, 0)}_flow_MFD_traci.png")

        else:
            plt.savefig(
                f"{config['plots_path_name']}e{e+1}_flow_MFD_traci.png")
    else:
        if e % n_jobs == 0:
            plt.savefig(
                f"{config['plots_path_name']}test\e{np.around((e)/n_jobs+1, 0)}_flow_MFD_output.png")

        else:
            plt.savefig(
                f"{config['plots_path_name']}e{e+1}_flow_MFD_output.png")
    plt.close()
    plt.close('all')

def plot_demand(config, demand_config, demand_interval_sec, demand_interval_total):

    
    Demand_Interval = np.array(
        [i * demand_interval_sec for i in range(demand_interval_total)])
    # Demand_Interval = np.reshape(Demand_Interval, (1,-1))
    # plt.figure()
    plt.xlabel('Time(sec)')
    plt.ylabel('Demand(veh/h)')
    plt.title('Demand profile')

    for DemandType in demand_config:
        plt.plot(Demand_Interval,
                 DemandType['Demand'][0],
                 label=f"{DemandType['DemandType']}")

    plt.legend()
    # plt.show()
    plt.savefig(f"{config['plots_path_name']}Demand.png")
    plt.close()

def plot_reward(reward):
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('Reward')
    plt.plot(range(len(reward)), reward, 'o-')
    plt.savefig(f"{config['plots_path_name']}Reward.png")
    plt.close()

def plot_penalty(penalty):
    
    plt.xlabel('episode')
    plt.ylabel('penalty')
    plt.title('buffer_queue along training')
    plt.plot(range(len(penalty)), penalty, 'o-')
    plt.savefig(f"{config['plots_path_name']}penalty.png")
    plt.close()

def plot_obj_reward_penalty(obj, penalty, reward):
    
    # reward
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('reward')
    plt.plot(range(len(reward)), reward, 'o-')
    plt.savefig(f"{config['plots_path_name']}metric/reward.png")
    plt.close()

    # penalty
    plt.xlabel('episode')
    plt.ylabel('penalty')
    plt.title('penalty')
    plt.plot(range(len(penalty)), penalty, 'o-')
    plt.savefig(f"{config['plots_path_name']}metric\penalty.png")
    plt.close()

    # objective
    plt.xlabel('episode')
    plt.ylabel('objective')
    plt.title('total objective')
    plt.plot(range(len(obj)), obj, 'o-')
    plt.savefig(f"{config['plots_path_name']}metric\objective.png")
    plt.close()

    # together
    plt.xlabel('episode')
    plt.ylabel('objective')
    plt.title('total objective')
    plt.plot(range(len(obj)), obj, 'ko-', label=f"objective")
    plt.plot(range(len(reward)), reward, 'bo-', label=f"reward")
    plt.plot(range(len(penalty)), penalty, 'go-', label=f"penalty")
    plt.legend()
    plt.savefig(f"{config['plots_path_name']}metric\objective_together.png")
    plt.close()

def plot_accu_critic(accu_list):
    
    # accu_critic
    plt.xlabel('episode')
    plt.ylabel('n_critical')
    plt.title('critical accumulation')
    plt.plot(range(len(accu_list)), accu_list, 'o-')
    plt.savefig(f"{config['plots_path_name']}metric/n_critic.png")
    plt.close()


def plot_computime(computime):
    plt.xlabel('episode')
    plt.ylabel('computational time (secs)')
    plt.title('computational time of each episode')
    plt.plot(range(len(computime)), computime, 'o-')
    plt.savefig(f"{config['plots_path_name']}metric\Computational time.png")
    plt.close()

def plot_accu(config, accu, throughput, buffer_queue, e):
    ''' Plot progression of accumulation and throughput of each simulation
    '''
    plt.subplot(3, 1, 1)
    # plt.xlabel('cycle')
    plt.ylabel('accumulation (veh)')
    plt.title(f'progression of (episode{e+1})')
    plt.plot(range(len(accu)), accu, 'o-', label=f"accumulation")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.xlabel('cycle')
    plt.ylabel('throughput (veh/cycle)')
    plt.plot(range(len(throughput)), throughput, 'g>-', label=f"throughput")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.xlabel('cycle')
    plt.ylabel('vehicle number (veh/cycle)')
    plt.plot(range(len(buffer_queue)), buffer_queue,
             'k>-', label=f"buffer queue")
    plt.legend()

    plt.savefig(f"{config['plots_path_name']}e{e+1}_accu_throuput.png")
    plt.close()

def plot_critic_loss(critic_loss, level, mode):  
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.title('Critic loss')
    plt.plot(range(len(critic_loss)), critic_loss)
    plt.savefig(f"{config['plots_path_name']}critic\{level}_{mode}_CriticLoss.png")

    plt.close()

def plot_critic_loss_cur_epis(critic_loss, cur_epis, lr):  
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.title('Critic loss')
    plt.plot(range(len(critic_loss)), critic_loss)
    plt.savefig(f"{config['plots_path_name']}critic\CriticLoss_e{cur_epis}.png")

    plt.close()

def plot_last_critic_loss(last_critic_loss):
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.title('Last Critic loss')
    plt.plot(range(len(last_critic_loss)), last_critic_loss)
    plt.savefig(f"{config['plots_path_name']}critic\LastCriticLoss.png")
    plt.close()

def plot_throughput(throughput):
    ''' Plot throughput in the training for each episode
    '''
    sum_throughput = [sum(i) for i in throughput]
    plt.xlabel('epoch')
    plt.ylabel('throughput (veh/hour)')
    plt.title('Throughput')
    plt.plot(range(len(sum_throughput)), sum_throughput, 'o-')
    plt.savefig(f"{config['plots_path_name']}metric/throughput.png")
    plt.close()

def plot_actions(config, actions, actions_excuted, e, action_type, n_jobs):

    # if peri_action_mode =='centralize' or action_type == 'Expert':
    #     ''' for centralized actions
    #     '''
    #     # plt.xlabel('')
    #     plt.ylabel('allowed vehicles')
    #     plt.title(f'action v.s. executed (episode{e+1})')
    #     plt.bar(range(len(actions)), actions)
    #     plt.bar(range(len(actions)), expert_action_list)
    #     plt.plot(range(len(actions_excuted)),
    #             actions_excuted,
    #             'ko-',
    #             label=f"excuted actions")
    #     plt.legend()
    #     plt.ylim((0., config['max_green'] * 4))
    #     # plt.show()
    #     plt.savefig(f"{config['plots_path_name']}e{e+1}_actions.png")
    #     plt.close()
    
    # else:
    ''' for decentralized actions
    '''
    actions = np.array(actions)
    actions_cum = actions.cumsum(axis=1)
    
    category_names = list(config['Peri_info'].keys())
    if action_type == 'Expert':
        category_colors = plt.get_cmap('seismic')(np.linspace(0.15, 0.85, actions_cum.shape[1]))
    else:
        category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, actions_cum.shape[1]))

    fig, ax = plt.subplots()
    ax.yaxis.set_visible(True)
    ax.set_ylim(0, config['max_green']*len(config['Peri_info']))

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        heights = actions[:, i]
        # 取第一列数值
        starts = actions_cum[:, i] - heights
        # 取每段的起始点
        ax.bar(range(len(actions)), heights, bottom=starts,
                label=colname, color=color)
        # xcenters = starts + heights / 2
        # r, g, b, _ = color
        # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # for y, (x, c) in enumerate(zip(xcenters, heights)):
        #     ax.text(y, x, str(int(c)), ha='center', va='center',
        #             color=text_color, rotation = 90)
        # fig.savefig(f"{config['plots_path_name']}e{e+1}_actions.png")
    plt.plot(range(len(actions_excuted)),
            actions_excuted,
            'ko-',
            label=f"excuted actions")
    ax.legend()
    if action_type == 'Expert':
        fig.savefig(f"{config['plots_path_name']}explore\e{e+1}_actions_expert.png")
    else:
        if e % n_jobs == 0:
            fig.savefig(f"{config['plots_path_name']}test\e{np.around((e)/n_jobs+1, 0)}_actions.png")
        else:
            fig.savefig(f"{config['plots_path_name']}explore\e{e+1}_actions.png")
    plt.close()

def plot_q_value(q_value):
    plt.xlabel('epoch')
    plt.ylabel('q_value')
    plt.title('Actor loss')
    plt.plot(range(len(q_value)), q_value)
    plt.savefig(f"{config['plots_path_name']}critic\ActorLoss.png")
    plt.close()

def plot_q_value_improve(q_value_improve):
    plt.xlabel('epoch')
    plt.ylabel('q_value_improve')
    plt.title('q_value improvement of each update')
    plt.plot(range(len(q_value_improve)), q_value_improve)
    plt.savefig(f"{config['plots_path_name']}critic\q_improve.png")
    plt.close()

def write_log(config):
    with open(f"{config['plots_path_name']}A-readme.txt", "a") as file:
        print(f"PURPOSE = {config['test_purpose']}\n", file=file)
        print(f"Mode = {config['mode']}\n", file=file)
        print(f"Upper_Mode = {config['upper_mode']}\n", file=file)
        print(f"Lower_Mode = {config['lower_mode']}\n", file=file)
        print(f"RL-TYPE = {config['rl_type']}\n", file=file)
        print(f"STATE = {config['states']}\n", file=file)
        print(f"Multi-step = {config['multi-step']}\n", file=file)
        print(f"Expert episodes = {config['expert_episode']}\n", file=file)
        print(f"Buffer size = {config['buffer_size']}\n", file=file)
        print(f"Sample_mode = {config['sample_mode']}\n", file=file)
        print(f"Penalty_type = {config['penalty_type']}\n", file=file)
        # print(f"STACKED_STEPS = {config['state_steps']}\n", file=file)
        print(f"DEMAND = {config['DemandConfig']}\n", file=file)
        print(f"Gamma = {config['gamma']}\n", file=file)
        print(f"Gamma_multi_step = {config['gamma_multi_step']}\n", file=file)
        print(f"LR_CRITIC = {config['lr_C']}\n", file=file)
        print(f"LR_CRITIC_DECAY = {config['lr_C_decay']}\n", file=file) 
        print(f"LR_ACTOR = {config['lr_A']}\n", file=file)
        print(f"TAU = {config['tau']}\n", file=file)
        print(f"Reuse time = {config['reuse_time']}\n", file=file)
        print(f"Epsilon = {config['epsilon']}\n", file=file)
        print(f"Epsilon_decay = {config['explore_decay']}\n", file=file)
        print(
            f"CREDIT ASSIGHMENT_reward_delay_steps= {config['reward_delay_steps']}\n", file=file)
        print(
            f"CREDIT ASSIGHMENT_penalty_delay_steps= {config['penalty_delay_steps']}\n", file=file)
        print(
            f"Target network update frequency= {config['target_update_freq']}\n", file=file)
        print(
            f"Reward normalization in batch replay= {config['reward_normalization']}\n", file=file)

def plot_tsc_delay(config, tsc_all, e, n_jobs):
    ''' plot delay for each junction
    '''
    plt.xlabel('signals(sec)')
    plt.ylabel('Delay(veh m/s)')
    plt.title('Delay profile')
    delay= {}
    for t_id, t_value in tsc_all.items():
        delay[t_id] = sum(t_value.ep_rewards)/1e5

    # delay=sorted(delay)
    sorted_tuples = sorted(delay.items(), key=lambda item: item[1])
    sort_delay = {k: v for k, v in sorted_tuples}
    plt.bar(sort_delay.keys(), sort_delay.values())

    # plt.legend()
    # plt.show()
    if e % n_jobs == 0:
        plt.savefig(
            f"{config['plots_path_name']}test\e{np.around((e)/n_jobs+1, 0)}_delay.png")

    else:
        plt.savefig(
            f"{config['plots_path_name']}e{e+1}_delay.png")
    plt.close()

def plot_lower_reward_epis(reward_epis):
    # reward
    plt.xlabel('episode')
    plt.ylabel('lower level reward')
    plt.title('reward')
    plt.plot(range(len(reward_epis)), reward_epis, 'o-')
    plt.savefig(f"{config['plots_path_name']}metric/lower_reward.png")
    plt.close()

def plot_phase_mean_time(config, controled_light, tsc, e, n_jobs):
    ''' plot the mean time of each phase of each tsc
    '''
    phase_time = {}
    ## calculate mean phase time
    for tl_id in controled_light:
        phase_time[tl_id] = np.mean(tsc[tl_id].phase_time_list)
    
    # plt.bar(range(len(phase_time)), phase_time.values())
    plt.bar(phase_time.keys(), phase_time.values())

    plt.xlabel('tl_id')
    plt.ylabel("phase mean time (sec)")
    plt.title(f"phase mean time of episode {e}")
    # for a, b in phase_time.items():
    #     plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=11)

    if e % n_jobs == 0:
        plt.savefig(
            f"{config['plots_path_name']}test\e{np.around((e)/n_jobs+1, 0)}_phase_mean_time.png")

    else:
        plt.savefig(
            f"{config['plots_path_name']}e{e+1}_phase_mean_time.png")
    plt.close()

def plot_flow_progression(config, flow,e, n_jobs):
    plt.subplot(2, 1, 1)
    plt.ylabel('flow')
    plt.title(f'flow progression of (episode{e+1})')
    plt.plot(range(len(flow)), flow, 'o-')
    plt.ylim((0., 350))

    # plt.legend()

    plt.subplot(2, 1, 2)
    plt.ylabel('total flow')
    # plt.title(f'penalty credit assignment of (episode{e+1})')
    plt.plot(range(len(flow)), np.cumsum(flow/400), 'o-')
    plt.ylim((0., 60))

    # plt.legend()

    if e % n_jobs == 0:
        plt.savefig(
            f"{config['plots_path_name']}test\e{np.around((e)/n_jobs+1, 0)}_flow_progression.png")

    else:
        plt.savefig(
            f"{config['plots_path_name']}e{e+1}_flow_progression.png")
    plt.close()

def plot_peri_waiting(config, peri_waiting_tot, peri_waiting_mean, e, n_jobs):
    ''' Plot perimeter waiting time during the simulation
    '''
    sum_peri_waiting = np.sum(peri_waiting_tot,1)/3600
    plt.xlabel('time')
    plt.ylabel('Perimeter delay (hour)')
    plt.title('Total perimeter delay in each cycle')
    plt.plot(range(len(sum_peri_waiting)), sum_peri_waiting)
    if e % n_jobs == 0:
        plt.savefig(
            f"{config['plots_path_name']}test\e{np.around((e)/n_jobs+1, 0)}_periwait_tot.png")

    else:
        plt.savefig(
            f"{config['plots_path_name']}explore\e{e+1}_periwait_tot.png")
    plt.close()

    plt.xlabel('time')
    plt.ylabel('Delay (veh.sec)')
    plt.title('Average perimeter delay in each cycle')
    plt.plot(range(len(peri_waiting_mean)), peri_waiting_mean)
    if e % n_jobs == 0:
        plt.savefig(
            f"{config['plots_path_name']}test\e{np.around((e)/n_jobs+1, 0)}_periwait_mean.png")

    else:
        plt.savefig(
            f"{config['plots_path_name']}explore\e{e+1}_periwait_mean.png")
    plt.close()

################# save #######################
def save_data_train_upper(agent_upper, agent_lower):
    ''' save objective, reward, penalty, throughput along the training process of upper agents
    '''
    data_train = {}

    ## upper
    # data_train['obj_epis'] = agent_upper.cul_obj #obj_epis
    # data_train['reward_epis'] = agent_upper.cul_reward #reward_epis
    # data_train['penalty_epis'] = agent_upper.cul_penalty #
    
    # data_train['reward_epis_all'] = agent_upper.reward_epis_all #
    # data_train['penalty_epis_all'] = agent_upper.penalty_epis_all #

    # # data_train['throughput_epis'] = agent_upper.throughput_episode# throughput_epis
    # data_train['accu_epis'] = agent_upper.accu_episode# accu_epis
    # data_train['flow_epis'] = agent_upper.flow_episode# 
    # data_train['speed_epis'] = agent_upper.speed_episode# 
    # data_train['TTD_epis'] = agent_upper.TTD_episode# 
    # data_train['PN_waiting_episode'] = agent_upper.PN_waiting_episode
    # data_train['entered_vehs_episode'] = agent_upper.entered_vehs_episode

    if agent_upper.accu_crit_list:
        agent_upper.record_epis['accu_crit_list'] = agent_upper.accu_crit_list# throughput_epis
    if agent_upper.mfdpara_list:
        agent_upper.record_epis['mfdpara_list'] = agent_upper.mfdpara_list# throughput_epis
    
    if agent_upper.peri_mode in ['DQN', 'DDPG', 'C_DQN']:
        agent_upper.record_epis['critic_loss'] = agent_upper.critic.qloss_list # critic_loss
        agent_upper.record_epis['last_critic_loss'] = agent_upper.critic.last_qloss_list # critic_loss
        
    data_train['record_epis'] = agent_upper.record_epis
    data_train['best_epis'] = agent_upper.best_epis
    

    ## lower
    data_train['reward_lower'] = agent_lower.ep_reward_all# throughput_epis


    save_dir = config['models_path_name'] + 'data_'+ config['mode'] +'_'+ agent_upper.peri_mode +'_'+ agent_lower.lower_mode +'.pkl'
    with open(save_dir, 'wb') as f:
        pickle.dump([data_train], f)

    print('###### Data save: Success ######')
    
def save_config(config):
    np.save(f"{config['models_path_name']}config.npy", config)

class Test:
    def __init__(self):
        self.obj_list = []
        self.reward_list = []
        self.penalty_list = []
        self.accu_list = []
        self.throu_list = []
        self.action_list = []
    
    def record_data(self,cumul_obj, cumul_reward, cumul_penalty, accu_episode, throughput_episode, actions):
        self.obj_list.append(cumul_obj)
        self.reward_list.append(cumul_reward)
        self.penalty_list.append(cumul_penalty)
        self.accu_list.append(accu_episode)
        self.throu_list.append(throughput_episode)
        self.action_list.append(actions)

    def save_data_test(self):
        ''' save objective, reward, penalty, throughput, actions along the testing process  
        '''
        data_test = {}

        data_test['obj_list'] = self.obj_list
        data_test['reward_list'] = self.reward_list
        data_test['penalty_epis'] = self.penalty_list
        data_test['accu_list'] = self.accu_list
        data_test['throu_list'] = self.throu_list
        data_test['action_list'] = self.action_list

        save_dir = config['models_path_name'] + 'data_test.pkl'
        with open(save_dir, 'wb') as f:
            pickle.dump([data_test], f)

        print('###### Data save: Success ######')
        


config = import_train_configuration()

if __name__ == "__main__":
    save_config(config)
    write_log(config)
