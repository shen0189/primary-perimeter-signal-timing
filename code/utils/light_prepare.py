import sys, os
# from libsumo.libsumo import lane
p = os.path.dirname(__file__)
print(p)
sys.path.append(os.path.join(p, '../')) 
import networkdata
from networkdata import NetworkData
from utilize import set_sumo, config
import numpy as np
import traci
from math import atan2, pi
import pickle

def get_direction(lane):
    shape = traci.lane.getShape(lane)

    ## get angle
    x1, y1 = shape[0][0] - shape[1][0], shape[0][1] - shape[1][1] # vec of the lane
    x2, y2 = 1, 0     # west direction as reference
    dot = x1*x2+y1*y2
    det = x1*y2-y1*x2
    theta = np.arctan2(det, dot)
    theta = theta if theta>=0 else 2*np.pi+theta
    # print(theta)

    ## get dir
    if theta>=0 and theta < pi/2:
        dir = 'e'   # from east to west
    elif theta>=pi/2 and theta<pi:
        dir = 's'  # from south to north
    elif theta>=pi and theta < pi*(3/2):
        dir = 'w'  # from west to east  
    else:
        dir = 'n'  # from north to south
    
    return dir

def get_inlane_direction(tl):
    ''' get directions of all inlanes
    '''
    tl.incoming_dir = {}
    tl.incoming_dir_edge = {}
    for inlane in tl.incoming_lanes:
        tl.incoming_dir[inlane] = get_direction(inlane)
        inedge = inlane.split('_')[0]
        tl.incoming_dir_edge[inedge] = tl.incoming_dir[inlane]
    return tl

def get_inlane_movements(tl):
    tl.incoming_move={inlane:set() for inlane in tl.incoming_lanes}
    
    moves = traci.trafficlight.getControlledLinks(tl.id)

    ## record the movement
    for move_idx, move in enumerate(moves):
        if move:
            inlane, outlane, _ = move[0]
            # if tl.node['tlsindexdir'][move_idx] != 'r': # not right turn
            tl.incoming_move[inlane].update(tl.node['tlsindexdir'][move_idx])
    
    ## process the single red turn and single T turn
    for lane_move in tl.incoming_move.values():
        if 'r' in lane_move and len(lane_move)>1:  # not single left turn
            # right_turn_idx = lane_move.index('r')
            lane_move.remove('r' )
        if 't' in lane_move and len(lane_move)>1:  # not single t turn
            # t_turn_idx = lane_move.index('t')
            lane_move.remove('t')
    
    tl.incoming_move_edge={inedge:set() for inedge in tl.incoming_edges}
    
    ## get inedge movements
    for inlane in tl.incoming_lanes:
        inedge = inlane.split('_')[0]
        tl.incoming_move_edge[inedge].update(tl.incoming_move[inlane])

    
    return tl


def get_phase_lane_matrix(tl):
    ## init 
    if len(tl.incoming_edges) ==4:
        phases = [ 
            ['s_slr'],    # south_left, north_left
            ['w_slr'],    # south_left, north_left
            ['e_slr'],    # south_left, north_left
            ['n_slr'],    # south_left, north_left
            ['s_slr','n_slr'],
            ['w_slr','e_slr'],
            ]
    if len(tl.incoming_edges) ==3:
        phases = [ 
            ['n_sl'],    #  north_left/straight
            ['w_sl'],    #  west_left/straight
            ['e_sl'],    #  east_left/straight,
            ['s_sl'],    #  south_left/straight, 
            ['n_rl'],    #  north_left/right
            ['w_rl'],    #  west_left/right
            ['e_rl'],    #  east_left/right
            ['s_rl'],    #  south_left/right
            ['n_sl','s_sl'],
            ['w_sl','e_sl'],
            ]
    if len(tl.incoming_edges) ==2:
        phases = [ 
            ['n_slr'],    #  north_left/straight
            ['w_slr'],    #  west_left/straight
            ['e_slr'],    #  east_left/straight,
            ['s_slr'],    #  south_left/straight, 
            ['n_slr','s_slr'],
            ['w_slr','e_slr'],
            ]


    phases = np.array(phases)
    phase_lane_matrix = np.zeros([phases.shape[0],12])
    
    ## complete phase_lane_matrix
    for inlane_idx, inlane in enumerate(tl.incoming_lanes): # each inlane
        lane_dir = tl.incoming_dir[inlane]
        lane_move = tl.incoming_move[inlane]
        for phase_idx, phase in enumerate(phases): # each phase
            for move in phase: # each move
                phase_dir, phase_move= move.split('_')
                phase_move = set(phase_move)
                inedge = inlane.split('_')[0]

                ## the phase can cover the edge move
                if set(tl.incoming_move_edge[inedge]).issubset(phase_move):
                    
                    ## the phase can cover the lane move
                    if lane_dir in phase_dir and lane_move.issubset(phase_move):
                        phase_lane_matrix[phase_idx, inlane_idx] = 1

    phase_capacity_matrix = np.ones([phases.shape[0],12])
    ## complete phase_capacity_matrix
    # if phase_capacity_matrix.shape != phase_lane_matrix.shape:
    #     print('shit')
    for inlane_idx, inlane in enumerate(tl.incoming_lanes): # each inlane
        lane_dir = tl.incoming_dir[inlane]
        lane_move = tl.incoming_move[inlane]
        for phase_idx, phase in enumerate(phases): # each phase
            
            ## the move is possible
            if phase_lane_matrix[phase_idx, inlane_idx]:

                ## more than one phase, discount is needed
                if len(phase)>1 and 'l' in lane_move:
                    phase_capacity_matrix[phase_idx, inlane_idx] = 0.5




    return    phase_lane_matrix, phase_capacity_matrix            

def get_signals(tl):
    '''
        Turn the matrix into signals
    '''
    print(tl.id)
    signals= []
    tlsindex = tl.node['tlsindex']
    tlsindexdir = tl.node['tlsindexdir']
    for phase in tl.matrix:

        ## no signal
        if sum(phase) == 0:
            signal = []
        else:
            ## init signal with all red
            signal = ['r'] * len(tlsindex)

            ## for each move
            for i in range(len(tlsindex)):
                inlane = tlsindex[i]
                dir = tlsindexdir[i]

                inlane_index = tl.incoming_lanes.index(inlane)
            
                if phase[inlane_index] ==1: ## is green
                    if dir=='s':
                        signal[i] = 'G' 
                    else:
                        # left turn, if only one direction is allowed, the left turn is G
                        if sum(phase)<=3:
                            signal[i] = 'G' 
                        else:
                            signal[i] = 'g' 
                elif dir == 'r':
                    signal[i] = 'g'
        str = '' 
        signal = str.join(signal)           
        signals.append(signal)
    return signals






if __name__ == "__main__":


    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])        
    nd = NetworkData(config['netfile_dir'], sumo_cmd)
    netdata = nd.get_net_data()
    tsc, _ = nd.update_netdata()
    print(len(tsc))

    traci.start(sumo_cmd)
    
    MATRIX = {}
    SIGNAL = {}
    CAPACITY = {}

    for tl in tsc.values():

        ## get inlane directions
        tl = get_inlane_direction(tl)

        ## get inlane movements
        tl = get_inlane_movements(tl)


        ## get matrix
        tl.matrix, tl.capacity = get_phase_lane_matrix(tl)
        MATRIX[tl.id] = tl.matrix
        CAPACITY[tl.id] = tl.capacity

        ## get signals
        tl.signals = get_signals(tl)
        SIGNAL[tl.id] = tl.signals


    traci.close()
    print('Done')    

    with open('tls_bloom_05.pkl', 'wb') as f:
        pickle.dump([MATRIX, SIGNAL, CAPACITY], f)
    
    print('##### save success ####')

