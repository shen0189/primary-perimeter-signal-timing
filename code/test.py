import sumolib
import traci
from utils.utilize import config, plot_MFD
import numpy as np
from envir.perimeter import Peri_Agent
import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt
from collections import deque
import sys, subprocess, os
from itertools import product
import pandas as pd

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    print(os.environ['SUMO_HOME'])
    sys.path.append(tools)
    import sumolib