#!/usr/bin/env python
'''
#######################################################################
# 
#   @file:   check_stats.py
#
#   @author: Isaiah Helt (ihelt3@gatech.edu)
#
#   @brief:  check RAPTOR output statistics
#
#######################################################################
'''

# =======================================================================
#   IMPORTS
# =======================================================================
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# =======================================================================
#   GLOBALS
# =======================================================================

OUTPUT_PATHS = ['~/Documents/01_Academia/Research/02_PROJECTS/01_JICF/01_CFD/04_RESULTS/04_DUCT_ONLY/04_COARSE_TEST_DOMAIN/Output/',
                '~/Documents/01_Academia/Research/02_PROJECTS/01_JICF/01_CFD/04_RESULTS/04_DUCT_ONLY/03_NEW_DOMAIN/1p08m/Output']
# OUTPUT_PATH = '~/Documents/01_Academia/Research/02_PROJECTS/01_JICF/01_CFD/04_RESULTS/04_DUCT_ONLY/03_NEW_DOMAIN/720k/Output'

NAMES = ['Coarse', 'Fine']

# =======================================================================
#   CLASSES
# =======================================================================


# =======================================================================
#   FUNCTIONS
# =======================================================================

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# <function name>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def parse_dat(file_path):
    """
    Parse RAPTOR .dat statistics file.

    Args:
        file_path (str): path to stats file

    Returns:
        stats (dict): 
    """

    data_start = None

    # Step 1 — Identify data block start and stat names
    stats = {}
    with open (file_path, 'r') as f:
        for i, line in enumerate(f):

            line = line.strip()
            if i == 0:
                continue

            if line.strip().startswith('"'):
                stats[line.strip().replace('"', '')] = np.array([])
            elif re.match(r"[+-]?\d", line):
                data_start = i
                break

    # Validate data block found
    if data_start is None:
        raise ValueError("No data block found")

    # Step 2 — Load numeric data in one vectorized call
    data = np.loadtxt(
        file_path,
        delimiter=",",
        skiprows=data_start,
        usecols = range(len(stats))
    )
    
    for i, key in enumerate(stats.keys()):
        stats[key] = data[:, i]

    return stats


# =======================================================================
#   MAIN
# =======================================================================
def main():
    """
    Main function to check RAPTOR output statistics.
    """

    global OUTPUT_PATHS

    if os.path.exists('./figs') == False:
        os.makedirs('./figs')

    data = []
    for i,OUTPUT_PATH in enumerate(OUTPUT_PATHS):
        # Prepare output path
        OUTPUT_PATH = os.path.abspath(os.path.expanduser(OUTPUT_PATH)) + os.sep

        # Parse statistics files
        files = ['qv.min.dat', 'qv.max.dat', 'qv.bar.dat']
        stats = {}
        for file in files:
            name = file.split('.')[1]
            file_path = OUTPUT_PATH + file
            stats[name] = parse_dat(file_path)

        data.append(stats)

    # Plot statistics
    
    variables = list(stats['min'].keys())
    variables.remove('Time')
    var_names = ['rho', 'p', 'u', 'v', 'w', 'T']
    for i,var in enumerate(variables):
        fig,ax = plt.subplots(3,1,figsize=(12,12))
        for j,stats in enumerate(data):
            for k,stat in enumerate(stats.keys()):
                ax[k].plot(stats[stat]['Time'], stats[stat][var], linestyle='-', label=NAMES[j])
                ax[k].set_xlabel('Time')
                ax[k].set_ylabel(var+' '+stat)
                ax[k].grid(True)
                if k == 0:
                    ax[k].legend()
            
        plt.savefig(f'./figs/stats_{var_names[i]}.png')

    

if __name__ == "__main__":
    main()
    


