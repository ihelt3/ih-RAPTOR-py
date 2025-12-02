# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#                   "raptorpy @ file:///Users/isaiahhelt/Documents/01_Academia/02_Masters/Research/00_COMMON/raptorpy",
#                ]
# ///
'''
#######################################################################
# 
#   @file:   write_periodic_box.py
#
#   @author: Isaiah Helt (ihelt3@gatech.edu)
#
#   @brief:  writes conn.inp file for box with periodic BCs
#
#######################################################################
'''

# =======================================================================
#   IMPORTS
# =======================================================================

import raptorpy as rp
import os
import glob

# =======================================================================
#   GLOBALS
# =======================================================================

# NOTE: requires raptorpy environment: source ~/Documents/01_Academia/02_Masters/Research/00_COMMON/raptorpy/.venv/bin/activate

# Inputs
PERIODIC_FACES = ['z']  # faces to make periodic (options: x, y, z)
GRID_DIR = './grid/'  # location of grid directory
OUTPUT_DIR = './z_periodic/'  # directory to write updated conn.inp file


# Globals
# dictionary to map face strings to face indices
face_dict = {
    '-x': 1,
    '+x': 2,
    '-y': 3,
    '+y': 4,
    '-z': 5,
    '+z': 6
}

# =======================================================================
#   CLASSES
# =======================================================================


# =======================================================================
#   FUNCTIONS
# =======================================================================

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# get_grid
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def get_grid(grid_dir):
    """
    gets the raptor grid from specified directory

    Args:
        grid_dir (str): path to grid directory

    Returns:
        mb (raptorpy.multiblock.grid): raptorpy grid object
    """

    # Determine the number of blocks in the grid
    grid_files = glob.glob(grid_dir + 'g.*')
    print(f'Number of blocks in grid: {len(grid_files)}')

    # Read the grid
    mb = rp.multiblock.grid(len(grid_files))
    rp.readers.read_raptor_grid(mb,path=grid_dir)
    rp.readers.read_raptor_conn(mb,GRID_DIR + 'conn.inp')
    
    return mb


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# find_boundary
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def find_boundary(mb, blk_id, face):
    """
    find the block on the boundary in a given direction relative to the given block

    Args:
        mb (raptorpy.multiblock.grid): raptorpy grid object
        blk_id (int): block ID to find boundary for IN RAPTOR INDEXING (1 indexed)
        face (str): face to find boundary on (options: -x, +x, -y, +y, -z, +z)

    Returns:
        boundary_blk_id (int): block ID of the boundary block
    """

    

    f1 = str(face_dict[face])
    jblk_id = blk_id-1 # convert to 0 indexing

    boundary_found = False
    while not boundary_found:
        if mb[jblk_id].connectivity[f1]['connection'] == '0':
            boundary_found = True
        else:
            jblk_id = int(mb[jblk_id].connectivity[f1]['connection']) - 1 # convert to 0 indexing
    
    return jblk_id+1 # convert back to 1 indexing


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# assign_periodicity
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def assign_periodicity(mb, direction):
    """
    find the block on the boundary in a given direction relative to the given block

    Args:
        mb (raptorpy.multiblock.grid): raptorpy grid object
        direction (str): direction to create periodic boundary (options: x, y, z)

    Returns:
        mb (raptorpy.multiblock.grid): raptorpy grid object with updated connectivity

    NOTE: does not account for block orientation, assumes all blocks are aligned the same way
    """

    # dictionary to map direction strings to face indices
    dir_dict = {
        'x': ('-x', '+x'),
        'y': ('-y', '+y'),
        'z': ('-z', '+z')
    }

    # Get face nomenclature
    f_neg, f_pos = dir_dict[direction]
    fwd = face_dict[f_pos]
    aft = face_dict[f_neg]

    # Loop through blocks and assign periodicity
    for blk in mb:
        # Check if this block has a boundary on the negative face
        if blk.connectivity[str(aft)]['connection'] == '0':
            # Find the corresponding block on the positive face
            boundary_blk_id = find_boundary(mb, blk.nblki, f_pos)

            # Update connectivity to make periodic
            blk.connectivity[str(aft)]['connection'] = str(boundary_blk_id)
            blk.connectivity[str(aft)]['bc'] = 'b 01'  # periodic BC
            blk.connectivity[str(aft)]['orientation'] = '123'  # see NOTE in header

            # Also update the corresponding block's connectivity
            mb[boundary_blk_id - 1].connectivity[str(fwd)]['connection'] = str(blk.nblki)
            mb[boundary_blk_id - 1].connectivity[str(fwd)]['bc'] = 'b 01'  # periodic BC
            mb[boundary_blk_id - 1].connectivity[str(fwd)]['orientation'] = '123'  # see NOTE in header

# =======================================================================
#   MAIN
# =======================================================================
def main():

    # Input Checking
    global GRID_DIR
    if not GRID_DIR.endswith(os.sep):
        GRID_DIR += os.sep
    
    # Grab the grid
    mb = get_grid(GRID_DIR)

    # Create periodic boundaries
    for direction in PERIODIC_FACES:
        assign_periodicity(mb, direction)

    '''
    # testing
    test_id = 984
    face = '6'
    print('Block {0:d} connectivity after periodic assignment:'.format(mb[test_id-1].nblki))
    print(mb[test_id-1].connectivity)
    test_connection = int(mb[test_id-1].connectivity[face]['connection'])-1
    print('Block {0:d} connectivity after periodic assignment:'.format(mb[test_connection].nblki))
    print(mb[test_connection].connectivity)
    '''

    # Write the updated connectivity to file
    global OUTPUT_DIR
    if OUTPUT_DIR.endswith(os.sep) == False:
        OUTPUT_DIR += os.sep
    if os.path.exists(OUTPUT_DIR) == False:
        os.makedirs(OUTPUT_DIR)
    else:
        # Clean the existing output directory
        os.system(f'rm -rf {OUTPUT_DIR}*')
    rp.writers.write_raptor_conn(mb, OUTPUT_DIR)

    

if __name__ == "__main__":
    main()
    


