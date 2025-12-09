# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#                   "raptorpy @ file:///Users/isaiahhelt/Documents/01_Academia/02_Masters/Research/00_COMMON/raptorpy",
#                   "matplotlib",
#                   "numpy",
#                   "os",
#                   "subprocess"
#                ]
# ///
'''
#######################################################################
# 
#   @file:   inflation_block.py
#
#   @author: Isaiah Helt (ihelt3@gatech.edu)
#
#   @brief:  use raptorpy to create an inflation block mesh
#
#######################################################################
'''

# =======================================================================
#   IMPORTS
# =======================================================================

# NOTE: requires raptorpy environment: source ~/Documents/01_Academia/02_Masters/Research/00_COMMON/raptorpy/.venv/bin/activate
import raptorpy as rp
import numpy as np
import math
import os
import subprocess

import matplotlib.pyplot as plt

# =======================================================================
#   GLOBALS
# =======================================================================

# Solver Parameters
ns = 0  # number of species
Lref = 10.0e-3  # reference length [m]

# multiblock parameters
BLOCK_DIMENSIONAL = True                            # If True, dimensional values are input for MB_LENGTHS and MB_ORIGIN
MB_DIMS = [24,8,6]                                  # number of blocks in each direction [nx,ny,nz] (NOTE: this can change cell count as functions adjust MB_NXYZ to fit cells evenly into blocks)
# MB_LENGTHS = [100, 12.8885, 5.72822]              # ORIGINAL NON-DIM BLOCK SIZE (with injector, Lref = 1.25e-3), block only used half this size in each dimension
# MB_LENGTHS = [0.05, 0.02, 0.005]
# MB_LENGTHS = [50,2,0.5]
MB_LENGTHS = np.array([200.0,100.0,50.0])*1e-3      # Dimensional block size [m]
MB_NXYZ = [30,40,30]                                # number of nodes in each block in each direction [nx,ny,nz]
MB_ORIGIN = [0.0,0.0,0.0]                           # origin of the multiblock domain [x0,y0,z0]

# Inflaction layer parameters
GROWTH_DIMENSIONAL = True       # If True, FIRST_LAYER_THICKNESS and ISO_LENGTH are dimensional values
GROWTH_RATE = 1.2
MAX_LAYERS = 100
# TARGET_HEIGHT = np.inf # target final height of inflation layers
FIRST_LAYER_THICKNESS = 1e-6  # first layer thickness [m]
DIRECTION = 'y'

# Isotropic region parameters (Falls under GROWTH_DIMENSIONAL)
#   Length of isotropic region after inflation layers
#       scalar: same isotropic length in each direction (overrides MB_NXYZ)
#       list: isotropic length in each direction [x,y,z] (overrides MB_NXYZ)
#       None: fill based on MB_NXYZ
ISO_LENGTH = np.array([100,5000,100])*FIRST_LAYER_THICKNESS 
# ISO_LENGTH = [100*FIRST_LAYER_THICKNESS, None, 100*FIRST_LAYER_THICKNESS ]
# ISO_LENGTH = None #100.0*FIRST_LAYER_THICKNESS

# Boundary Conditions
BCS = ['u01','e02','s01','s02','s02','s02']    # [-x face, +x face, -y face, +y face, -z face, +z face]

# Output Options
WRITE_VISUALIZATION = True
OUTPUT_PATH = './test_grid/'

# =======================================================================
#   CLASSES
# =======================================================================


# =======================================================================
#   FUNCTIONS
# =======================================================================

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# plot_multiblock
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def plot_multiblock(mb):
    """
    __summary__: plots a multiblock mesh using matplotlib (not very useful since it's so much data)

    Args:
        mb (raptorpy.multiblock.grid): multiblock grid object to plot

    Returns:
        None
    """

    fig,ax = plt.subplots(subplot_kw={"projection":"3d"})
    for blk in mb:
        ax.scatter(blk.x, blk.y, blk.z, s=1)
    
    plt.show()

    return

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# create_inflation_distribution
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def create_inflation_distribution(
                                    y0,
                                    rate,
                                    max_inflation_layers=None,
                                    max_nodes=None,
                                    total_height=None,
                                    iso_length=None,
                                    n_blocks=None
                                 ):
    """
    __summary__: creates an inflation layer distribution

    Args:
        y0 (float): 
            first layer thickness
        rate (float): 
            growth rate
        max_inflation_layers (int,optional): 
            maximum number of inflation layers
        max_nodes (int): 
            total number of nodes in the inflation direction
        total_height (float, optional): 
            total height in the inflation direction. Defaults to height of max_nodes with growth rate
        iso_length (float, optional): 
            length of isotropic region after inflation layers. Defaults to None.
            NOTE: total_height must be specified if iso_length is used.
        n_blocks (int, optional):
            number of blocks in the inflation direction to ensure proper cell count distribution: mod(cells,n_block) = 0. Defaults to None.


    Returns:
        layer_thicknesses (list): 
            thickness of each cell
        layer_height (list): 
            height of each node (including first node at 0)
    """


    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    # Input Validation

    ''' 
    Cases to consider:
        1) Inflation layers hit max height first
        2) Inflation layers hit max nodes first
        3) Inflation layers hit iso length first
    '''

    if iso_length is None and total_height is None:
        raise Exception("Must specify either max_nodes or iso_length.")

    max_cells = None
    if max_nodes is not None:
        # Subtract max_nodes by 1 to account first node starting at 0
        max_cells = max_nodes - 1
    
    if max_inflation_layers is None:
        max_inflation_layers = np.inf

    # Special considerations for multiblock geometry
    if n_blocks is not None:

        # Correct max_nodes to be multiple of n_blocks
        if max_cells is not None:
            remainder = max_cells % n_blocks
            if remainder != 0:
                max_cells -= remainder
                print(f"Corrected max inflation layer cells to {max_cells} to be multiple of n_blocks={n_blocks}.")

        # Check for case 1, where total height is hit first 
        if total_height is not None:

            # Estimate number of layers needed to hit total height
            est_layers = int( np.log( 1 - ( total_height*(1-rate) ) / y0 ) / np.log(rate) )-1

            # Check if max_inflation_layers, max_cells, or iso_length will be hit first
            adjust_r = True
            if max_inflation_layers is not None:
                if est_layers > max_inflation_layers:
                    adjust_r = False
            if max_cells is not None:
                if est_layers > max_cells:
                    adjust_r = False
            if iso_length is not None:
                max_h = y0 * (rate ** est_layers)
                if max_h > iso_length:
                    adjust_r = False

            if adjust_r:
                # If not multiple of n_blocks, increase total layers to be multiple of n_blocks, then recalculate growth rate to hit total height at appropriate number of cells
                total_layers = est_layers
                remainder = total_layers % n_blocks
                if remainder != 0:
                    total_layers += n_blocks - remainder
                    # calculate the new growth rate to hit the total height with corrected layers
                    rate = find_r_for_height(y0, total_layers, total_height)
                    print(f"Recalculated growth rate to {rate} to exactly hit total height appropriate number of layers based on multiblock dimensions.")


    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    # Calculations

    layer_thicknesses = []
    # Loop through inflation layers
    for i in range(max_inflation_layers):
        thickness = y0 * (rate ** i)
        layer_thicknesses.append(thickness)

        # Check if we have exceeded the total height (if specified)
        if total_height is not None:
            layer_height = sum(layer_thicknesses)
            if layer_height > total_height:
                print(f"Warning: Total height exceeded at layer {i+1}. Stopping layer creation.")
                break

            # Check if next layer will exceed isotropic layer thickness
            if max_cells is not None:
                iso_layer_thickness = (total_height - layer_height) / (max_cells - len(layer_thicknesses))
                if thickness*rate >= iso_layer_thickness:
                    print(f"Isotropic layer thickness reached at layer {i+1} based on total nodes. Stopping layer creation.")
                    break
            elif iso_length is not None:
                if thickness*rate >= iso_length:
                    print(f"Specified Isotropic layer thickness reached at layer {i+1}. Stopping layer creation.")
                    break

    # Creating distribution from the end of the inflation to the total height (if specified)
    if total_height is not None:

        # Calculate remaining height and distribute isotropically
        current_height = sum(layer_thicknesses)
        remaining_height = total_height - current_height

        if remaining_height > 0:

            # If total nodes is specified, distribute remaining layers to satisfy total nodes
            if max_nodes is not None:

                remaining_layers = max_nodes - len(layer_thicknesses)

                # Correct remaining layers to be multiple of n_blocks
                if n_blocks is not None:
                    remainder = remaining_layers % n_blocks
                    if remainder != 0:
                        remaining_layers += n_blocks - remainder

                if remaining_layers > 0:
                    iso_thickness = remaining_height / remaining_layers
                    for _ in range(remaining_layers):
                        layer_thicknesses.append(iso_thickness)
            
            # If iso_length is specified, distribute remaining layers to achieve close to iso_length
            elif iso_length is not None:
                remaining_layers = int( (total_height - current_height ) / iso_length )

                # Correct remaining layers to be multiple of n_blocks
                if n_blocks is not None:
                    total_layers = remaining_layers + len(layer_thicknesses)
                    remainder = total_layers % n_blocks
                    if remainder != 0:
                        remaining_layers += n_blocks - remainder
                        

                iso_thickness = remaining_height / remaining_layers
                for _ in range(remaining_layers):
                    layer_thicknesses.append(iso_thickness)
    
    # Node Positions, accounting for first node at 0
    layer_height = np.insert( np.cumsum(layer_thicknesses), 0, 0.0 )

    return layer_thicknesses, layer_height


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Calculate the growth rate required to hit a target height
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def calc_total_height(r, h0, N):
    '''
    __summary__: calculates the total height of N layers with growth rate r and first layer thickness h0
    
    Args:
        r (float): 
            growth rate
        h0 (float): 
            first layer thickness
        N (int):
            number of layers
            
    Returns:
        float: total height
    '''
    
    if abs(r - 1.0) < 1e-12:
        return h0 * N
    return h0 * (1.0 - r**N) / (1.0 - r)

def find_r_for_height(h0, N, H, r_low=1e-12, r_high=2.0, tol=1e-12, max_iter=200):
    '''
    __summary__: finds the growth rate r needed to achieve a target height H with N layers starting from h0
    NOTE: careful with this function, largely generated with AI assistance
    
    Args:
        h0 (float): 
            first layer thickness
        N (int): 
            number of layers
        H (float): 
            target total height
        r_low (float, optional): 
            lower bound for growth rate search. Defaults to 1e-12.
        r_high (float, optional): 
            upper bound for growth rate search. Defaults to 2
        tol (float, optional): 
            tolerance for convergence. Defaults to 1e-12.
        max_iter (int, optional): 
            maximum number of iterations. Defaults to 200.

    Returns:
        float: growth rate r
    '''

    # Quick checks
    if H <= h0 + 1e-15:
        # If desired height <= first cell, any r->0 works; return small r
        return 0.0
    # Handle r=1 special case if H approx N*h0
    if abs(H - h0*N) < 1e-12:
        return 1.0

    # Expand r_high until total_height(r_high) >= H
    th_high = calc_total_height(r_high, h0, N)
    it = 0
    while th_high < H and it < 200:
        r_high *= 2.0
        th_high = calc_total_height(r_high, h0, N)
        it += 1
    if th_high < H:
        raise ValueError("Could not bracket root: try larger initial r_high or check inputs.")

    low = r_low
    high = r_high
    f_low = calc_total_height(low, h0, N) - H
    f_high = th_high - H

    for i in range(max_iter):
        mid = 0.5*(low + high)
        f_mid = calc_total_height(mid, h0, N) - H
        if abs(f_mid) <= tol:
            return mid
        # Choose side with sign change
        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    # return best estimate
    return 0.5*(low + high)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Distribute block with inflation layers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def distribute_inflation_layers(mb, direction, layer_heights):
    """
    __summary__: distributes block nodes according to inflation layer heights

    Args:
        mb (raptorpy.block.grid): multiblock grid object
        direction (str): direction to apply inflation ('x','y','z')
        layer_heights (list): list of layer heights

    Returns:
        None

    NOTE: block connectivity is defined as 

                    +-----------------------------+
                   /                             /|
                  / |                           / |
                 /                             /  |
                /   |         6               /   |
               /                             /    |
              /     |                       /     |
             /                    4        /      |
            /       |                     /       |
           +-----------------------------+        |
           |        |                    |        |
           |                             |        |
           |        |                    |        |
           |    1                        |    2   |
           |               CUBE          |        |
           |        + - - - - - - - - - -|- - - - +
           |                             |       / 
           |      /       3              |      /  
           |                             |     /   
           |    /                        |    /           z
           |                  5          |   /            ^ 
           |  /                          |  /             |  / y
           |                             | /              | /
           |/                            |/               |/
           +-----------------------------+                +-----> x

    """

    # Determine index based on direction
    idx = {'x':0,'y':1,'z':2}[direction]

    # Get number of nodes in the specified direction
    nodes_in_dir = mb[0].__dict__[direction].shape[idx]

    f1 = str(2*idx+1) # index for face first face 1

    # Get slice in the provided direction
    slice_obj = [slice(None), slice(None), slice(None)]
    slice_obj[idx] = 0  # We only need one slice in the inflation direction

    for blk in mb:
        # Check which number block is in the multiblock array
        iblk_id = blk.nblki-1 # Block ID with zero indexing
        jblk_id = iblk_id
        bottom_found = False
        bottom_idx = 0
        while not bottom_found:
            if mb[jblk_id].connectivity[f1]['connection'] == '0':
                bottom_found = True
            else:
                jblk_id = int(mb[jblk_id].connectivity[f1]['connection']) - 1
                bottom_idx += 1

        # print(f"Block ID: {blk.nblki}, Bottom Block ID: {iblk_id}, Bottom Index: {bottom_idx}")

        # Distribute the layer heights to the block
        height_indices = np.arange( bottom_idx*(nodes_in_dir-1), (bottom_idx+1)*(nodes_in_dir-1)+1 )
        # print(f"bottom index: {bottom_idx}")
        # print(f"nodes in dir: {nodes_in_dir}")
        # print(f"length height indices: {len(height_indices)}")
        for i,h in enumerate(height_indices):
            slice_obj[idx] = i
            slice_tup = tuple(slice_obj)
            mb[iblk_id].__dict__[direction][slice_tup] = layer_heights[h]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Assign Boundary Conditions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def assign_cube_bcs(mb,bcs):
    """
    __summary__: assigns boundary conditions to a cube multiblock mesh

    Args:
        mb (raptorpy.multiblock.grid): multiblock grid object
        bcs (list): list of boundary conditions in order [-x face, +x face, -y face, +y face, -z face, +z face]

    Returns:
        None
    """

    # Face indices for cube connectivity
    face_indices = {
        '-x': '1',
        '+x': '2',
        '-y': '3',
        '+y': '4',
        '-z': '5',
        '+z': '6'
    }

    # Loop through each block and assign BCs
    for blk in mb:
        for i,face in enumerate(['-x','+x','-y','+y','-z','+z']):
            face_id = face_indices[face]
            # Check if this is a boundary face
            if blk.connectivity[face_id]['connection'] == '0':
                blk.connectivity[face_id]['bc'] = bcs[i]

    return


# =======================================================================
#   MAIN
# =======================================================================
def main():

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Call out global parameters, and non-dimensionalize if needed
    global MB_LENGTHS
    global MB_ORIGIN
    global FIRST_LAYER_THICKNESS
    global MB_NXYZ
    global ISO_LENGTH

    # Change ISO_LENGTH to array if needed
    if ISO_LENGTH is not None and isinstance(ISO_LENGTH, float):
        ISO_LENGTH = np.array([ISO_LENGTH]*3)

    # Non-dimensionalize multiblock parameters if needed
    if BLOCK_DIMENSIONAL:
        MB_LENGTHS = [l / Lref for l in MB_LENGTHS]
        MB_ORIGIN = [o / Lref for o in MB_ORIGIN]
    if GROWTH_DIMENSIONAL:
        FIRST_LAYER_THICKNESS = FIRST_LAYER_THICKNESS / Lref

        # Correct non None ISO_LENGTH values
        if ISO_LENGTH is not None:
            for i,il in enumerate(ISO_LENGTH):
                if il is not None:
                    ISO_LENGTH[i] = il / Lref

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Create the inflation distribution

    idx = {'x':0,'y':1,'z':2}[DIRECTION]

    # If ISO_LENGTH is not specified, calculate inflation layers based on MB_NXYZ
    if ISO_LENGTH is None:
        # Grab the total number of layers and total height in the inflation direction
        
        total_height = MB_LENGTHS[idx]
        max_nodes = (MB_NXYZ[idx]-1)*MB_DIMS[idx]+1 # num cells in that direction
        layer_thicknesses, layer_height = create_inflation_distribution(FIRST_LAYER_THICKNESS, 
                                                                        GROWTH_RATE, 
                                                                        MAX_LAYERS, 
                                                                        max_nodes=max_nodes, 
                                                                        total_height=total_height, 
                                                                        n_blocks=MB_DIMS[idx])

    # If ISO_LENGTH is specified, calculate inflation layers based on ISO_LENGTH
    else:
        total_height = MB_LENGTHS[idx]
        layer_thicknesses, layer_height = create_inflation_distribution(FIRST_LAYER_THICKNESS, 
                                                                        GROWTH_RATE, 
                                                                        MAX_LAYERS, 
                                                                        total_height=total_height, 
                                                                        iso_length=ISO_LENGTH[idx],
                                                                        n_blocks=MB_DIMS[idx])

        # Assign MB_NXYZ based on the number of layers created
        for i,dim in enumerate(MB_DIMS):

            if i == idx:
                MB_NXYZ[i] = len(layer_thicknesses) // MB_DIMS[i] + 1
            else:
                MB_NXYZ[i] = int( MB_LENGTHS[i] / (ISO_LENGTH[i]*MB_DIMS[i]) ) + 1

        print('Updated MB_NXYZ:', MB_NXYZ)


    # Give user the option to quit if too many layers are created
    Ncells = np.prod( [ (n-1)*d for n,d in zip(MB_NXYZ, MB_DIMS) ] )
    print(f"Total number of cells in the multiblock mesh will be: {Ncells:,}")
    proceed = input("Proceed? (/n): ")
    if proceed.lower() == 'n':
        print("Exiting...")
        return

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Initialize a multiblock restart object
    nblk = np.prod(MB_DIMS)
    mb = rp.multiblock.restart(nblk, ns)

    # Use raptorpy to create multiblock mesh
    rp.grid.create.multiblock_cube(mb, MB_ORIGIN, MB_LENGTHS, MB_DIMS, MB_NXYZ)


    # Distribute the inflation layers to the multiblock
    distribute_inflation_layers(mb, DIRECTION, layer_height)
    assign_cube_bcs(mb, BCS)

    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Output Grid

    # Create output directory if it doesn't exist
    if os.path.exists(OUTPUT_PATH) == False:
        os.makedirs(OUTPUT_PATH)
    else:
        # Clean the existing output directory
        os.system(f'rm -rf {OUTPUT_PATH}/*')

    # Write the grid files
    if WRITE_VISUALIZATION:
        rp.writers.write_hdf5_grid(mb, OUTPUT_PATH)     # For visualization
        rp.writers.write_hdf5_restart(mb, OUTPUT_PATH)  # For visualization
    rp.writers.write_raptor_grid(mb, OUTPUT_PATH)   # Raptor input grid (g.*)
    rp.writers.write_raptor_conn(mb, OUTPUT_PATH)   # Raptor connectivity input (conn.inp)

    # Write the ernk.inp file
    cwd = os.getcwd()
    cmd = 'source ~/Documents/01_Academia/02_Masters/Research/00_COMMON/raptorpy/.venv/bin/activate;'
    cmd += 'python ~/Documents/01_Academia/02_Masters/Research/00_COMMON/raptorpy/utilities/create_ERNK.py'
    subprocess.Popen(cmd, cwd=OUTPUT_PATH, shell=True, stdout=subprocess.PIPE).wait()
    


if __name__ == "__main__":
    main()
    


