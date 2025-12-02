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
BLOCK_DIMENSIONAL = True # If True, dimensional values are input for MB_LENGTHS and MB_ORIGIN
MB_DIMS = [24,8,6]
# MB_LENGTHS = [100, 12.8885*2.0, 5.72822*2.0]
MB_LENGTHS = [0.05, 0.02, 0.005]
# MB_LENGTHS = [50,2,0.5]
MB_NXYZ = [30,40,30]
MB_ORIGIN = [0.0,0.0,0.0]

# Inflaction layer parameters
GROWTH_DIMENSIONAL = True       # If True, FIRST_LAYER_THICKNESS and ISO_LENGTH are dimensional values
GROWTH_RATE = 1.1
MAX_LAYERS = 100
FIRST_LAYER_THICKNESS = 1e-6  # first layer thickness [m]
DIRECTION = 'y'

# Isotropic region parameters (Falls under GROWTH_DIMENSIONAL)
#   Length of isotropic region after inflation layers
ISO_LENGTH = 50*FIRST_LAYER_THICKNESS # Set to None to fill based on MB_NXYZ, otherwise ISO_LENGTH will override MB_NXYZ

# Boundary Conditions
BCS = ['u01','e02','s01','s02','s02','s02']    # [-x face, +x face, -y face, +y face, -z face, +z face]

# Basic Options
WRITE_VISUALIZATION = True

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
def create_inflation_distribution(y0,rate,max_inflation_layers=None,total_nodes=None,total_height=None,iso_length=None):
    """
    __summary__: creates an inflation layer distribution

    Args:
        y0 (float): 
            first layer thickness
        rate (float): 
            growth rate
        max_inflation_layers (int,optional): 
            maximum number of inflation layers
        total_nodes (int): 
            total number of nodes in the inflation direction
        total_height (float, optional): 
            total height in the inflation direction. Defaults to height of total_nodes with growth rate
        iso_length (float, optional): 
            length of isotropic region after inflation layers. Defaults to None.
            NOTE: total_height must be specified if iso_length is used.


    Returns:
        list: list of layer thicknesses
    """

    # Validate Inputs
    if total_nodes is not None and iso_length is not None:
        raise Exception("Cannot specify both total_nodes and iso_length.")
    elif total_nodes is not None:
        # Subtract total_nodes by 1 to account first node starting at 0
        total_nodes -= 1
    elif iso_length is None and total_height is None:
        raise Exception("Must specify either total_nodes or iso_length.")
    
    if max_inflation_layers is None:
        max_inflation_layers = np.inf

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
            if total_nodes is not None:
                iso_layer_thickness = (total_height - layer_height) / (total_nodes - len(layer_thicknesses))
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
            if total_nodes is not None:
                remaining_layers = total_nodes - len(layer_thicknesses)
                if remaining_layers > 0:
                    iso_thickness = remaining_height / remaining_layers
                    for _ in range(remaining_layers):
                        layer_thicknesses.append(iso_thickness)
            
            # If iso_length is specified, distribute remaining layers to achieve close to iso_length
            elif iso_length is not None:
                remaining_layers = int( (total_height - current_height ) / iso_length )
                iso_thickness = remaining_height / remaining_layers
                for _ in range(remaining_layers):
                    layer_thicknesses.append(iso_thickness)
    
    # Node Positions, accounting for first node at 0
    layer_height = np.insert( np.cumsum(layer_thicknesses), 0, 0.0 )

    return layer_thicknesses, layer_height


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

    # Non-dimensionalize multiblock parameters if needed
    if BLOCK_DIMENSIONAL:
        MB_LENGTHS = [l / Lref for l in MB_LENGTHS]
        MB_ORIGIN = [o / Lref for o in MB_ORIGIN]
    if GROWTH_DIMENSIONAL:
        FIRST_LAYER_THICKNESS = FIRST_LAYER_THICKNESS / Lref
        ISO_LENGTH = ISO_LENGTH / Lref if ISO_LENGTH is not None else None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Create the inflation distribution

    idx = {'x':0,'y':1,'z':2}[DIRECTION]

    # If ISO_LENGTH is not specified, calculate inflation layers based on MB_NXYZ
    if ISO_LENGTH is None:
        # Grab the total number of layers and total height in the inflation direction
        
        total_height = MB_LENGTHS[idx]
        total_nodes = (MB_NXYZ[idx]-1)*MB_DIMS[idx]+1 # num cells in that direction
        layer_thicknesses, layer_height = create_inflation_distribution(FIRST_LAYER_THICKNESS, GROWTH_RATE, MAX_LAYERS, total_nodes, total_height)

    # If ISO_LENGTH is specified, calculate inflation layers based on ISO_LENGTH
    else:
        total_height = MB_LENGTHS[idx]
        layer_thicknesses, layer_height = create_inflation_distribution(FIRST_LAYER_THICKNESS, GROWTH_RATE, MAX_LAYERS, total_height=total_height, iso_length=ISO_LENGTH)

        # Assign MB_NXYZ based on the number of layers created
        for i,dim in enumerate(MB_DIMS):

            if i == idx:
                MB_NXYZ[i] = len(layer_thicknesses) // MB_DIMS[i] + 1
            else:
                MB_NXYZ[i] = int( MB_LENGTHS[i] / (ISO_LENGTH*MB_DIMS[i]) ) + 1

        print('Updated MB_NXYZ:', MB_NXYZ)


        # Give user the option to quit if too many layers are created
        Ncells = np.prod( [ (n-1)*d for n,d in zip(MB_NXYZ, MB_DIMS) ] )
        print(f"Total number of cells in the multiblock mesh will be: {Ncells}")
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

    output_path = './grid/'

    # Create output directory if it doesn't exist
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)
    else:
        # Clean the existing output directory
        os.system(f'rm -rf {output_path}/*')

    # Write the grid files
    if WRITE_VISUALIZATION:
        rp.writers.write_hdf5_grid(mb, output_path)     # For visualization
        rp.writers.write_hdf5_restart(mb, output_path)  # For visualization
    rp.writers.write_raptor_grid(mb, output_path)   # Raptor input grid (g.*)
    rp.writers.write_raptor_conn(mb, output_path)   # Raptor connectivity input (conn.inp)

    # Write the ernk.inp file
    cwd = os.getcwd()
    cmd = 'source ~/Documents/01_Academia/02_Masters/Research/00_COMMON/raptorpy/.venv/bin/activate;'
    cmd += 'python ~/Documents/01_Academia/02_Masters/Research/00_COMMON/raptorpy/utilities/create_ERNK.py'
    subprocess.Popen(cmd, cwd=output_path, shell=True).wait()
    


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plotting
    # fig,ax = plt.subplots(1,3, figsize=(18,6))
    # for i in range(3):
    #     slice_obj = [0,0,0]
    #     slice_obj[i] = slice(None)  # We only need one slice in the inflation direction
    #     slice_tup = tuple(slice_obj)
    #     for j in range(MB_DIMS[i]):
    #         ax[i].plot(mb[j].__dict__[DIRECTION][slice_tup], marker='o')
    #     ax[i].set_title(f'{DIRECTION.upper()} Direction Slice {i}')

    # fig, ax = plt.subplots(1,2, figsize=(12,6))
    # ax[0].plot(layer_thicknesses, marker='o')
    # ax[0].set_title('Layer Thicknesses')
    # ax[0].set_xlabel('Layer Number')
    # ax[0].set_ylabel('Thickness')
    # ax[1].plot(layer_height, marker='o', color='orange')
    # ax[1].set_title('Cumulative Layer Height')
    # ax[1].set_xlabel('Layer Number')
    # ax[1].set_ylabel('Cumulative Height')
    plt.show()


if __name__ == "__main__":
    main()
    


