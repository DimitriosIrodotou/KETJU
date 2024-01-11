import re
import os
import time
import KETJU_IO
import warnings
import matplotlib
import plot_utilities
import analysis_utilities

import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt

# Aesthetic parameters #
colors = ['black', '#e66101', '#5e3c99']
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)

# Path parameters #
data_path = '/scratch/project_2007836/dirodotou/analysis/data/'
plots_path = '/scratch/project_2007836/dirodotou/analysis/plots/'


def stellar_surface_density(d, f, read=False, hfree=False, physical=False, verbose=False):
    """
    Plot the face-on and edge-on stellar surface density.
    :param d: directory path
    :param f: file name
    :param read: boolean: read new data
    :param hfree: boolean: convert to hfree units
    :param physical: boolean: convert to physical units
    :param verbose: Boolean: print duration of function(s)
    :return: None
    """
    print('Entering stellar_surface_density from projections.py')
    redshift = KETJU_IO.read_hdf5_header(d, f, 'Redshift')
    local_data_path = data_path + 'ssd/' + re.split('.hdf5', f)[0] + '/'
    local_plots_path = plots_path + 'ssd/' + re.split('.hdf5', f)[0] + '/'

    # Declare parameters #
    id = 44153861  # 13494108  # 6066668
    radial_threshold = 30  # In kpc.

    # Read new data #
    if read is True:
        local_time = time.time()  # Start the local time.
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(local_data_path):
            os.makedirs(local_data_path)

        # Black hole data #
        pt5_ids = KETJU_IO.read_hdf5_dataset(d, f, 'PartType5', 'ParticleIDs', hfree=hfree, physical=physical)
        pt5_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType5', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.
        pt5_velocities = KETJU_IO.read_hdf5_dataset(d, f, 'PartType5', 'Velocities', hfree=hfree,
                                                    physical=physical)  # In km s^-1.
        # Select a specific galaxy based on a black hole id #
        pt5_mask, = np.where(pt5_ids == id)
        pt5_coordinates = pt5_coordinates[pt5_mask]
        pt5_velocities = pt5_velocities[pt5_mask]

        # Stellar data #
        pt4_masses = KETJU_IO.read_hdf5_dataset(d, f, 'PartType4', 'Masses', hfree=hfree,
                                                physical=physical) * 1e10  # In Msun.
        pt4_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType4', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.
        pt4_velocities = KETJU_IO.read_hdf5_dataset(d, f, 'PartType4', 'Velocities', hfree=hfree,
                                                    physical=physical)  # In km s^-1.
        # Select stellar particles within 'radial_threshold' kpc from a specific black hole #
        pt4_mask, = np.where((np.linalg.norm(pt4_coordinates - pt5_coordinates, axis=1) <= radial_threshold))
        # Mask and normalise stellar masses, coordinates, and velocities #
        pt4_masses = pt4_masses[pt4_mask]
        pt4_velocities = pt4_velocities[pt4_mask] - pt5_velocities
        pt4_coordinates = pt4_coordinates[pt4_mask] - pt5_coordinates

        # Rotate coordinates and velocities of stellar particles so the galactic angular momentum points along the
        # z axis #
        rotated_pt4_coordinates, rotated_pt4_velocities, prc_unit_vector, glx_unit_vector = \
            analysis_utilities.KETJU.rotate_Jz(pt4_masses, pt4_coordinates, pt4_velocities)

        # Rotate coordinates of stellar particles so the bar is along the x axis #
        rotated_pt4_x_coordinates, rotated_pt4_y_coordinates, rotated_pt4_z_coordinates = \
            analysis_utilities.KETJU.rotate_bar(rotated_pt4_coordinates[:, 0], rotated_pt4_coordinates[:, 1],
                                                rotated_pt4_coordinates[:, 2])

        face_on_count, x_edges, y_edges = np.histogram2d(rotated_pt4_x_coordinates, rotated_pt4_y_coordinates,
                                                         weights=pt4_masses, bins=300,
                                                         range=[[-radial_threshold / 2, radial_threshold / 2],
                                                                [-radial_threshold / 2, radial_threshold / 2]])

        edge_on_count, x_edges, y_edges = np.histogram2d(rotated_pt4_x_coordinates, rotated_pt4_z_coordinates,
                                                         weights=pt4_masses, bins=300,
                                                         range=[[-radial_threshold / 2, radial_threshold / 2],
                                                                [-radial_threshold / 2, radial_threshold / 2]])

        # Save data in numpy arrays #
        np.save(local_data_path + 'face_on_count_' + re.split('run_|.hdf5', f)[1], face_on_count)
        np.save(local_data_path + 'edge_on_count_' + re.split('run_|.hdf5', f)[1], edge_on_count)
        print('Spent %.4s ' % (time.time() - local_time) +
              's reading data in stellar_surface_density from projections.py' if verbose else '--------')

    local_time = time.time()  # Start the local time.

    # Check if a folder to save the plot(s) exists, if not then create one #
    if not os.path.exists(local_plots_path):
        os.makedirs(local_plots_path)

    # Load the data #
    face_on_count = np.load(local_data_path + 'face_on_count_' + re.split('run_|.hdf5', f)[1] + '.npy')
    edge_on_count = np.load(local_data_path + 'edge_on_count_' + re.split('run_|.hdf5', f)[1] + '.npy')

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(15, 10))
    gs = matplotlib.gridspec.GridSpec(2, 2, wspace=0)
    axis00, axis01 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])
    plot_utilities.set_axis(axis00, x_label=r'$\mathrm{x/kpc}$', y_label=r'$\mathrm{y/kpc}$', which='major')
    plot_utilities.set_axis(axis01, x_label=r'$\mathrm{x/kpc}$', y_label=r'$\mathrm{z/kpc}$', which='major')
    cmap = cmr.copper
    axis00.set_facecolor(cmap(0))
    axis01.set_facecolor(cmap(0))
    figure.text(0.01, 0.90, r'$\mathrm{z=%.4s}$' % str(redshift), fontsize=20, color='w', transform=axis00.transAxes)

    # Plot the face-on and edge-on stellar surface density #
    axis00.imshow(np.log10(face_on_count.T),
                  extent=[-radial_threshold / 2, radial_threshold / 2, -radial_threshold / 2, radial_threshold / 2],
                  origin='lower', cmap=cmap, rasterized=True, aspect='equal')
    axis01.imshow(np.log10(edge_on_count.T),
                  extent=[-radial_threshold / 2, radial_threshold / 2, -radial_threshold / 2, radial_threshold / 2],
                  origin='lower', cmap=cmap, rasterized=True, aspect='equal')

    # Save and close the figure #
    plt.savefig(local_plots_path + 'SSD-' + time.strftime('%d_%m_%y_%H%M') + '.png', bbox_inches='tight')
    plt.close()

    print('Spent %.4s ' % (time.time() - local_time) +
          's plotting data in stellar_surface_density from projections.py' if verbose else '--------')
    return None


def gas_surface_density(d, f, read=False, hfree=False, physical=False, verbose=False):
    """
    Plot the face-on and edge-on gas surface density.
    :param d: directory path
    :param f: file name
    :param read: boolean: read new data
    :param hfree: boolean: convert to hfree units
    :param physical: boolean: convert to physical units
    :param verbose: Boolean: print duration of function(s)
    :return: None
    """
    print('Entering gas_surface_density from projections.py')
    redshift = KETJU_IO.read_hdf5_header(d, f, 'Redshift')
    local_data_path = data_path + 'gsd/' + re.split('.hdf5', f)[0] + '/'
    local_plots_path = plots_path + 'gsd/' + re.split('.hdf5', f)[0] + '/'

    # Declare parameters #
    id = 6066668  # 13494108
    radial_threshold = 30  # In kpc.

    # Read new data #
    if read is True:
        local_time = time.time()  # Start the local time.
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(local_data_path):
            os.makedirs(local_data_path)

        # Black hole data #
        pt5_ids = KETJU_IO.read_hdf5_dataset(d, f, 'PartType5', 'ParticleIDs', hfree=hfree, physical=physical)
        pt5_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType5', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.
        pt5_velocities = KETJU_IO.read_hdf5_dataset(d, f, 'PartType5', 'Velocities', hfree=hfree,
                                                    physical=physical)  # In km s^-1.
        # Select a specific galaxy based on a black hole id #
        pt5_mask, = np.where(pt5_ids == id)
        pt5_coordinates = pt5_coordinates[pt5_mask]
        pt5_velocities = pt5_velocities[pt5_mask]

        # Gas data #
        pt0_masses = KETJU_IO.read_hdf5_dataset(d, f, 'PartType0', 'Masses', hfree=hfree,
                                                physical=physical) * 1e10  # In Msun.
        pt0_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType0', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.
        pt0_velocities = KETJU_IO.read_hdf5_dataset(d, f, 'PartType0', 'Velocities', hfree=hfree,
                                                    physical=physical)  # In km s^-1.
        # Select gas particles within 'radial_threshold' kpc from a specific black hole #
        pt0_mask, = np.where((np.linalg.norm(pt0_coordinates - pt5_coordinates, axis=1) <= radial_threshold))
        # Mask and normalise gas masses, coordinates, and velocities #
        pt0_masses = pt0_masses[pt0_mask]
        pt0_velocities = pt0_velocities[pt0_mask] - pt5_velocities
        pt0_coordinates = pt0_coordinates[pt0_mask] - pt5_coordinates

        # Rotate coordinates and velocities of gas particles so the galactic angular momentum points along the z axis #
        rotated_pt0_coordinates, rotated_pt0_velocities, prc_unit_vector, glx_unit_vector = \
            analysis_utilities.KETJU.rotate_Jz(pt0_masses, pt0_coordinates, pt0_velocities)

        face_on_count, x_edges, y_edges = np.histogram2d(rotated_pt0_coordinates[:, 0], rotated_pt0_coordinates[:, 1],
                                                         weights=pt0_masses, bins=300,
                                                         range=[[-radial_threshold / 2, radial_threshold / 2],
                                                                [-radial_threshold / 2, radial_threshold / 2]])

        edge_on_count, x_edges, y_edges = np.histogram2d(rotated_pt0_coordinates[:, 0], rotated_pt0_coordinates[:, 2],
                                                         weights=pt0_masses, bins=300,
                                                         range=[[-radial_threshold / 2, radial_threshold / 2],
                                                                [-radial_threshold / 2, radial_threshold / 2]])

        # Save data in numpy arrays #
        np.save(local_data_path + 'face_on_count_' + re.split('run_|.hdf5', f)[1], face_on_count)
        np.save(local_data_path + 'edge_on_count_' + re.split('run_|.hdf5', f)[1], edge_on_count)

        print('Spent %.4s ' % (time.time() - local_time) +
              's reading data in gas_surface_density from projections.py' if verbose else '--------')

    local_time = time.time()  # Start the local time.
    # Check if a folder to save the plot(s) exists, if not then create one #
    if not os.path.exists(local_plots_path):
        os.makedirs(local_plots_path)

    # Load the data #
    face_on_count = np.load(local_data_path + 'face_on_count_' + re.split('run_|.hdf5', f)[1] + '.npy')
    edge_on_count = np.load(local_data_path + 'edge_on_count_' + re.split('run_|.hdf5', f)[1] + '.npy')

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(15, 10))
    gs = matplotlib.gridspec.GridSpec(2, 2, wspace=0)
    axis00, axis01 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])
    plot_utilities.set_axis(axis00, x_label=r'$\mathrm{x/kpc}$', y_label=r'$\mathrm{y/kpc}$', which='major')
    plot_utilities.set_axis(axis01, x_label=r'$\mathrm{x/kpc}$', y_label=r'$\mathrm{z/kpc}$', which='major')
    cmap = cmr.copper
    axis00.set_facecolor(cmap(0))
    axis01.set_facecolor(cmap(0))
    figure.text(0.01, 0.90, r'$\mathrm{z=%.4s}$' % str(redshift), fontsize=20, color='w', transform=axis00.transAxes)

    # Plot the face-on and edge-on gas surface density #
    axis00.imshow(np.log10(face_on_count.T),
                  extent=[-radial_threshold / 2, radial_threshold / 2, -radial_threshold / 2, radial_threshold / 2],
                  origin='lower', cmap=cmap, rasterized=True, aspect='equal')
    axis01.imshow(np.log10(edge_on_count.T),
                  extent=[-radial_threshold / 2, radial_threshold / 2, -radial_threshold / 2, radial_threshold / 2],
                  origin='lower', cmap=cmap, rasterized=True, aspect='equal')

    # Save and close the figure #
    plt.savefig(local_plots_path + 'GSD-' + time.strftime('%d_%m_%y_%H%M') + '.png', bbox_inches='tight')
    plt.close()

    print('Spent %.4s ' % (time.time() - local_time) +
          's plotting data in gas_surface_density from projections.py' if verbose else '--------')
    return None
