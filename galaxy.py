 import re
import os
import time
import KETJU_IO
import warnings
import matplotlib
import plot_utilities
import analysis_utilities

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from scipy.special import gamma
from astropy.constants import G
from scipy.optimize import curve_fit

# Aesthetic parameters #
colors = ['black', '#e66101', '#5e3c99']
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)

# Path parameters #
data_path = '/scratch/project_2007836/dirodotou/analysis/data/'
plots_path = '/scratch/project_2007836/dirodotou/analysis/plots/'

# Constants #
astronomical_G = G.to(u.km ** 2 * u.kpc * u.Msun ** -1 * u.s ** -2).value


def bar_strength_profile(d, f, read=False, hfree=False, physical=False, verbose=False):
    """
    Plot the bar strength radial profile from Fourier modes of the face-on stellar surface density.
    :param d: directory path
    :param f: file name
    :param read: boolean: read new data
    :param hfree: boolean: convert to hfree units
    :param physical: boolean: convert to physical units
    :param verbose: Boolean: print duration of function(s)
    :return: None
    """
    print('Entering bar_strength_profile from galaxy.py')
    redshift = KETJU_IO.read_hdf5_header(d, f, 'Redshift')
    local_data_path = data_path + 'bsp/' + re.split('.hdf5', f)[0] + '/'
    local_plots_path = plots_path + 'bsp/' + re.split('.hdf5', f)[0] + '/'

    # Declare parameters #
    id = 13494108  # 6066668
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

        # Split up the galaxy in radial bins and calculate Fourier components #
        n_bins = 40  # Number of radial bins.
        r_m, beta_2, alpha_0, alpha_2 = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
        r = np.sqrt(rotated_pt4_x_coordinates ** 2 + rotated_pt4_y_coordinates ** 2)  # Radius of each particle.
        for i in range(0, n_bins):
            r_s = float(i) * 0.25
            r_b = float(i) * 0.25 + 0.25
            r_m[i] = float(i) * 0.25 + 0.125
            x_fit = rotated_pt4_x_coordinates[(r < r_b) & (r > r_s)]
            y_fit = rotated_pt4_y_coordinates[(r < r_b) & (r > r_s)]
            for k in range(0, len(x_fit)):
                th_i = np.arctan2(y_fit[k], x_fit[k])
                alpha_0[i] = alpha_0[i] + 1
                alpha_2[i] = alpha_2[i] + np.cos(2 * th_i)
                beta_2[i] = beta_2[i] + np.sin(2 * th_i)

        # Calculate bar strength A_2 #
        ratio = np.divide(np.sqrt(alpha_2[:] ** 2 + beta_2[:] ** 2), alpha_0[:])

        # Save data in numpy arrays #
        np.save(local_data_path + 'r_m_' + re.split('run_|.hdf5', f)[1], r_m)
        np.save(local_data_path + 'ratio_' + re.split('run_|.hdf5', f)[1], ratio)

        print('Spent %.4s ' % (time.time() - local_time) +
              's reading data in stellar_surface_density from galaxy.py' if verbose else '--------')

    local_time = time.time()  # Start the local time.

    # Check if a folder to save the plot(s) exists, if not then create one #
    if not os.path.exists(local_plots_path):
        os.makedirs(local_plots_path)

    # Load the data #
    r_m = np.load(local_data_path + 'r_m_' + re.split('run_|.hdf5', f)[1] + '.npy')
    ratio = np.load(local_data_path + 'ratio_' + re.split('run_|.hdf5', f)[1] + '.npy')

    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 10))
    plot_utilities.set_axis(axis, x_lim=[0, 10], y_lim=[-0.1, 1.1], x_label=r'$\mathrm{R/kpc}$',
                            y_label=r'$\mathrm{\sqrt{a_{2}^{2}+b_{2}^{2}}/a_{0}}$')
    figure.text(0.01, 0.90, r'$\mathrm{z=%.4s}$' % str(redshift), fontsize=20, transform=axis.transAxes)

    # Plot the bar strength radial profile from Fourier modes of the face-on stellar surface density #
    a_2 = max(ratio)
    plt.plot(r_m, ratio, color=colors[0], lw=3, label=r'$\mathrm{A_{2}=%.2f}$' % a_2)
    plt.plot([r_m[np.where(ratio == a_2)], r_m[np.where(ratio == a_2)]], [-0.0, a_2], color=colors[0], lw=3,
             linestyle='dashed', label=r'$\mathrm{r_{A_{2}}=%.2fkpc}$' % r_m[np.where(ratio == a_2)])

    # Create the legends, save and close the figure #
    plt.legend(loc='upper center', fontsize=20, frameon=False, scatterpoints=3)
    plt.savefig(local_plots_path + 'BSP-' + time.strftime('%d_%m_%y_%H%M') + '.png', bbox_inches='tight')
    plt.close()

    print('Spent %.4s ' % (time.time() - local_time) +
          's plotting data in stellar_surface_density from galaxy.py' if verbose else '--------')
    return None


def mass_surface_density_profile(d, f, read=False, hfree=False, physical=False, verbose=False):
    """
    Plot the mass surface density profile and fit a Sersic plus exponential profile.
    :param d: directory path
    :param f: file name
    :param read: boolean: read new data
    :param hfree: boolean: convert to hfree units
    :param physical: boolean: convert to physical units
    :param verbose: Boolean: print duration of function(s)
    :return: None
    """
    print('Entering mass_surface_density_profile from galaxy.py')
    redshift = KETJU_IO.read_hdf5_header(d, f, 'Redshift')
    local_data_path = data_path + 'msdp/' + re.split('.hdf5', f)[0] + '/'
    local_plots_path = plots_path + 'msdp/' + re.split('.hdf5', f)[0] + '/'

    # Declare parameters #
    id = 13494108  # 6066668
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

        # Calculate the mass surface density #
        cylindrical_distance = np.sqrt(rotated_pt4_coordinates[:, 0] ** 2 + rotated_pt4_coordinates[:, 1] ** 2)
        mass, edges = np.histogram(cylindrical_distance, bins=20 * radial_threshold, range=(0, radial_threshold),
                                   weights=pt4_masses)
        centers = 0.5 * (edges[1:] + edges[:-1])
        surface = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        sigma = mass / surface

        try:
            popt, pcov = curve_fit(analysis_utilities.total_profile, centers, sigma, sigma=0.1 * sigma,
                                   p0=[sigma[0], 2, sigma[0], 2, 4])  # p0 = [sigma_0d, r_d, sigma_0b, b, n]

            # Calculate galactic attributes #
            sigma_0d, r_d, sigma_0b, b, n = popt[0], popt[1], popt[2], popt[3], popt[4]
            r_eff = b * analysis_utilities.sersic_b_n(n) ** (1 / n)
            disc_mass = 2.0 * np.pi * sigma_0d * r_d ** 2
            spheroid_mass = np.pi * sigma_0b * r_eff ** 2 * gamma(2.0 / n + 1)

        except RuntimeError:
            print('Could not fit a Sersic+exponential profile')

        # Save data in numpy arrays #
        np.save(local_data_path + 'b_' + re.split('snap_|.hdf5', f)[1], b)
        np.save(local_data_path + 'n_' + re.split('snap_|.hdf5', f)[1], n)
        np.save(local_data_path + 'r_d_' + re.split('snap_|.hdf5', f)[1], r_d)
        np.save(local_data_path + 'r_eff_' + re.split('snap_|.hdf5', f)[1], r_eff)
        np.save(local_data_path + 'sigma_' + re.split('snap_|.hdf5', f)[1], sigma)
        np.save(local_data_path + 'centers_' + re.split('snap_|.hdf5', f)[1], centers)
        np.save(local_data_path + 'sigma_0d_' + re.split('snap_|.hdf5', f)[1], sigma_0d)
        np.save(local_data_path + 'sigma_0b_' + re.split('snap_|.hdf5', f)[1], sigma_0b)

        print('Spent %.4s ' % (time.time() - local_time) +
              's reading data in mass_surface_density_profile from galaxy.py' if verbose else '--------')

    local_time = time.time()  # Start the local time.

    # Check if a folder to save the plot(s) exists, if not then create one #
    if not os.path.exists(local_plots_path):
        os.makedirs(local_plots_path)

    # Load the data #
    b = np.load(local_data_path + 'b_' + re.split('snap_|.hdf5', f)[1] + '.npy')
    n = np.load(local_data_path + 'n_' + re.split('snap_|.hdf5', f)[1] + '.npy')
    r_d = np.load(local_data_path + 'r_d_' + re.split('snap_|.hdf5', f)[1] + '.npy')
    r_eff = np.load(local_data_path + 'r_eff_' + re.split('snap_|.hdf5', f)[1] + '.npy')
    sigma = np.load(local_data_path + 'sigma_' + re.split('snap_|.hdf5', f)[1] + '.npy')
    centers = np.load(local_data_path + 'centers_' + re.split('snap_|.hdf5', f)[1] + '.npy')
    sigma_0d = np.load(local_data_path + 'sigma_0d_' + re.split('snap_|.hdf5', f)[1] + '.npy')
    sigma_0b = np.load(local_data_path + 'sigma_0b_' + re.split('snap_|.hdf5', f)[1] + '.npy')

    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 10))
    plot_utilities.set_axis(axis, x_lim=[0, 10], y_lim=[1e7, 1e11], y_scale='log', x_label=r'$\mathrm{R/kpc}$',
                            y_label=r'$\mathrm{\Sigma_{\bigstar}/(M_{\odot}\;kpc^{-2})}$')
    figure.text(0.32, 0.85,
                r'$\mathrm{R_{d.}} = %.2f$ kpc' '\n' r'$\mathrm{R_{eff.}} = %.2f$ kpc' '\n' r'$\mathrm{n} = %.2f$' % (
                    r_d, r_eff, n), fontsize=20, transform=axis.transAxes)

    # Plot the mass surface density profile and fit a Sersic plus exponential profile #
    plt.scatter(centers, sigma, marker='o', s=50, color=colors[0], linewidth=0.0, label=r'$\mathrm{Total}$')
    plt.plot(centers, analysis_utilities.sersic_profile(centers, sigma_0b, b, n), color=colors[1], lw=3,
             label=r'$\mathrm{S\acute{e}rsic}$')
    plt.plot(centers, analysis_utilities.exponential_profile(centers, sigma_0d, r_d), color=colors[2], lw=3,
             label=r'$\mathrm{Exp.}$')
    plt.plot(centers, analysis_utilities.total_profile(centers, sigma_0d, r_d, sigma_0b, b, n), color=colors[0], lw=3,
             label=r'$\mathrm{Total}$')

    # Create the legends, save and close the figure #
    plt.legend(loc='upper right', fontsize=20, frameon=False, scatterpoints=3)
    plt.savefig(local_plots_path + 'BSP-' + time.strftime('%d_%m_%y_%H%M') + '.png', bbox_inches='tight')
    plt.close()

    print('Spent %.4s ' % (time.time() - local_time) +
          's plotting data in stellar_surface_density from galaxy.py' if verbose else '--------')
    return None


def circular_velocity_curve(d, f, read=False, hfree=False, physical=False, verbose=False):
    """
    Plot the circular velocity curve.
    :param d: directory path
    :param f: file name
    :param read: boolean: read new data
    :param hfree: boolean: convert to hfree units
    :param physical: boolean: convert to physical units
    :param verbose: Boolean: print duration of function(s)
    :return: None
    """
    print('Entering circular_velocity_curve from galaxy.py')
    redshift = KETJU_IO.read_hdf5_header(d, f, 'Redshift')
    local_data_path = data_path + 'cvc/' + re.split('.hdf5', f)[0] + '/'
    local_plots_path = plots_path + 'cvc/' + re.split('.hdf5', f)[0] + '/'

    # Declare parameters #
    id = 13494108  # 6066668
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
        # Select a specific galaxy based on a black hole id #
        pt5_mask, = np.where(pt5_ids == id)
        pt5_coordinates = pt5_coordinates[pt5_mask]

        # Stellar data #
        pt4_masses = KETJU_IO.read_hdf5_dataset(d, f, 'PartType4', 'Masses', hfree=hfree,
                                                physical=physical) * 1e10  # In Msun.
        pt4_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType4', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.
        # Select stellar particles within 'radial_threshold' kpc from a specific black hole #
        pt4_mask, = np.where((np.linalg.norm(pt4_coordinates - pt5_coordinates, axis=1) <= radial_threshold))
        # Mask and normalise stellar masses, coordinates, and velocities #
        pt4_masses = pt4_masses[pt4_mask]
        pt4_coordinates = pt4_coordinates[pt4_mask] - pt5_coordinates

        # Dark matter data #
        pt1_masses = KETJU_IO.read_hdf5_dataset(d, f, 'PartType1', 'Masses', hfree=hfree,
                                                physical=physical) * 1e10  # In Msun.
        pt1_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType1', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.
        # Select dark matter particles within 'radial_threshold' kpc from a specific black hole #
        pt1_mask, = np.where((np.linalg.norm(pt1_coordinates - pt5_coordinates, axis=1) <= radial_threshold))
        # Mask and normalise gas masses, coordinates, and velocities #
        pt1_masses = pt1_masses[pt1_mask]
        pt1_coordinates = pt1_coordinates[pt1_mask] - pt5_coordinates

        # Gas data #
        pt0_masses = KETJU_IO.read_hdf5_dataset(d, f, 'PartType0', 'Masses', hfree=hfree,
                                                physical=physical) * 1e10  # In Msun.
        pt0_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType0', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.
        # Select gas particles within 'radial_threshold' kpc from a specific black hole #
        pt0_mask, = np.where((np.linalg.norm(pt0_coordinates - pt5_coordinates, axis=1) <= radial_threshold))
        # Mask and normalise gas masses, coordinates, and velocities #
        pt0_masses = pt0_masses[pt0_mask]
        pt0_coordinates = pt0_coordinates[pt0_mask] - pt5_coordinates

        # Calculate the circular velocity of each component #
        pt4_data, pt1_data, pt0_data = {}, {}, {}
        pt4_data['Masses'], pt1_data['Masses'], pt0_data['Masses'] = pt4_masses, pt1_masses, pt0_masses
        pt4_data['Coordinates'], pt1_data['Coordinates'], pt0_data[
            'Coordinates'] = pt4_coordinates, pt1_coordinates, pt0_coordinates
        # Save data in numpy arrays #
        np.save(local_data_path + 'pt4_data_' + re.split('snap_|.hdf5', f)[1], pt4_data)
        np.save(local_data_path + 'pt1_data_' + re.split('snap_|.hdf5', f)[1], pt1_data)
        np.save(local_data_path + 'pt0_data_' + re.split('snap_|.hdf5', f)[1], pt0_data)

        print('Spent %.4s ' % (time.time() - local_time) +
              's reading data in circular_velocity_curve from galaxy.py' if verbose else '--------')

    local_time = time.time()  # Start the local time.

    # Check if a folder to save the plot(s) exists, if not then create one #
    if not os.path.exists(local_plots_path):
        os.makedirs(local_plots_path)

    # Load the data #
    pt4_data = np.load(local_data_path + 'pt4_data_' + re.split('snap_|.hdf5', f)[1] + '.npy', allow_pickle=True)
    pt1_data = np.load(local_data_path + 'pt1_data_' + re.split('snap_|.hdf5', f)[1] + '.npy', allow_pickle=True)
    pt0_data = np.load(local_data_path + 'pt0_data_' + re.split('snap_|.hdf5', f)[1] + '.npy', allow_pickle=True)

    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 10))
    plot_utilities.set_axis(axis, x_lim=[0, 30], y_lim=[1e1, 1e3], y_scale='log', x_label=r'$\mathrm{R/kpc}$',
                            y_label=r'$\mathrm{V_{c}/(km\;s^{-1})}$')

    # Plot the circular velocity curve for stellar, dark matter and gas particles #
    data = pt4_data.item(), pt1_data.item(), pt0_data.item()
    labels = r'$\mathrm{Stars}$', r'$\mathrm{Dark\;matter}$', r'$\mathrm{Gas}$'
    for data, label in zip(data, labels):
        prc_spherical_radius = np.sqrt(np.sum(data['Coordinates'] ** 2, axis=1))
        sort = np.argsort(prc_spherical_radius)
        sorted_prc_spherical_radius = prc_spherical_radius[sort]
        cumulative_mass = np.cumsum(data['Masses'][sort])
        circular_velocity = np.sqrt(np.divide(astronomical_G * cumulative_mass, sorted_prc_spherical_radius))
        plt.plot(sorted_prc_spherical_radius, circular_velocity, label=label)  # Plot the circular velocity curve.

    # Create the legends, save and close the figure #
    plt.legend(loc='upper right', fontsize=20, frameon=False, scatterpoints=3)
    plt.savefig(local_plots_path + 'CVC-' + time.strftime('%d_%m_%y_%H%M') + '.png', bbox_inches='tight')
    plt.close()

    print('Spent %.4s ' % (time.time() - local_time) +
          's plotting data in circular_velocity_curve from galaxy.py' if verbose else '--------')
    return None


def black_hole_vs_masses(d, f, read=False, hfree=False, physical=False, verbose=False):
    """
    Plot the mass of black holes versus the stellar, gas, and dark matter mass around them.
    :param d: directory path
    :param f: file name
    :param read: boolean: read new data
    :param hfree: boolean: convert to hfree units
    :param physical: boolean: convert to physical units
    :param verbose: Boolean: print duration of function(s)
    :return: None
    """
    print('Entering black_hole_vs_masses from galaxy.py')
    redshift = KETJU_IO.read_hdf5_header(d, f, 'Redshift')
    local_data_path = data_path + 'bhvm/' + re.split('.hdf5', f)[0] + '/'
    local_plots_path = plots_path + 'bhvm/' + re.split('.hdf5', f)[0] + '/'

    radial_threshold = 5  # In kpc.
    mass_threshold = 1e6  # In Msun.

    # Read new data #
    if read is True:
        local_time = time.time()  # Start the local time.
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(local_data_path):
            os.makedirs(local_data_path)

        # Gas data #
        pt0_masses = KETJU_IO.read_hdf5_dataset(d, f, 'PartType0', 'Masses', hfree=hfree,
                                                physical=physical) * 1e10  # In Msun.
        pt0_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType0', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.

        # Dark matter data #
        pt1_mass = KETJU_IO.read_hdf5_header(d, f, 'MassTable')[1] * 1e10 / \
                   KETJU_IO.read_hdf5_header(d, f, 'HubbleParam')  # In Msun.
        pt1_masses = np.ones(KETJU_IO.read_hdf5_header(d, f, 'NumPart_ThisFile')[1]) * pt1_mass
        pt1_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType1', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.

        # Low resolution data #
        pt2_masses = KETJU_IO.read_hdf5_dataset(d, f, 'PartType2', 'Masses', hfree=hfree,
                                                physical=physical) * 1e10  # In Msun.
        pt2_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType2', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.

        # Stellar data #
        pt4_masses = KETJU_IO.read_hdf5_dataset(d, f, 'PartType4', 'Masses', hfree=hfree,
                                                physical=physical) * 1e10  # In Msun.
        pt4_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType4', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.

        # Black hole data #
        pt5_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType5', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.
        pt5_masses = KETJU_IO.read_hdf5_dataset(d, f, 'PartType5', 'BH_Mass', hfree=hfree,
                                                physical=physical) * 1e10  # In Msun.
        # Select black hole particles with masses less than 'mass_threshold' Msun #
        pt5_mask, = np.where(pt5_masses < mass_threshold)
        pt5_masses = pt5_masses[pt5_mask]
        pt5_coordinates = pt5_coordinates[pt5_mask]

        # Select stellar, gas, and dark matter particles within 'radial_threshold' kpc from a specific black hole #
        pt0_total_mass, pt1_total_mass, pt2_total_mass, pt4_total_mass = np.zeros(len(pt5_masses), dtype=object), \
                                                                         np.zeros(len(pt5_masses), dtype=object), \
                                                                         np.zeros(len(pt5_masses), dtype=object), \
                                                                         np.zeros(len(pt5_masses), dtype=object)
        for i in range(len(pt5_masses)):
            # Select stellar particles within 'radial_threshold' kpc from a specific black hole #
            pt0_mask, = np.where((np.linalg.norm(pt0_coordinates - pt5_coordinates[i], axis=1) <= radial_threshold))
            pt1_mask, = np.where((np.linalg.norm(pt1_coordinates - pt5_coordinates[i], axis=1) <= radial_threshold))
            pt2_mask, = np.where((np.linalg.norm(pt2_coordinates - pt5_coordinates[i], axis=1) <= radial_threshold))
            pt4_mask, = np.where((np.linalg.norm(pt4_coordinates - pt5_coordinates[i], axis=1) <= radial_threshold))

            # Mask and calculate the total stellar, gas and dark matter masses #
            pt0_total_mass[i] = np.sum(pt0_masses[pt0_mask])
            pt1_total_mass[i] = np.sum(pt1_masses[pt1_mask])
            pt2_total_mass[i] = np.sum(pt2_masses[pt2_mask])
            pt4_total_mass[i] = np.sum(pt4_masses[pt4_mask])

        # Save data in numpy arrays #
        np.save(local_data_path + 'pt0_total_mass_' + re.split('snap_|.hdf5', f)[1], pt0_total_mass)
        np.save(local_data_path + 'pt1_total_mass_' + re.split('snap_|.hdf5', f)[1], pt1_total_mass)
        np.save(local_data_path + 'pt2_total_mass_' + re.split('snap_|.hdf5', f)[1], pt2_total_mass)
        np.save(local_data_path + 'pt4_total_mass_' + re.split('snap_|.hdf5', f)[1], pt4_total_mass)
        np.save(local_data_path + 'pt5_masses_' + re.split('snap_|.hdf5', f)[1], pt5_masses)

        print('Spent %.4s ' % (time.time() - local_time) +
              's reading data in black_hole_vs_masses from galaxy.py' if verbose else '--------')

    local_time = time.time()  # Start the local time.

    # Check if a folder to save the plot(s) exists, if not then create one #
    if not os.path.exists(local_plots_path):
        os.makedirs(local_plots_path)

    # Load the data #
    pt0_total_mass = np.load(local_data_path + 'pt0_total_mass_' + re.split('snap_|.hdf5', f)[1] + '.npy',
                             allow_pickle=True)
    pt1_total_mass = np.load(local_data_path + 'pt1_total_mass_' + re.split('snap_|.hdf5', f)[1] + '.npy',
                             allow_pickle=True)
    pt2_total_mass = np.load(local_data_path + 'pt2_total_mass_' + re.split('snap_|.hdf5', f)[1] + '.npy',
                             allow_pickle=True)
    pt4_total_mass = np.load(local_data_path + 'pt4_total_mass_' + re.split('snap_|.hdf5', f)[1] + '.npy',
                             allow_pickle=True)
    pt5_masses = np.load(local_data_path + 'pt5_masses_' + re.split('snap_|.hdf5', f)[1] + '.npy', allow_pickle=True)

    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 10))
    plot_utilities.set_axis(axis, x_lim=[3e4, 3e6], y_lim=[1e3, 1e10], x_scale='log', y_scale='log',
                            x_label=r'$\mathrm{M_\bullet/M_\odot}$', y_label=r'$\mathrm{M(r<5kpc)/M_\odot}$',
                            which='major')

    # Plot the mass of black holes versus the stellar, gas, and dark matter mass around them #
    pt0_total_mass[pt0_total_mass == 0] = 1e4
    pt1_total_mass[pt1_total_mass == 0] = 1e4
    pt2_total_mass[pt2_total_mass == 0] = 1e4
    pt4_total_mass[pt4_total_mass == 0] = 1e4

    plt.scatter(pt5_masses, pt0_total_mass, marker='o', s=30, color='tab:red', linewidth=0.0, label=r'$\mathrm{Gas}$')
    plt.scatter(pt5_masses, pt1_total_mass, marker='o', s=30, color='tab:blue', linewidth=0.0, label=r'$\mathrm{DM}$')
    plt.scatter(pt5_masses, pt2_total_mass, marker='o', s=30, color='tab:brown', linewidth=0.0, label=r'$\mathrm{LR}$')
    plt.scatter(pt5_masses, pt4_total_mass, marker='o', s=30, color='tab:green', linewidth=0.0,
                label=r'$\mathrm{Stellar}$')
    figure.text(0.01, 0.90, r'$\mathrm{z=%.4s}$' % str(redshift), fontsize=20, transform=axis.transAxes)

    # Create the legends, save and close the figure #
    plt.legend(loc='upper right', fontsize=20, frameon=False, scatterpoints=1)
    plt.savefig(local_plots_path + 'BHVM-' + time.strftime('%d_%m_%y_%H%M') + '.png', bbox_inches='tight')
    plt.close()

    print('Spent %.4s ' % (time.time() - local_time) +
          's plotting data in black_hole_vs_masses from galaxy.py' if verbose else '--------')
    return None


def mass_vs_specific_angular_momentum(d, f, read=False, hfree=False, physical=False, verbose=False):
    """
    Plot the mass-spin relation.
    :param d: directory path
    :param f: file name
    :param read: boolean: read new data
    :param hfree: boolean: convert to hfree units
    :param physical: boolean: convert to physical units
    :param verbose: Boolean: print duration of function(s)
    :return: None
    """
    print('Entering mass_vs_specific_angular_momentum from galaxy.py')
    redshift = KETJU_IO.read_hdf5_header(d, f, 'Redshift')
    local_data_path = data_path + 'mvsam/' + re.split('.hdf5', f)[0] + '/'
    local_plots_path = plots_path + 'mvsam/' + re.split('.hdf5', f)[0] + '/'

    # Declare parameters #
    id = 13494108  # 6066668
    radial_threshold = 30  # In kpc.

    # Read new data #
    if read is True:
        local_time = time.time()  # Start the local time.
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(local_data_path):
            os.makedirs(local_data_path)

        # Black hole data #
        pt5_ids = KETJU_IO.read_hdf5_dataset(d, f, 'PartType5', 'ParticleIDs', hfree=hfree, physical=physical)
        pt5_masses = KETJU_IO.read_hdf5_dataset(d, f, 'PartType5', 'BH_Mass', hfree=hfree,
                                                physical=physical) * 1e10  # In Msun.
        pt5_velocities = KETJU_IO.read_hdf5_dataset(d, f, 'PartType5', 'Velocities', hfree=True,
                                                    physical=True)  # In km s^-1.
        pt5_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType5', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.
        # Select a specific galaxy based on a black hole id #
        pt5_mask, = np.where(pt5_ids == id)
        pt5_masses = pt5_masses[pt5_mask]
        pt5_velocities = pt5_velocities[pt5_mask]
        pt5_coordinates = pt5_coordinates[pt5_mask]

        # Gas data #
        pt0_masses = KETJU_IO.read_hdf5_dataset(d, f, 'PartType0', 'Masses', hfree=hfree,
                                                physical=physical) * 1e10  # In Msun.
        pt0_velocities = KETJU_IO.read_hdf5_dataset(d, f, 'PartType0', 'Velocities', hfree=True,
                                                    physical=True)  # In km s^-1.
        pt0_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType0', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.
        # Select gas particles within 'radial_threshold' kpc from a specific black hole #
        pt0_mask, = np.where((np.linalg.norm(pt0_coordinates - pt5_coordinates, axis=1) <= radial_threshold))
        # Mask and normalise gas masses, coordinates, and velocities #
        pt0_masses = pt0_masses[pt0_mask]
        pt0_velocities = pt0_velocities[pt0_mask]
        pt0_coordinates = pt0_coordinates[pt0_mask]

        # Dark matter data #
        pt1_mass = KETJU_IO.read_hdf5_header(d, f, 'MassTable')[1] * 1e10 / \
                   KETJU_IO.read_hdf5_header(d, f, 'HubbleParam')  # In Msun.
        pt1_masses = np.ones(KETJU_IO.read_hdf5_header(d, f, 'NumPart_ThisFile')[1]) * pt1_mass
        pt1_velocities = KETJU_IO.read_hdf5_dataset(d, f, 'PartType1', 'Velocities', hfree=True,
                                                    physical=True)  # In km s^-1.
        pt1_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType1', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.
        # Select dark matter particles within 'radial_threshold' kpc from a specific black hole #
        pt1_mask, = np.where((np.linalg.norm(pt1_coordinates - pt5_coordinates, axis=1) <= radial_threshold))
        # Mask and normalise gas masses, coordinates, and velocities #
        pt1_masses = pt1_masses[pt1_mask]
        pt1_velocities = pt1_velocities[pt1_mask]
        pt1_coordinates = pt1_coordinates[pt1_mask]

        # Stellar data #
        pt4_masses = KETJU_IO.read_hdf5_dataset(d, f, 'PartType4', 'Masses', hfree=hfree,
                                                physical=physical) * 1e10  # In Msun.
        pt4_velocities = KETJU_IO.read_hdf5_dataset(d, f, 'PartType4', 'Velocities', hfree=True,
                                                    physical=True)  # In km s^-1.
        pt4_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType4', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.
        # Select stellar particles within 'radial_threshold' kpc from a specific black hole #
        pt4_mask, = np.where((np.linalg.norm(pt4_coordinates - pt5_coordinates, axis=1) <= radial_threshold))
        # Mask and normalise stellar masses, coordinates, and velocities #
        pt4_masses = pt4_masses[pt4_mask]
        pt4_velocities = pt4_velocities[pt4_mask]
        pt4_coordinates = pt4_coordinates[pt4_mask]

        # Calculate the centre of mass coordinates and velocity #
        pt_masses = np.hstack([pt0_masses, pt1_masses, pt4_masses, pt5_masses])
        pt_velocities = np.vstack([pt0_velocities, pt1_velocities, pt4_velocities, pt5_velocities])
        pt_coordinates = np.vstack([pt0_coordinates, pt1_coordinates, pt4_coordinates, pt5_coordinates])

        com_coordinates = np.divide(np.sum(pt_masses[:, np.newaxis] * pt_coordinates, axis=0),
                                    np.sum(pt_masses, axis=0))  # In kpc.

        com_velocity = np.divide(np.sum(pt_masses[:, np.newaxis] * pt_velocities, axis=0),
                                 np.sum(pt_masses, axis=0))  # In km s^-1.

        # No need to add the Hubble flow H * (pt4_coordinates - com_coordinates) to velocities when computing AM #
        pt4_specific_angular_momenta = np.cross(pt4_masses[:, np.newaxis] * (pt4_coordinates - com_coordinates),
                                                pt4_velocities - com_velocity)

        glx_specific_angular_momenta_magnitude = np.linalg.norm(np.sum(pt4_specific_angular_momenta, axis=0)) / np.sum(
            pt4_masses, axis=0)

        print(np.log10(glx_specific_angular_momenta_magnitude))

    #     # Save data in numpy arrays #
    #     np.save(local_data_path + 'com_coordinates_' + re.split('snap_|.hdf5', f)[1], com_coordinates)
    #
    #     print('Spent %.4s ' % (time.time() - local_time) +
    #           's reading data in mass_vs_specific_angular_momentum from galaxy.py' if verbose else '--------')
    #
    # local_time = time.time()  # Start the local time.
    #
    # # Check if a folder to save the plot(s) exists, if not then create one #
    # if not os.path.exists(local_plots_path):
    #     os.makedirs(local_plots_path)
    #
    # # Load the data #
    # pt4_data = np.load(local_data_path + 'pt4_data_' + re.split('snap_|.hdf5', f)[1] + '.npy', allow_pickle=True)
    # pt1_data = np.load(local_data_path + 'pt1_data_' + re.split('snap_|.hdf5', f)[1] + '.npy', allow_pickle=True)
    # pt0_data = np.load(local_data_path + 'pt0_data_' + re.split('snap_|.hdf5', f)[1] + '.npy', allow_pickle=True)
    #
    # # Generate the figure and set its parameters #
    # figure, axis = plt.subplots(1, figsize=(10, 10))
    # plot_utilities.set_axis(axis, x_lim=[0, 30], y_lim=[1e1, 1e3], y_scale='log', x_label=r'$\mathrm{R/kpc}$',
    #                         y_label=r'$\mathrm{V_{c}/(km\;s^{-1})}$')
    #
    # # Plot the circular velocity curve for stellar, dark matter and gas particles #
    # data = pt4_data.item(), pt1_data.item(), pt0_data.item()
    # labels = r'$\mathrm{Stars}$', r'$\mathrm{Dark\;matter}$', r'$\mathrm{Gas}$'
    # for data, label in zip(data, labels):
    #     prc_spherical_radius = np.sqrt(np.sum(data['Coordinates'] ** 2, axis=1))
    #     sort = np.argsort(prc_spherical_radius)
    #     sorted_prc_spherical_radius = prc_spherical_radius[sort]
    #     cumulative_mass = np.cumsum(data['Masses'][sort])
    #     circular_velocity = np.sqrt(np.divide(astronomical_G * cumulative_mass, sorted_prc_spherical_radius))
    #     plt.plot(sorted_prc_spherical_radius, circular_velocity, label=label)  # Plot the circular velocity curve.
    #
    # # Create the legends, save and close the figure #
    # plt.legend(loc='upper right', fontsize=20, frameon=False, scatterpoints=3)
    # plt.savefig(local_plots_path + 'CVC-' + time.strftime('%d_%m_%y_%H%M') + '.png', bbox_inches='tight')
    # plt.close()
    #
    # print('Spent %.4s ' % (time.time() - local_time) +
    #       's plotting data in mass_vs_specific_angular_momentum from galaxy.py' if verbose else '--------')
    return None


def flow_rates(d, f, read=False, hfree=False, physical=False, verbose=False):
    """
    Plot the inflow/outflow rates.
    :param d: directory path
    :param f: file name
    :param read: boolean: read new data
    :param hfree: boolean: convert to hfree units
    :param physical: boolean: convert to physical units
    :param verbose: Boolean: print duration of function(s)
    :return: None
    """
    print('Entering flow_rates from galaxy.py')
    redshift = KETJU_IO.read_hdf5_header(d, f, 'Redshift')
    local_data_path = data_path + 'mvsam/' + re.split('.hdf5', f)[0] + '/'
    local_plots_path = plots_path + 'mvsam/' + re.split('.hdf5', f)[0] + '/'

    # Declare parameters #
    id = 13494108  # 6066668
    time_interval = 250  # In Myr.
    radial_threshold = 30  # In kpc.
    radial_cut = np.divide(5e3, KETJU_IO.read_hdf5_header(d, f, 'HubbleParam') * (
            1 + KETJU_IO.read_hdf5_header(d, f, 'Redshift')))  # In kpc.

    # Read new data #
    if read is True:
        local_time = time.time()  # Start the local time.
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(local_data_path):
            os.makedirs(local_data_path)

        # Gas data #
        pt0_masses = KETJU_IO.read_hdf5_dataset(d, f, 'PartType0', 'Masses', hfree=hfree,
                                                physical=physical) * 1e10  # In Msun.
        pt0_velocities = KETJU_IO.read_hdf5_dataset(d, f, 'PartType0', 'Velocities', hfree=hfree,
                                                    physical=physical)  # In km s^-1.
        pt0_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType0', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.
        # Dark matter data #
        pt1_mass = KETJU_IO.read_hdf5_header(d, f, 'MassTable')[1] * 1e10 / \
                   KETJU_IO.read_hdf5_header(d, f, 'HubbleParam')  # In Msun.
        pt1_masses = np.ones(KETJU_IO.read_hdf5_header(d, f, 'NumPart_ThisFile')[1]) * pt1_mass
        pt1_velocities = KETJU_IO.read_hdf5_dataset(d, f, 'PartType1', 'Velocities', hfree=hfree,
                                                    physical=physical)  # In km s^-1.
        pt1_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType1', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.
        # Stellar data #
        pt4_masses = KETJU_IO.read_hdf5_dataset(d, f, 'PartType4', 'Masses', hfree=hfree,
                                                physical=physical) * 1e10  # In Msun.
        pt4_velocities = KETJU_IO.read_hdf5_dataset(d, f, 'PartType4', 'Velocities', hfree=hfree,
                                                    physical=physical)  # In km s^-1.
        pt4_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType4', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.

        # Black hole data #
        pt5_masses = KETJU_IO.read_hdf5_dataset(d, f, 'PartType5', 'BH_Mass', hfree=hfree,
                                                physical=physical) * 1e10  # In Msun.
        pt5_velocities = KETJU_IO.read_hdf5_dataset(d, f, 'PartType5', 'Velocities', hfree=hfree,
                                                    physical=physical)  # In km s^-1.
        pt5_coordinates = KETJU_IO.read_hdf5_dataset(d, f, 'PartType5', 'Coordinates', hfree=hfree,
                                                     physical=physical)  # In kpc.

        # Calculate the centre of mass coordinates and velocity #
        pt_masses = np.hstack([pt0_masses, pt1_masses, pt4_masses, pt5_masses])
        pt_velocities = np.vstack([pt0_velocities, pt1_velocities, pt4_velocities, pt5_velocities])
        pt_coordinates = np.vstack([pt0_coordinates, pt1_coordinates, pt4_coordinates, pt5_coordinates])

        com_coordinates = np.divide(np.sum(pt_masses[:, np.newaxis] * pt_coordinates, axis=0),
                                    np.sum(pt_masses, axis=0))  # In kpc.

        com_velocity = np.divide(np.sum(pt_masses[:, np.newaxis] * pt_velocities, axis=0),
                                 np.sum(pt_masses, axis=0))  # In km s^-1.

        # Calculate the gas outflow rate from a spherical surface with radius 'radial_cut' at time 'time_interval' #
        spherical_radius = np.sqrt(np.sum((pt0_coordinates - com_coordinates) ** 2, axis=1))
        radial_velocity = np.divide(
            np.sum((pt0_velocities - com_velocity) * (pt0_coordinates - com_coordinates), axis=1), spherical_radius)
        outflow_mask, = np.where((spherical_radius < radial_cut) & (
                spherical_radius + radial_velocity * u.km.to(u.kpc) * time_interval * u.Myr.to(u.s) > radial_cut))
        inflow_mask, = np.where((spherical_radius > radial_cut) & (
                spherical_radius + radial_velocity * u.km.to(u.kpc) * time_interval * u.Myr.to(u.s) < radial_cut))
        mass_outflow = np.divide(np.sum(pt0_masses[outflow_mask]), time_interval * 1e6)  # In Msun/yr.
        mass_inflow = np.divide(np.sum(pt0_masses[inflow_mask]), time_interval * 1e6)  # In Msun/yr.

    #     # Save data in numpy arrays #
    #     np.save(local_data_path + 'com_coordinates_' + re.split('snap_|.hdf5', f)[1], com_coordinates)
    #
    #     print('Spent %.4s ' % (time.time() - local_time) +
    #           's reading data in flow_rates from galaxy.py' if verbose else '--------')
    #
    # local_time = time.time()  # Start the local time.
    #
    # # Check if a folder to save the plot(s) exists, if not then create one #
    # if not os.path.exists(local_plots_path):
    #     os.makedirs(local_plots_path)
    #
    # # Load the data #
    # pt4_data = np.load(local_data_path + 'pt4_data_' + re.split('snap_|.hdf5', f)[1] + '.npy', allow_pickle=True)
    # pt1_data = np.load(local_data_path + 'pt1_data_' + re.split('snap_|.hdf5', f)[1] + '.npy', allow_pickle=True)
    # pt0_data = np.load(local_data_path + 'pt0_data_' + re.split('snap_|.hdf5', f)[1] + '.npy', allow_pickle=True)
    #
    # # Generate the figure and set its parameters #
    # figure, axis = plt.subplots(1, figsize=(10, 10))
    # plot_utilities.set_axis(axis, x_lim=[0, 30], y_lim=[1e1, 1e3], y_scale='log', x_label=r'$\mathrm{R/kpc}$',
    #                         y_label=r'$\mathrm{V_{c}/(km\;s^{-1})}$')
    #
    # # Plot the circular velocity curve for stellar, dark matter and gas particles #
    # data = pt4_data.item(), pt1_data.item(), pt0_data.item()
    # labels = r'$\mathrm{Stars}$', r'$\mathrm{Dark\;matter}$', r'$\mathrm{Gas}$'
    # for data, label in zip(data, labels):
    #     prc_spherical_radius = np.sqrt(np.sum(data['Coordinates'] ** 2, axis=1))
    #     sort = np.argsort(prc_spherical_radius)
    #     sorted_prc_spherical_radius = prc_spherical_radius[sort]
    #     cumulative_mass = np.cumsum(data['Masses'][sort])
    #     circular_velocity = np.sqrt(np.divide(astronomical_G * cumulative_mass, sorted_prc_spherical_radius))
    #     plt.plot(sorted_prc_spherical_radius, circular_velocity, label=label)  # Plot the circular velocity curve.
    #
    # # Create the legends, save and close the figure #
    # plt.legend(loc='upper right', fontsize=20, frameon=False, scatterpoints=3)
    # plt.savefig(local_plots_path + 'CVC-' + time.strftime('%d_%m_%y_%H%M') + '.png', bbox_inches='tight')
    # plt.close()
    #
    # print('Spent %.4s ' % (time.time() - local_time) +
    #       's plotting data in flow_rates from galaxy.py' if verbose else '--------')
    return None
