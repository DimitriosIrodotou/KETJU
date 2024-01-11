import time
import numpy as np
import healpy as hlp
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.style as style

from astropy.constants import G
from astropy_healpix import HEALPix
from morpho_kinematics import MorphoKinematic

# import satellite_utilities

res = 512
boxsize = 0.06

style.use("classic")
plt.rcParams.update({'font.family': 'serif'})


class KETJU:
    """
    Scripts to analyse KETJU data.
    """

    @staticmethod
    def rotate_bar(x, y, z, verbose=False):
        """
        Calculate bar strength and rotate bar to horizontal position.
        :param x: the x-position of the particles
        :param y: the y-position of the particles
        :param z: the z-position of the particles
        :param verbose: boolean: print duration of function
        :return: x_pos, y_pos, z_pos
        """
        print('Entering rotate_bar from analysis_utilities.py')
        local_time = time.time()  # Start the local time.

        # Declare arrays to store the data #
        n_bins = 40  # Number of radial bins.
        r_m = np.zeros(n_bins)
        beta_2 = np.zeros(n_bins)
        alpha_0 = np.zeros(n_bins)
        alpha_2 = np.zeros(n_bins)

        # Split disc in radial bins and calculate Fourier components #
        r = np.sqrt(x[:] ** 2 + y[:] ** 2)  # Radius of each particle.
        for i in range(0, n_bins):
            r_s = float(i) * 0.25
            r_b = float(i) * 0.25 + 0.25
            r_m[i] = float(i) * 0.25 + 0.125
            xfit = x[(r < r_b) & (r > r_s)]
            yfit = y[(r < r_b) & (r > r_s)]
            l = len(xfit)
            for k in range(0, l):
                th_i = np.arctan2(yfit[k], xfit[k])
                alpha_0[i] = alpha_0[i] + 1
                alpha_2[i] = alpha_2[i] + np.cos(2 * th_i)
                beta_2[i] = beta_2[i] + np.sin(2 * th_i)

        # Calculate bar rotation angle for each time by averaging over radii between 1 and 5 kpc #
        r_b = 2  # In kpc.
        r_s = 1  # In kpc.
        k = 0.0
        phase_in = 0.0
        for i in range(0, n_bins):
            if (r_m[i] < r_b) & (r_m[i] > r_s):
                k = k + 1.
                phase_in = phase_in + 0.5 * np.arctan2(beta_2[i], alpha_2[i])
        phase_in = phase_in / k

        # Transform back -tangle to horizontal position #
        z_pos = z[:]
        y_pos = np.cos(-phase_in) * (y[:]) + np.sin(-phase_in) * (x[:])
        x_pos = np.cos(-phase_in) * (x[:]) - np.sin(-phase_in) * (y[:])

        print('Spent %.4s ' % (
                time.time() - local_time) + 's in rotate_Jz from analysis_utilities.py' if verbose else '--------')
        return x_pos, y_pos, z_pos

    @staticmethod
    def rotate_Jz(masses, coordinates, velocities, verbose=False):
        """
        Rotate a galaxy such that its angular momentum is along the z axis.
        :param masses: stellar particle masses
        :param coordinates: stellar particle coordinates
        :param velocities: stellar particle velocities
        :param verbose: boolean: print duration of function
        :return: rotated_coordinates, rotated_velocities, prc_angular_momentum, glx_angular_momentum
        """
        print('Entering rotate_Jz from analysis_utilities.py')
        local_time = time.time()  # Start the local time.

        # Calculate the angular momentum of the galaxy #
        prc_angular_momentum = masses[:, np.newaxis] * np.cross(coordinates, velocities)  # In Msun kpc km s^-1.
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # In Msun kpc km s^-1.

        # Define the rotation matrices #
        a = np.matrix([glx_angular_momentum[0], glx_angular_momentum[1], glx_angular_momentum[2]]) / np.linalg.norm(
            [glx_angular_momentum[0], glx_angular_momentum[1], glx_angular_momentum[2]])
        b = np.matrix([0, 0, 1])
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = np.dot(a, b.T)
        vx = np.matrix([[0, -v[0, 2], v[0, 1]], [v[0, 2], 0, -v[0, 0]], [-v[0, 1], v[0, 0], 0]])
        transform = np.eye(3, 3) + vx + (vx * vx) * ((1 - c[0, 0]) / s ** 2)

        # Rotate the coordinates and velocities #
        rotated_coordinates = np.array([np.matmul(transform, coordinates[i].T) for i in
                                        range(0, len(coordinates))])[
                              :, 0]
        rotated_velocities = np.array([np.matmul(transform, velocities[i].T) for i in
                                       range(0, len(velocities))])[:, 0]

        # Calculate the rotated angular momentum of the galaxy #
        prc_angular_momentum = masses[:, np.newaxis] * np.cross(rotated_coordinates,
                                                                rotated_velocities)  # In Msun kpc km s^-1.
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # In Msun kpc km s^-1.

        print('Spent %.4s ' % (
                time.time() - local_time) + 's in rotate_Jz from analysis_utilities.py' if verbose else '--------')
        return rotated_coordinates, rotated_velocities, prc_angular_momentum, glx_angular_momentum


def binned_median_1sigma(x_data, y_data, bin_type, n_bins, log=False):
    """
    Calculate the binned median and 1-sigma lines in either equal number of width bins.
    :param x_data: x-axis data.
    :param y_data: y-axis data.
    :param bin_type: equal number or width type of the bin.
    :param n_bins: number of the bin.
    :param log: boolean.
    :return: x_values, median, shigh, slow
    """
    if log is True:
        x = np.log10(x_data)
    else:
        x = x_data
    x_low = min(x)

    if bin_type == 'equal_number':
        # Declare arrays to store the data #
        n_bins = np.quantile(np.sort(x), np.linspace(0, 1, n_bins + 1))
        slow = np.zeros(len(n_bins))
        shigh = np.zeros(len(n_bins))
        median = np.zeros(len(n_bins))
        x_values = np.zeros(len(n_bins))

        # Loop over all bins and calculate the median and 1-sigma lines #
        for i in range(len(n_bins) - 1):
            index, = np.where((x >= n_bins[i]) & (x < n_bins[i + 1]))
            x_values[i] = np.mean(x_data[index])
            if len(index) > 0:
                median[i] = np.nanmedian(y_data[index])
                slow[i] = np.nanpercentile(y_data[index], 15.87)
                shigh[i] = np.nanpercentile(y_data[index], 84.13)

        return x_values, median, shigh, slow

    elif bin_type == 'equal_width':
        # Declare arrays to store the data #
        bin_width = (max(x) - min(x)) / n_bins
        slow = np.zeros(n_bins)
        shigh = np.zeros(n_bins)
        median = np.zeros(n_bins)
        x_values = np.zeros(n_bins)

        # Loop over all bins and calculate the median and 1-sigma lines #
        for i in range(n_bins):
            index, = np.where((x >= x_low) & (x < x_low + bin_width))
            x_values[i] = np.mean(x_data[index])
            if len(index) > 0:
                median[i] = np.nanmedian(y_data[index])
                slow[i] = np.nanpercentile(y_data[index], 15.87)
                shigh[i] = np.nanpercentile(y_data[index], 84.13)
            x_low += bin_width

        return x_values, median, shigh, slow


def binned_sum(x_data, y_data, n_bins, log=False):
    """
    Calculate the binned sum line.
    :param x_data: x-axis data.
    :param y_data: y-axis data.
    :param n_bins: number of the bin.
    :param log: boolean.
    :return: x_values, sum
    """
    if log is True:
        x = np.log10(x_data)
    else:
        x = x_data
    x_low = min(x)

    # Declare arrays to store the data #
    bin_width = (max(x) - min(x)) / n_bins
    sum = np.zeros(n_bins)
    x_values = np.zeros(n_bins)

    # Loop over all bins and calculate the sum line #
    for i in range(n_bins):
        index, = np.where((x >= x_low) & (x < x_low + bin_width))
        x_values[i] = np.mean(x_data[index])
        if len(index) > 0:
            sum[i] = np.sum(y_data[index])
        x_low += bin_width
    return x_values, sum


class RotateCoordinates:
    """
    Rotate coordinates and velocities wrt different quantities.
    """

    @staticmethod
    def rotate_densest(prc_unit_vector, glx_unit_vector):
        """
        Rotate first about z-axis to set y=0and then about the y-axis to set z=0
        :param prc_unit_vector:
        :param glx_unit_vector:
        :return: prc_unit_vector, glx_unit_vector
        """
        # Calculate the ra and el of the (unit vector of) angular momentum for each particle #
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        el = np.degrees(np.arcsin(prc_unit_vector[:, 2]))

        # Create HEALPix map #
        nside = 2 ** 4  # Define the resolution of the grid (number of divisions along the side of a base-resolution grid cell).
        hp = HEALPix(nside=nside)  # Initialise the HEALPix pixelisation class.
        indices = hp.lonlat_to_healpix(ra * u.deg,
                                       el * u.deg)  # Create a list of HEALPix indices from particles' ra and el.
        densities = np.bincount(indices,
                                minlength=hp.npix)  # Count number of data points in each HEALPix grid cell.

        # Perform a top-hat smoothing on the densities #
        smoothed_densities = np.zeros(hp.npix)
        # Loop over all grid cells #
        for i in range(hp.npix):
            mask = hlp.query_disc(nside, hlp.pix2vec(nside, i),
                                  np.pi / 6.0)  # Do a 30degree cone search around each grid cell.
            smoothed_densities[i] = np.mean(
                densities[mask])  # Average the densities of the ones inside and assign this value to the grid cell.

        # Find location of density maximum and plot its positions and the ra (lon) and el (lat) of the galactic angular momentum #
        index_densest = np.argmax(smoothed_densities)
        lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2

        # Calculate the rotation matrices and combine them #
        ra = np.float(lon_densest)
        el = np.float(lat_densest)
        print(densities[index_densest])
        # ra = np.arctan2(glx_unit_vector[1], glx_unit_vector[0])
        # el = np.arcsin(glx_unit_vector[2])

        Rz = np.array([[np.cos(ra), np.sin(ra), 0], [-np.sin(ra), np.cos(ra), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(el), 0, np.sin(el)], [0, 1, 0], [-np.sin(el), 0, np.cos(el)]])
        Ryz = np.matmul(Ry, Rz)

        prc_unit_vector = np.matmul(Ryz, prc_unit_vector[..., None]).squeeze()
        glx_unit_vector = np.matmul(Ryz, glx_unit_vector)

        return prc_unit_vector, glx_unit_vector

    @staticmethod
    def rotate_component(stellar_data_tmp, mask):
        """
        Rotate a component such that its angular momentum is along the z axis.
        :param stellar_data_tmp: from read_add_attributes.py.
        :return: coordinates, velocities, component_data
        """
        # Select on particles that belong to the component #
        component_data = {}
        for attribute in ['Coordinates', 'Mass', 'Velocity']:
            component_data[attribute] = np.copy(stellar_data_tmp[attribute])[mask]

        # Calculate the angular momentum of the component #
        prc_angular_momentum = component_data['Mass'][:, np.newaxis] * np.cross(component_data['Coordinates'],
                                                                                component_data[
                                                                                    'Velocity'])  # In Msun kpc km s^-1.
        component_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # In Msun kpc km s^-1.

        # Define the rotation matrices #
        a = np.matrix([component_angular_momentum[0], component_angular_momentum[1],
                       component_angular_momentum[2]]) / np.linalg.norm(
            [component_angular_momentum[0], component_angular_momentum[1], component_angular_momentum[2]])
        b = np.matrix([0, 0, 1])
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = np.dot(a, b.T)
        vx = np.matrix([[0, -v[0, 2], v[0, 1]], [v[0, 2], 0, -v[0, 0]], [-v[0, 1], v[0, 0], 0]])
        transform = np.eye(3, 3) + vx + (vx * vx) * ((1 - c[0, 0]) / s ** 2)

        # Rotate the coordinates and velocities #
        coordinates = np.array([np.matmul(transform, component_data['Coordinates'][i].T) for i in
                                range(0, len(component_data['Coordinates']))])[:, 0]
        velocities = np.array([np.matmul(transform, component_data['Velocity'][i].T) for i in
                               range(0, len(component_data['Velocity']))])[:, 0]

        return coordinates, velocities, component_data


def linear_resample(original_array, target_length):
    """
    Resample (downsample or upsample) an array.
    :param original_array: original array
    :param target_length: target length
    :return: interpolated_array
    """
    original_array = np.array(original_array, dtype=np.float)
    index_arr = np.linspace(0, len(original_array) - 1, num=target_length, dtype=np.float)
    index_floor = np.array(index_arr, dtype=np.int)  # Round down.
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor  # Remain.

    val1 = original_array[index_floor]
    val2 = original_array[index_ceil % len(original_array)]
    interpolated_array = val1 * (1.0 - index_rem) + val2 * index_rem
    assert (len(interpolated_array) == target_length)
    return interpolated_array


def sersic_profile(r, I_0b, b, n):
    """
    Calculate a Sersic profile.
    :param r: radius.
    :param I_0b: Spheroid central intensity.
    :param b: Sersic b parameter
    :param n: Sersic index
    :return: I_0b * np.exp(-(r / b) ** (1 / n))
    """
    return I_0b * np.exp(-(r / b) ** (1 / n))  # b = R_eff / b_n ^ n


def exponential_profile(r, I_0d, R_d):
    """
    Calculate an exponential profile.
    :param r: radius
    :param I_0d: Disc central intensity.
    :param R_d: Disc scale length.
    :return: I_0d * np.exp(-r / R_d)
    """
    return I_0d * np.exp(-r / R_d)


def total_profile(r, I_0d, R_d, I_0b, b, n):
    """
    Calculate a total (Sersic + exponential) profile.
    :param r: radius.
    :param I_0d: Disc central intensity.
    :param R_d: Disc scale length.
    :param I_0b: Spheroid central intensity.
    :param b: Sersic b parameter.
    :param n: Sersic index.
    :return: exponential_profile(r, I_0d, R_d) + sersic_profile(r, I_0b, b, n)
    """
    y = exponential_profile(r, I_0d, R_d) + sersic_profile(r, I_0b, b, n)
    return y


def sersic_b_n(n):
    """
    Calculate the Sersic b parameter.
    :param n: Sersic index.
    :return: b_n
    """
    if n <= 0.36:
        b_n = 0.01945 + n * (- 0.8902 + n * (10.95 + n * (- 19.67 + n * 13.43)))
    else:
        x = 1.0 / n
        b_n = -1.0 / 3.0 + 2. * n + x * (
                4.0 / 405. + x * (46. / 25515. + x * (131. / 1148175 - x * 2194697. / 30690717750.)))
    return b_n
