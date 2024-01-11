import h5py
import time
import psutil
import schwimmbad

import numpy as np

from functools import partial

date = time.strftime('%d_%m_%y_%H%M')  # Date.


def read_hdf5_header(d, f, attribute, verbose=False):
    """
    Read a header 'attribute' from an hdf5 file 'f' inside a directory 'd'.
    :param d: directory path
    :param f: file name
    :param attribute: attribute name
    :param verbose: Boolean: print duration of function(s)
    :return: data
    """
    print('Entering read_hdf5_header from KETJU_IO.py to read', attribute)
    local_time = time.time()  # Start the local time.

    # Open the file and read the attribute if it exist, if not then print the available attributes #
    with h5py.File(d + f, 'r') as group:
        if attribute in group['Header'].attrs.keys():
            data = group['Header'].attrs[attribute]
        else:
            attributes = list(group['Header'].attrs.keys())
            print('Invalid attribute! Please choose one of the following attributes: \n', attributes)
            exit()

    print('Spent %.4s ' % (
            time.time() - local_time) + 's in read_hdf5_header from KETJU_IO.py' if verbose else '--------')
    return data


def read_hdf5_group(d, f, group, verbose=False):
    """
    Read a 'group' from an hdf5 file 'f' inside a directory 'd'.
    :param d: directory path
    :param f: file name
    :param group: group name
    :param verbose: Boolean: print duration of function(s)
    :return: data
    """
    print('Entering read_hdf5_group from KETJU_IO.py to read', group)
    local_time = time.time()  # Start the local time.

    # Open the file and read the group if it exist, if not then print the available groups #
    with h5py.File(d + f, 'r') as groups:
        if group in groups.keys():
            data = np.array(groups.get(group))
        else:
            groups = list(groups.keys())
            print('Invalid group! Please choose one of the following groups: \n', groups)
            exit()

    print(
        'Spent %.4s ' % (time.time() - local_time) + 's in read_hdf5_group from KETJU_IO.py' if verbose else '--------')
    return data


def read_hdf5_dataset(d, f, group, dataset, hfree=False, physical=False, verbose=False):
    """
    Read a 'dataset' inside a 'group' from an hdf5 file 'f' inside a directory 'd'.
    :param d: directory path
    :param f: file name
    :param group: group name
    :param dataset: dataset name
    :param hfree: boolean: convert to hfree units
    :param physical: boolean: convert to physical units
    :param verbose: Boolean: print duration of function(s)
    :return: data
    """
    print('Entering read_hdf5_dataset from KETJU_IO.py to read', group, '/', dataset)
    local_time = time.time()  # Start the local time.

    # Open the file and read the dataset if it exist, if not then print the available groups or datasets #
    with h5py.File(d + f, 'r') as groups:
        if group in groups.keys():
            if dataset in groups[group].keys():
                data = np.array(groups[group][dataset])

                # Convert units to h-free and/or physical #
                if hfree is True:
                    data = convert_to_hfree_units(d, f, data, dataset)
                if physical is True:
                    data = convert_to_physical_units(d, f, data, dataset)
            else:
                datasets = list(groups[group].keys())
                print('Invalid dataset! Please choose one of the following datasets: \n', datasets)
                exit()
        else:
            groups = list(groups.keys())
            print('Invalid group! Please choose one of the following groups: \n', groups)
            exit()

    print('Spent %.4s ' % (
            time.time() - local_time) + 's in read_hdf5_dataset from KETJU_IO.py' if verbose else '--------')
    return data


def read_hdf5_dataset_parallel(d, f, group, dataset, pt5_ids, pt5_coordinates, pt_coordinates, pt5_id, radial_threshold,
                               hfree=False, physical=False, verbose=False):
    """
    Read in parallel a 'dataset' inside a 'group' from an hdf5 file 'f' inside a directory 'd'.
    :param d: directory path
    :param f: file name
    :param group: group name
    :param dataset: dataset name
    :param pt5_ids: ids of black hole particles
    :param pt5_coordinates: coordinates of black hole particles
    :param pt_coordinates: coordinates of particles
    :param pt5_id: id of a mapped black hole particle
    :param radial_threshold: maximum distance from a black hole mass in kpc
    :param hfree: boolean: convert to hfree units
    :param physical: boolean: convert to physical units
    :param verbose: Boolean: print duration of function(s)
    :return: data
    """
    print('Entering read_hdf5_dataset_parallel from KETJU_IO.py to read', group, '/', dataset)
    local_time = time.time()  # Start the local time.

    # Open the file and read the dataset if it exist, if not then print the available groups or datasets #
    with h5py.File(d + f, 'r') as groups:
        if group in groups.keys():
            if dataset in groups[group].keys():
                data = np.array(groups[group][dataset])

                # Convert units to h-free and/or physical #
                if hfree is True:
                    data = convert_to_hfree_units(d, f, data, dataset)
                if physical is True:
                    data = convert_to_physical_units(d, f, data, dataset)

                # Select a specific black hole #
                pt5_mask, = np.where(pt5_ids == pt5_id)
                pt5_coordinates = pt5_coordinates[pt5_mask]
                radial_threshold = radial_threshold[pt5_mask]

                # Select particles within 'radial_threshold' kpc from a specific black hole and mask the data #
                mask, = np.where((np.linalg.norm(pt_coordinates - pt5_coordinates, axis=1) <= radial_threshold))
                data = data[mask]

            else:
                datasets = list(groups[group].keys())
                print('Invalid dataset! Please choose one of the following datasets: \n', datasets)
                exit()
        else:
            groups = list(groups.keys())
            print('Invalid group! Please choose one of the following groups: \n', groups)
            exit()

    print('Spent %.4s ' % (
            time.time() - local_time) + 's in read_hdf5_dataset from KETJU_IO.py' if verbose else '--------')
    return data


def read_hdf5_galaxies_parallel(d, f, group, dataset, mass_threshold=0, radial_threshold=0, hfree=False,
                                physical=False, verbose=False):
    """
    Read for all galaxies in parallel a 'dataset' inside a 'group' from an hdf5 file 'f' inside a directory 'd'.
    :param d: directory path
    :param f: file name
    :param group: group name
    :param dataset: dataset name
    :param mass_threshold: minimum black hole mass in Msun
    :param radial_threshold: maximum distance from a black hole mass in kpc
    :param hfree: boolean: convert to hfree units
    :param physical: boolean: convert to physical units
    :param verbose: Boolean: print duration of function(s)
    :return: data
    """
    print('Entering read_hdf5_galaxies_parallel from KETJU_IO.py to read', group, '/', dataset)
    local_time = time.time()  # Start the local time.

    # Black hole data #
    pt5_ids = read_hdf5_dataset(d, f, 'PartType5', 'ParticleIDs', hfree=hfree, physical=physical)
    pt5_masses = read_hdf5_dataset(d, f, 'PartType5', 'BH_Mass', hfree=hfree, physical=physical) * 1e10  # In Msun.
    pt5_coordinates = read_hdf5_dataset(d, f, 'PartType5', 'Coordinates', hfree=hfree, physical=physical)  # In kpc.

    # Particle coordinates #
    pt_coordinates = read_hdf5_dataset(d, f, group, 'Coordinates', hfree=hfree, physical=physical)  # In kpc.

    # Select only black holes with masses greater than 'mass_threshold' Msun #
    pt5_mask, = np.where(pt5_masses > mass_threshold)

    # Create a pool based on the number of available CPUs #
    processes = max(1, min(len(pt5_mask), len(psutil.Process().cpu_affinity())))
    if processes == 1:
        pool = schwimmbad.SerialPool()
    else:
        pool = schwimmbad.MultiPool(processes=processes)

    # Map the partial function 'read_hdf5_dataset_parallel' to each black hole particle ID #
    function = partial(read_hdf5_dataset_parallel, d, f, group, dataset, pt5_ids[pt5_mask], pt5_coordinates[pt5_mask],
                       pt_coordinates, radial_threshold=radial_threshold[pt5_mask], hfree=hfree, physical=physical,
                       verbose=verbose)

    data = pool.map(function, pt5_ids[pt5_mask])
    # data = np.concatenate(list(pool.map(function, pt5_ids[pt5_mask])), axis=0)
    pool.close()

    print('Spent %.4s ' % (
            time.time() - local_time) + 's in read_hdf5_galaxies_parallel from KETJU_IO.py' if verbose else '--------')
    return data


def convert_to_hfree_units(d, f, data, dataset, verbose=False):
    """
    Convert a quantity 'data' to h-free units.
    :param d: directory path
    :param f: file name
    :param data: quantity
    :param dataset: dataset name
    :param verbose: Boolean: print duration of function(s)
    :return: data
    """
    print('Entering convert_to_hfree_units from KETJU_IO.py to convert', dataset)
    local_time = time.time()  # Start the local time.

    # Split the datasets into groups #
    spatial = ['BH_AccreationLength', 'BH_Mass', 'BH_Mdot', 'Coordinates', 'InitialMass', 'Masses', 'SmoothingLength']
    voluminous = ['Densities']

    # Read the hubble parameter from the header and convert the units #
    hubble_parameter = read_hdf5_header(d, f, 'HubbleParam')
    if dataset in spatial:
        data *= hubble_parameter ** (-1)
    if dataset in voluminous:
        data *= hubble_parameter ** 2

    print(
        'Spent %.4s ' % (
                time.time() - local_time) + 's in convert_to_hfree_units from KETJU_IO.py' if verbose else '--------')
    return data


def convert_to_physical_units(d, f, data, dataset, verbose=False):
    """
    Convert a quantity 'data' to physical units.
    :param d: directory path
    :param f: file name
    :param data: quantity
    :param dataset: dataset name
    :param verbose: Boolean: print duration of function(s)
    :return: data
    """
    print('Entering convert_to_physical_units from KETJU_IO.py to convert', dataset)
    local_time = time.time()  # Start the local time.

    # Split the datasets into groups #
    spatial = ['BH_AccreationLength', 'Coordinates', 'SmoothingLength']
    temporal = ['Velocities']
    voluminous = ['Densities']

    # Read the scale factor from the header and convert the units #
    scale_factor = read_hdf5_header(d, f, 'Time')
    if dataset in spatial:
        data *= scale_factor
    if dataset in temporal:
        data *= scale_factor ** 0.5
    if dataset in voluminous:
        data *= scale_factor ** (-3)

    print('Spent %.4s ' % (
            time.time() - local_time) + 's in convert_to_physical_units from KETJU_IO.py' if verbose else '--------')
    return data


if __name__ == '__main__':
    d = '/scratch/pjohanss/cosmo_run/ketju_run/'
    d = '/scratch/pjohanss/soisonja/ketju_run/output/'
    f = '100_Mpc_box_high_res_halo_2_ketju_run_091.hdf5'

    # pt5_ids = read_hdf5_dataset(d, f, 'PartType5', 'ParticleIDs')
    pt5_masses = read_hdf5_dataset(d, f, 'PartType5', 'BH_Mass', hfree=True, physical=True) * 1e10  # In Msun.
    print(np.log10(np.sum(pt5_masses)))
    print(len(pt5_masses))
    # pt5_coordinates = read_hdf5_dataset(d, f, 'PartType5', 'Coordinates', hfree=True, physical=True)  # In kpc.

    # print(np.linalg.norm(pt5_coordinates[pt5_ids == 15521194] - pt5_coordinates[pt5_ids == 13494108], axis=1))
    # print(read_hdf5_header(d, f, 'Time'))

    # pt1_mass = read_hdf5_header(d, f, 'MassTable')[1] * 1e10 / read_hdf5_header(d, f, 'HubbleParam')  # In Msun.
    # pt0_masses = read_hdf5_dataset(d, f, 'PartType0', 'Masses', hfree=True, physical=True) * 1e10  # In Msun.
    # pt4_masses = read_hdf5_dataset(d, f, 'PartType4', 'Masses', hfree=True, physical=True) * 1e10  # In Msun.
    # pt5_masses = read_hdf5_dataset(d, f, 'PartType5', 'Masses', hfree=True, physical=True) * 1e10  # In Msun.

    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Check the conservation of mass in the simulation #
    # print(len(np.where(np.log10(pt0_masses) > 5.7)[0]))
    # print((np.sum(len(pt0_masses) + np.sum(len(pt4_masses) + np.sum(len(pt5_masses))))) - 410 ** 3)

    # # # # # # # # # # # # # # # # # # # #
    # Create a dictionary of dictionaries #
    # import os
    # import re
    #
    # # Find all snap files, get their names,f and sort them #
    # d = '/scratch/pjohanss/cosmo_run/gadget_run/'
    # snap_names = os.listdir(d)
    # for i in range(len(snap_names)):
    #     snap_names[i] = re.split('run_|.hdf5', snap_names[i])[1]
    # snap_names = sorted(snap_names)
    #
    # # Create an empty dictionary #
    # dict = {}
    #
    # # Loop over all snap files and create an internal dictionary for each snapshot #
    # for j, name in enumerate(snap_names):
    #     internal_dict = {}
    #     dict[name] = internal_dict
    #
    #     # Give some values #
    #     internal_dict['IDs'] = np.array([1, 2, 3])
    #     internal_dict['r'] = np.array([4, 5, 6])
    #
    #     # If you find an ID in one of the snapshots add an extra value in the radii elements #
    #     if name == '004' and 2 in internal_dict['IDs']:
    #         print(dict[name]['r'])
    #         new_rs = np.array([8, 9, 10])
    #         dict[name]['r'] = np.append(dict[name]['r'], 7)
    #         dict[name]['r'] = np.append(dict[name]['r'], new_rs)
    #
    # print(dict['004']['r'][np.where(dict['004']['IDs'] == 3)])

    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Check PartType2 contamination
    # pt5_ids = pt5_ids[pt5_masses > 7.5e7]
    # pt5_coordinates = pt5_coordinates[pt5_masses > 7.5e7]
    # used_ids = bh_labels = [44153861, 47145713, 37027654, 37574716, 52881207, 47804170, 50663072, 61621144, 58305478,
    #                         13494108, 15521194]
    # for i in range(len(pt5_coordinates)):
    #     if pt5_ids[i] in used_ids:
    #         mask, = np.where((np.linalg.norm(pt2_coordinates - pt5_coordinates[i], axis=1) <= 300))
    #         print('ID:', pt5_ids[i], 'particle number:', len(mask), 'total PT2 mass:', np.sum(pt2_masses[mask]))
