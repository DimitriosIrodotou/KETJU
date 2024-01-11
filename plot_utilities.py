import time
# import satellite_utilities

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

from matplotlib import gridspec

# Plot parameters #
res = 512
box_size = 0.06

# Aesthetic parameters #
style.use("classic")
plt.rcParams.update({'font.family': 'serif'})


def set_axis(axis, x_lim=None, y_lim=None, x_scale=None, y_scale=None, x_label=None, y_label=None, log=False,
             semilog_x=False, semilog_y=False, aspect=True, which='both', size=20, verbose=False):
    """
    Set axis parameters.
    :param axis: name of the axis
    :param x_lim: x axis limits
    :param y_lim: y axis limits
    :param x_scale: x axis scale
    :param y_scale: y axis scale
    :param x_label: x axis label
    :param y_label: y axis label
    :param log: boolean: data in log-space or not
    :param semilog_x: boolean: data in semi-log-space or not
    :param semilog_y: boolean: data in semi-log-space or not
    :param which: major, minor or both for grid and ticks
    :param size: text size
    :param verbose: Boolean: print duration of function(s)
    :return: None
    """
    print('Invoking set_axis from plot_utilities.py')
    local_time = time.time()  # Start the local time.

    # Set axis limits #
    if x_lim:
        axis.set_xlim(x_lim)
    if y_lim:
        axis.set_ylim(y_lim)

    # Set axis labels #
    if x_label:
        axis.set_xlabel(x_label, size=size)
    if y_label:
        axis.set_ylabel(y_label, size=size)

    # Set axis scales #
    if x_scale:
        axis.set_xscale(x_scale)
    if y_scale:
        axis.set_yscale(y_scale)

    if not x_lim and not x_label:
        axis.set_xticks([])
        axis.set_xticklabels([])
    if not y_lim and not y_label:
        axis.set_yticks([])
        axis.set_yticklabels([])

    # Set axis ratio #
    if aspect is True:
        if log is True:
            x_min, x_max = axis.get_xbound()
            y_min, y_max = axis.get_ybound()
            data_ratio = (np.log10(y_max) - np.log10(y_min)) / (np.log10(x_max) - np.log10(x_min))
        elif semilog_x is True:
            x_min, x_max = axis.get_xbound()
            y_min, y_max = axis.get_ybound()
            data_ratio = (y_max - y_min) / (np.log10(x_max) - np.log10(x_min))
        elif semilog_y is True:
            x_min, x_max = axis.get_xbound()
            y_min, y_max = axis.get_ybound()
            data_ratio = (np.log10(y_max) - np.log10(y_min)) / (x_max - x_min)
        else:
            data_ratio = axis.get_data_ratio()
        axis.set_aspect(1. / data_ratio, adjustable='box')

    # Set grid and tick parameters #
    axis.set_axisbelow(True)  # Place grid lines below other artists.
    axis.grid(True, which=which, axis='both', color='gray', linestyle='-', alpha=0.7)
    axis.tick_params(direction='out', which='major', top=False, bottom='on', left='on', right=False, labelsize=size,
                     width=2, length=size / 3)
    axis.tick_params(direction='out', which='minor', top=False, bottom='on', left='on', right=False, labelsize=size,
                     width=2, length=size / 5)

    print('Spent %.4s ' % (
            time.time() - local_time) + 's in set_axis from plot_utilities.py' if verbose else '--------')
    return None


def create_colorbar(axis, plot, label, orientation='vertical', top=True, ticks=None, ticklabels=None, size=20,
                    extend='neither', verbose=False):
    """
    Create a colorbar.
    :param axis: colorbar axis
    :param plot: corresponding plot
    :param label: colorbar label
    :param orientation: colorbar orientation
    :param top: move ticks and labels on top of the colorbar.
    :param ticks: array of ticks
    :param ticklabels: array of tick labels
    :param size: text size
    :param extend: make pointed end(s) for out-of-range values
    :param verbose: Boolean: print duration of function(s)
    :return: None
    """
    print('Entering create_colorbar from plot_utilities.py')
    local_time = time.time()  # Start the local time.

    cbar = plt.colorbar(plot, cax=axis, ticks=ticks, orientation=orientation, extend=extend)
    cbar.set_label(label, size=size)
    axis.tick_params(direction='out', which='major', right='on', labelsize=size, width=2, length=size / 3)
    axis.tick_params(direction='out', which='minor', right='on', labelsize=size, width=2, length=size / 5)

    if orientation == 'horizontal':
        if ticklabels:
            cbar.ax.set_xticklabels(ticklabels)
        if top is True:
            axis.xaxis.tick_top()
            axis.xaxis.set_label_position("top")
            axis.tick_params(direction='out', which='major', top='on', labelsize=size, width=2, length=size / 3)
            axis.tick_params(direction='out', which='minor', top='on', labelsize=size, width=2, length=size / 5)
    else:
        if ticklabels:
            cbar.ax.set_yticklabels(ticklabels)

    print('Spent %.4s ' % (
            time.time() - local_time) + 's in create_colorbar from plot_utilities.py' if verbose else '--------')
    return None


def set_twin_axis(axis, x_lim=None, x_scale=None, x_label=None, ticks=None, aspect=True, size=20, verbose=False):
    """
    Set twin axis parameters.
    :param axis: name of the axis
    :param x_lim: x axis limits
    :param x_scale: x axis scale
    :param x_label: x axis label
    :param aspect: boolean: create square plot or not
    :param ticks: array of ticks
    :param which: major, minor or both for grid and ticks
    :param size: text size
    :param verbose: Boolean: print duration of function(s)
    :return: None
    """
    print('Invoking set_twin_axis from plot_utilities.py')
    local_time = time.time()  # Start the local time.

    axis.set_facecolor("None")

    # Set axes limits #
    axis.set_xlim(x_lim)

    # Set axis scales #
    if x_scale:
        axis.set_xscale(x_scale)

    # Set axes labels #
    axis.set_yticklabels([])
    # axis.set_xticklabels(lb)
    axis.set_xlabel(x_label, size=size)
    axis.xaxis.set_label_position('top')

    # Set axis ratio #
    if aspect is True:
        axis.set_aspect(1 / axis.get_data_ratio(), adjustable='box')

    # Set grid and tick parameters #
    axis.xaxis.tick_top()
    # axis.set_xticks(ticks)
    axis.tick_params(direction='out', which='major', top='on', bottom=False, left=False, right=False, labelsize=size,
                     width=2, length=size / 3)
    axis.tick_params(direction='out', which='minor', top='on', bottom=False, left=False, right=False, labelsize=size,
                     width=2, length=size / 5)

    print('Spent %.4s ' % (
            time.time() - local_time) + 's in set_twin_axis from plot_utilities.py' if verbose else '--------')
    return None


def set_axes_evolution(axis, axis2, y_lim=None, y_scale=None, y_label=None, aspect=True, which='both', size=20,
                       verbose=False):
    """
    Set axes parameters for evolution plots.
    :param axis: name of the axis
    :param axis2: name of the twin axis
    :param y_lim: y axis limit
    :param y_scale: y axis scale
    :param y_label: y axis label
    :param aspect: boolean: create square plot or not
    :param which: major, minor or both for grid and ticks
    :param size: text size
    :param verbose: Boolean: print duration of function(s)
    :return: None
    """
    print('Invoking set_axes_evolution from plot_utilities.py')
    local_time = time.time()  # Start the local time.

    # Convert redshifts to look back times #
    z = np.array([5.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.0])
    times = satellite_utilities.return_lookbacktime_from_a((z + 1.0) ** (-1.0))  # In Gyr.

    # Set the text format #
    lb = []
    for v in z:
        if v >= 1.0:
            lb += ["%.0f" % v]
        else:
            if v:
                lb += ["%.1f" % v]
            else:
                lb += ["%.0f" % v]

    # Set axes limits #
    axis2.xaxis.tick_top()
    axis2.set_xticks(times)
    axis2.set_xticklabels(lb)
    axis2.xaxis.set_label_position('top')
    if y_lim:
        axis.set_ylim(y_lim)
    axis.set_xlim(13, 0)
    axis2.set_xlim(axis.get_xlim())

    # Set axes scales #
    if y_scale:
        axis.set_yscale(y_scale)

    # Set axes labels #
    if y_label:
        axis2.set_yticklabels([])
        axis.set_ylabel(y_label, size=size)
    axis.set_xlabel(r'$\mathrm{t_{look}/Gyr}$', size=size)
    axis2.set_xlabel(r'$\mathrm{z}$', size=size)

    # Set axes ratio #
    if aspect is True:
        axis.set_aspect(1 / axis.get_data_ratio(), adjustable='box')
        axis2.set_aspect(1 / axis2.get_data_ratio(), adjustable='box')

    # Set grid and tick parameters #
    axis.set_axisbelow(True)  # Place grid lines below other artists.
    axis.grid(True, which=which, axis='both', color='gray', linestyle='-', alpha=0.7)
    axis.tick_params(direction='out', which='major', top=False, bottom='on', left='on', right=False, labelsize=size,
                     width=2, length=size / 3)
    axis.tick_params(direction='out', which='minor', top=False, bottom='on', left='on', right=False, labelsize=size,
                     width=2, length=size / 5)
    axis2.tick_params(direction='out', which='major', top='on', bottom=False, left=False, right=False, labelsize=size,
                      width=2, length=size / 3)
    axis2.tick_params(direction='out', which='minor', top='on', bottom=False, left=False, right=False, labelsize=size,
                      width=2, length=size / 5)

    print('Spent %.4s ' % (
            time.time() - local_time) + 's in set_axes_evolution from plot_utilities.py' if verbose else '--------')
    return None


def create_axes_combinations(res=res, box_size=box_size, contour=False, colorbar=False, velocity_vectors=False,
                             multiple=False, multiple2=False, multiple3=False, multiple4=False, multiple5=False,
                             multiple6=False, multiple7=False, mollweide=False, multiple8=False, multiple9=False,
                             multiple10=False, multiple11=False, multiple12=False, multiple13=False, multiple14=False):
    """
    Generate plot axes.
    :param res: resolution
    :param box_size: box_size
    :param contour: 2x2 matrix plus colorbar
    :param colorbar: 2x1 matrix plus colorbar
    :param velocity_vectors: 2x1 matrix
    :param multiple: 2x6 matrix plus 6 colorbars
    :param multiple2: 6x3 matrix
    :param multiple3: 5x4 matrix plus colorbar
    :param multiple4: 1x3 matrix
    :param multiple5: 3x3 matrix
    :param multiple6: 3x3 matrix plus colorbar
    :param multiple7: 3x3 matrix plus 9 colorbars
    :param mollweide: 3x3 mollweide projection
    :param multiple8: 6x3 matrix
    :param multiple9: 8x3 matrix
    :param multiple10: 2x3 matrix
    :param multiple11: 4x3 matrix
    :param multiple12: 3x4 matrix
    :param multiple13: 2x3 matrix
    :param multiple14: 2x1 matrix
    :return: axes
    """
    print('Invoking create_axes_combinations from plot_utilities.py')

    # Set the axes values #
    area = (box_size / res) ** 2  # Calculate the area.
    x = np.linspace(-0.5 * box_size, +0.5 * box_size, res + 1)
    y = np.linspace(-0.5 * box_size, +0.5 * box_size, res + 1)
    y2 = np.linspace(-0.5 * box_size, +0.5 * box_size, res / 2 + 1)

    # Generate the panels #
    if contour is True:
        gs = gridspec.GridSpec(2, 3, hspace=0.05, wspace=0.05, height_ratios=[1, 0.5], width_ratios=[1, 1, 0.05])
        axis00, axis01 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])
        axis10, axis11 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])
        axiscbar = plt.subplot(gs[:, 2])

        return axis00, axis01, axis10, axis11, axiscbar, x, y, y2, area

    elif colorbar is True:
        gs = gridspec.GridSpec(2, 2, hspace=0.05, wspace=0.05, height_ratios=[1, 0.5], width_ratios=[1, 0.05])
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])
        axiscbar = plt.subplot(gs[:, 1])

        return axis00, axis10, axiscbar, x, y, y2, area

    elif velocity_vectors is True:
        gs = gridspec.GridSpec(2, 1, hspace=0.05)
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])

        return axis00, axis10, x, y, y2, area

    elif multiple is True:
        gs = gridspec.GridSpec(3, 6, hspace=0.0, wspace=0.05, height_ratios=[1, 0.05, 1])
        axis00, axis01, axis02, axis03, axis04, axis05 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(
            gs[0, 2]), plt.subplot(gs[0, 3]), plt.subplot(gs[0, 4]), plt.subplot(gs[0, 5])
        axis10, axis11, axis12, axis13, axis14, axis15 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(
            gs[1, 2]), plt.subplot(gs[1, 3]), plt.subplot(gs[1, 4]), plt.subplot(gs[1, 5])
        axis20, axis21, axis22, axis23, axis24, axis25 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(
            gs[2, 2]), plt.subplot(gs[2, 3]), plt.subplot(gs[2, 4]), plt.subplot(gs[2, 5])

        return axis00, axis01, axis02, axis03, axis04, axis05, axis10, axis11, axis12, axis13, axis14, axis15, \
               axis20, axis21, axis22, axis23, axis24, axis25, x, y, area

    elif multiple2 is True:
        gs = gridspec.GridSpec(6, 3, hspace=0, wspace=0)
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axis30, axis31, axis32 = plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2])
        axis40, axis41, axis42 = plt.subplot(gs[4, 0]), plt.subplot(gs[4, 1]), plt.subplot(gs[4, 2])
        axis50, axis51, axis52 = plt.subplot(gs[5, 0]), plt.subplot(gs[5, 1]), plt.subplot(gs[5, 2])

        return axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, \
               axis40, axis41, axis42, axis50, axis51, axis52

    elif multiple3 is True:
        gs = gridspec.GridSpec(8, 4, hspace=0, wspace=0.05, height_ratios=[1, 0.5, 0.1, 1, 0.5, 0.1, 1, 0.5],
                               width_ratios=[1, 1, 1, 0.1])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis_space20, axis_space21, axis_space22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axis30, axis31, axis32 = plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2])
        axis40, axis41, axis42 = plt.subplot(gs[4, 0]), plt.subplot(gs[4, 1]), plt.subplot(gs[4, 2])
        axis_space50, axis_space51, axis_space52 = plt.subplot(gs[5, 0]), plt.subplot(gs[5, 1]), plt.subplot(gs[5, 2])
        axis60, axis61, axis62 = plt.subplot(gs[6, 0]), plt.subplot(gs[6, 1]), plt.subplot(gs[6, 2])
        axis70, axis71, axis72 = plt.subplot(gs[7, 0]), plt.subplot(gs[7, 1]), plt.subplot(gs[7, 2])
        axiscbar = plt.subplot(gs[:, 3])

        for axis in [axis_space20, axis_space21, axis_space22, axis_space50, axis_space51, axis_space52]:
            axis.axis('off')

        return axis00, axis01, axis02, axis10, axis11, axis12, axis30, axis31, axis32, axis40, axis41, axis42, \
               axis60, axis61, axis62, axis70, axis71, axis72, axiscbar, x, y, y2, area

    elif multiple4 is True:
        gs = gridspec.GridSpec(1, 3, hspace=0, wspace=0.05)
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])

        return axis00, axis01, axis02

    elif multiple5 is True:
        gs = gridspec.GridSpec(3, 3, hspace=0.05, wspace=0.05)
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])

        return axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22

    elif multiple6 is True:
        gs = gridspec.GridSpec(3, 4, hspace=0.05, wspace=0.05, width_ratios=[1, 1, 1, 0.1])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis20, axis21, axis22 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis40, axis41, axis42 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axiscbar = plt.subplot(gs[:, 3])

        return axis00, axis01, axis02, axis20, axis21, axis22, axis40, axis41, axis42, axiscbar

    elif multiple7 is True:
        gs = gridspec.GridSpec(6, 3, hspace=0.4, wspace=0, height_ratios=[0.05, 1, 0.05, 1, 0.05, 1])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axis30, axis31, axis32 = plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2])
        axis40, axis41, axis42 = plt.subplot(gs[4, 0]), plt.subplot(gs[4, 1]), plt.subplot(gs[4, 2])
        axis50, axis51, axis52 = plt.subplot(gs[5, 0]), plt.subplot(gs[5, 1]), plt.subplot(gs[5, 2])

        return axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, \
               axis40, axis41, axis42, axis50, axis51, axis52

    elif mollweide is True:
        gs = gridspec.GridSpec(3, 4, hspace=0, wspace=0, width_ratios=[1, 1, 1, 0.1])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0], projection='mollweide'), plt.subplot(gs[0, 1],
                                                                                            projection='mollweide'), \
                                 plt.subplot(gs[0, 2], projection='mollweide')
        axis10, axis11, axis12 = plt.subplot(gs[1, 0], projection='mollweide'), plt.subplot(gs[1, 1],
                                                                                            projection='mollweide'), \
                                 plt.subplot(gs[1, 2], projection='mollweide')
        axis20, axis21, axis22 = plt.subplot(gs[2, 0], projection='mollweide'), plt.subplot(gs[2, 1],
                                                                                            projection='mollweide'), \
                                 plt.subplot(gs[2, 2], projection='mollweide')
        axiscbar = plt.subplot(gs[:, 3])

        return axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axiscbar

    elif multiple8 is True:
        gs = gridspec.GridSpec(6, 3, hspace=0, wspace=0)
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axis30, axis31, axis32 = plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2])
        axis40, axis41, axis42 = plt.subplot(gs[4, 0]), plt.subplot(gs[4, 1]), plt.subplot(gs[4, 2])
        axis50, axis51, axis52 = plt.subplot(gs[5, 0]), plt.subplot(gs[5, 1]), plt.subplot(gs[5, 2])

        return axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, \
               axis40, axis41, axis42, axis50, axis51, axis52

    elif multiple9 is True:
        gs = gridspec.GridSpec(8, 3, hspace=0, wspace=0.05, height_ratios=[1, 0.5, 0.1, 1, 0.5, 0.1, 1, 0.5])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis_space20, axis_space21, axis_space22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axis30, axis31, axis32 = plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2])
        axis40, axis41, axis42 = plt.subplot(gs[4, 0]), plt.subplot(gs[4, 1]), plt.subplot(gs[4, 2])
        axis_space50, axis_space51, axis_space52 = plt.subplot(gs[5, 0]), plt.subplot(gs[5, 1]), plt.subplot(gs[5, 2])
        axis60, axis61, axis62 = plt.subplot(gs[6, 0]), plt.subplot(gs[6, 1]), plt.subplot(gs[6, 2])
        axis70, axis71, axis72 = plt.subplot(gs[7, 0]), plt.subplot(gs[7, 1]), plt.subplot(gs[7, 2])

        for axis in [axis_space20, axis_space21, axis_space22, axis_space50, axis_space51, axis_space52]:
            axis.axis('off')

        return axis00, axis01, axis02, axis10, axis11, axis12, axis30, axis31, axis32, axis40, axis41, axis42, \
               axis60, axis61, axis62, axis70, axis71, axis72

    elif multiple10 is True:
        gs = gridspec.GridSpec(2, 3, hspace=0, wspace=0.05, height_ratios=[1, 0.5])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])

        return axis00, axis01, axis02, axis10, axis11, axis12

    elif multiple11 is True:
        gs = gridspec.GridSpec(4, 3, hspace=0, wspace=0.05)
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axis30, axis31, axis32 = plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2])

        return axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32

    elif multiple12 is True:
        gs = gridspec.GridSpec(3, 4, hspace=0.05, wspace=0, width_ratios=[1, 1, 1, 0.1])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axiscbar = plt.subplot(gs[:, 3])

        return axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axiscbar, x, y, y2, area

    elif multiple13 is True:
        gs = gridspec.GridSpec(2, 3, hspace=0.05, wspace=0.1)
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])

        return axis00, axis01, axis02, axis10, axis11, axis12

    elif multiple14 is True:
        gs = gridspec.GridSpec(2, 1, hspace=0.05)
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])

        return axis00, axis10
    else:
        gs = gridspec.GridSpec(2, 1, hspace=0.05, height_ratios=[1, 0.5])
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])

        return axis00, axis10, x, y, y2, area


def create_axes_projections(res=res, box_size=box_size, contour=False, colorbar=False, velocity_vectors=False,
                            multiple=False, multiple2=False, multiple3=False, multiple4=False):
    """
    Generate plot axes.
    :param res: resolution
    :param box_size: box_size
    :param contour: contour
    :param colorbar: colorbar
    :param velocity_vectors: velocity_vectors
    :param multiple: multiple
    :return: axes
    """
    print('Invoking create_axes_combinations from plot_utilities.py')

    # Set the axis values #
    area = (box_size / res) ** 2  # Calculate the area.
    x = np.linspace(-0.5 * box_size, +0.5 * box_size, res + 1)
    y = np.linspace(-0.5 * box_size, +0.5 * box_size, res + 1)
    y2 = np.linspace(-0.5 * box_size, +0.5 * box_size, res / 2 + 1)

    # Generate the panels #
    if contour is True:
        gs = gridspec.GridSpec(2, 3, hspace=0.05, wspace=0.0, height_ratios=[1, 0.5], width_ratios=[1, 1, 0.05])
        axis00, axis01 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])
        axis10, axis11 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])
        axiscbar = plt.subplot(gs[:, 2])

        return axis00, axis01, axis10, axis11, axiscbar, x, y, y2, area

    elif colorbar is True:
        gs = gridspec.GridSpec(2, 2, hspace=0.05, wspace=0.0, height_ratios=[1, 0.5], width_ratios=[1, 0.05])
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])
        axiscbar = plt.subplot(gs[:, 1])

        return axis00, axis10, axiscbar, x, y, y2, area

    elif velocity_vectors is True:
        gs = gridspec.GridSpec(2, 1, hspace=0.05)
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])

        return axis00, axis10, x, y, y2, area

    elif multiple is True:
        gs = gridspec.GridSpec(3, 6, hspace=0.0, wspace=0.05, height_ratios=[1, 0.05, 1])
        axis00, axis01, axis02, axis03, axis04, axis05 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(
            gs[0, 2]), plt.subplot(gs[0, 3]), plt.subplot(gs[0, 4]), plt.subplot(gs[0, 5])
        axis10, axis11, axis12, axis13, axis14, axis15 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(
            gs[1, 2]), plt.subplot(gs[1, 3]), plt.subplot(gs[1, 4]), plt.subplot(gs[1, 5])
        axis20, axis21, axis22, axis23, axis24, axis25 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(
            gs[2, 2]), plt.subplot(gs[2, 3]), plt.subplot(gs[2, 4]), plt.subplot(gs[2, 5])

        return axis00, axis10, axis20, axis01, axis11, axis21, axis02, axis12, axis22, axis03, axis13, axis23, \
               axis04, axis14, axis24, axis05, axis15, axis25, x, y, area

    elif multiple2 is True:
        gs = gridspec.GridSpec(4, 3, hspace=0, wspace=0, height_ratios=[1, 0.5, 1, 0.5])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 2]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axis30, axis31, axis32 = plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2])

        return axis00, axis10, axis20, axis30, axis01, axis11, axis21, axis31, axis02, axis12, axis22, axis32, x, y, \
               y2, area

    elif multiple3 is True:
        gs = gridspec.GridSpec(4, 4, hspace=0.05, wspace=0, height_ratios=[1, 0.5, 1, 0.5], width_ratios=[1, 1, 1, 0.1])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axis30, axis31, axis32 = plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2])
        axiscbar = plt.subplot(gs[:, 3])

        return axis00, axis10, axis20, axis30, axis01, axis11, axis21, axis31, axis02, axis12, axis22, axis32, \
               axiscbar, x, y, y2, area

    if multiple4 is True:
        gs = gridspec.GridSpec(2, 3, hspace=0, wspace=0)
        axis00, axis01 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])
        axis10, axis11 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])

        return axis00, axis01, axis10, axis11, x, y, y2, area


    else:
        gs = gridspec.GridSpec(2, 1, hspace=0.05)
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])

        return axis00, axis10, x, y, y2, area
