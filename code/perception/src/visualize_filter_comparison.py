#!/usr/bin/env python

import matplotlib.pyplot as plt
from math import pi
import numpy as np

NEW_FILTER_FILE_NAME = "02"
OLD_FILTER_FILE_NAME = "02"
GT_FILE_NAME = "02"


# open the file with the estimated positions of the new filter
nf_pos_file = open(
    "/workspace/code/perception/src/experiments/filter_comparison/"
    + "new_filter_pos/data_"
    + NEW_FILTER_FILE_NAME
    + ".csv",
    mode="r",
)
# throw away the first line (column names)
nf_pos_first_line = nf_pos_file.readline()

# open the file with the estimated headings of the new filter
nf_heading_file = open(
    "/workspace/code/perception/src/experiments/filter_comparison/"
    + "new_filter_heading/data_"
    + NEW_FILTER_FILE_NAME
    + ".csv",
    mode="r",
)
# throw away the first line (column names)
nf_heading_first_line = nf_heading_file.readline()

# open the file with the estimated positions of the new filter
of_pos_file = open(
    "/workspace/code/perception/src/experiments/filter_comparison/"
    + "old_filter_pos/data_"
    + OLD_FILTER_FILE_NAME
    + ".csv",
    mode="r",
)
# throw away the first line (column names)
of_pos_first_line = of_pos_file.readline()

# open the file with the estimated positions of the new filter
of_heading_file = open(
    "/workspace/code/perception/src/experiments/filter_comparison/"
    + "old_filter_heading/data_"
    + OLD_FILTER_FILE_NAME
    + ".csv",
    mode="r",
)
# throw away the first line (column names)
of_heading_first_line = of_heading_file.readline()

# open the ground truth file
gt_file = open(
    "/workspace/code/perception/src/experiments/filter_comparison/"
    + "ground_truth/data_"
    + GT_FILE_NAME
    + ".csv",
    mode="r",
)
# throw away the first line (column names)
gt_first_line = gt_file.readline()


# put all data into a 2-dimensional list of floats
nf_pos_lines = []
while True:
    next_line = nf_pos_file.readline()
    if len(next_line) == 0:
        break
    next_line = next_line.split(",")
    next_line[-1] = next_line[-1].strip()
    new_line = []
    for nr in next_line:
        new_line.append(float(nr))
    nf_pos_lines.append(new_line)

# put all data into a 2-dimensional list of floats
nf_heading_lines = []
while True:
    next_line = nf_heading_file.readline()
    if len(next_line) == 0:
        break
    next_line = next_line.split(",")
    next_line[-1] = next_line[-1].strip()
    new_line = []
    for nr in next_line:
        new_line.append(float(nr))
    nf_heading_lines.append(new_line)

# put all data into a 2-dimensional list of floats
of_pos_lines = []
while True:
    next_line = of_pos_file.readline()
    if len(next_line) == 0:
        break
    next_line = next_line.split(",")
    next_line[-1] = next_line[-1].strip()
    new_line = []
    for nr in next_line:
        new_line.append(float(nr))
    of_pos_lines.append(new_line)

# put all data into a 2-dimensional list of floats
of_heading_lines = []
while True:
    next_line = of_heading_file.readline()
    if len(next_line) == 0:
        break
    next_line = next_line.split(",")
    next_line[-1] = next_line[-1].strip()
    new_line = []
    for nr in next_line:
        new_line.append(float(nr))
    of_heading_lines.append(new_line)

# put all data into a 2-dimensional list of floats
gt_lines = []
while True:
    next_line = gt_file.readline()
    if len(next_line) == 0:
        break
    next_line = next_line.split(",")
    next_line[-1] = next_line[-1].strip()
    new_line = []
    for nr in next_line:
        new_line.append(float(nr))
    gt_lines.append(new_line)


# times

nf_pos_time_stamps = []
for line in nf_pos_lines:
    nf_pos_time_stamps.append(line[0])

nf_heading_time_stamps = []
for line in nf_heading_lines:
    nf_heading_time_stamps.append(line[0])

of_pos_time_stamps = []
for line in of_pos_lines:
    of_pos_time_stamps.append(line[0])

of_heading_time_stamps = []
for line in of_heading_lines:
    of_heading_time_stamps.append(line[0])

gt_time_stamps = []
for line in gt_lines:
    gt_time_stamps.append(line[0])


# x positions

nf_x_positions = []
for line in nf_pos_lines:
    nf_x_positions.append(line[1])

of_x_positions = []
for line in of_pos_lines:
    of_x_positions.append(line[1])

gt_x_positions = []
for line in gt_lines:
    gt_x_positions.append(line[1])

# y positions

nf_y_positions = []
for line in nf_pos_lines:
    nf_y_positions.append(line[2])

of_y_positions = []
for line in of_pos_lines:
    of_y_positions.append(line[2])

gt_y_positions = []
for line in gt_lines:
    gt_y_positions.append(line[2])

# z positions

nf_z_positions = []
for line in nf_pos_lines:
    nf_z_positions.append(line[3])

of_z_positions = []
for line in of_pos_lines:
    of_z_positions.append(line[3])

gt_z_positions = []
for line in gt_lines:
    gt_z_positions.append(line[3])

# headings

nf_headings = []
for line in nf_heading_lines:
    nf_headings.append(line[1])

of_headings = []
for line in of_heading_lines:
    of_headings.append(line[1])

gt_headings = []
for line in gt_lines:
    heading_in_degrees = line[4]
    heading_in_rad = heading_in_degrees * pi / 180.0
    gt_headings.append(heading_in_rad)


def plot_x_position():
    plt.subplot(321)
    plt.title("Filtervergleich X-Position")
    plt.plot(
        nf_pos_time_stamps,
        nf_x_positions,
        label="nf x positions " + NEW_FILTER_FILE_NAME,
        color="blue",
    )
    plt.plot(
        of_pos_time_stamps,
        of_x_positions,
        label="of x positions " + OLD_FILTER_FILE_NAME,
        color="orange",
    )
    plt.plot(
        gt_time_stamps,
        gt_x_positions,
        label="gt x positions " + GT_FILE_NAME,
        color="green",
    )

    upper_limit = []
    lower_limit = []
    for i in range(len(gt_time_stamps)):
        upper_limit.append(gt_x_positions[i] + 0.5)
        lower_limit.append(gt_x_positions[i] - 0.5)
    plt.plot(
        gt_time_stamps,
        upper_limit,
        color="red",
        linestyle="dashed",
        label="upper limit",
    )
    plt.plot(
        gt_time_stamps,
        lower_limit,
        color="red",
        linestyle="dashed",
        label="lower limit",
    )
    plt.legend()

    plt.subplot(323)
    plt.title("Fehler X-Position")
    gt_x_small = []
    index = 0
    for j in range(len(nf_x_positions)):

        for i in range(0, len(gt_time_stamps)):
            if gt_time_stamps[i] == nf_pos_time_stamps[j]:
                index = i
                break
        gt_x_small.append(gt_x_positions[index])

    diff = np.abs(np.subtract(gt_x_small, nf_x_positions))
    print("MSE NF X-Pos: " + str(np.square(diff.mean())))
    plt.plot(nf_pos_time_stamps, diff, color="blue", label="Error NF X-Pos")

    gt_x_small = []
    index = 0
    for j in range(len(of_x_positions)):
        for i in range(0, len(gt_time_stamps)):
            if gt_time_stamps[i] == of_pos_time_stamps[j]:
                index = i
                break
        gt_x_small.append(gt_x_positions[index])
    diff = np.abs(np.subtract(gt_x_small, of_x_positions))
    print("MSE OF X-Pos: " + str(np.square(diff.mean())))
    plt.plot(of_pos_time_stamps, diff, color="orange", label="Error OF X-Pos")

    limit = []
    for i in range(0, len(nf_pos_time_stamps)):
        limit.append(0.5)
    plt.plot(
        nf_pos_time_stamps,
        limit,
        color="red",
        linestyle="dashed",
        label="Accepted error",
    )
    plt.legend()

    plt.subplot(325)
    plt.title("Heading GT")
    plt.plot(gt_time_stamps, gt_headings, label="gt heading " + GT_FILE_NAME)
    plt.legend()
    plot_y_position()
    plt.show()


def plot_x_error(subplotId):
    plt.subplot(subplotId)
    plt.title("Fehler X-Position")
    gt_x_small = []
    index = 0
    for j in range(len(nf_x_positions)):
        for i in range(0, len(gt_time_stamps)):
            if gt_time_stamps[i] == nf_pos_time_stamps[j]:
                index = i
                break
        gt_x_small.append(gt_x_positions[index])

    diff = np.abs(np.subtract(gt_x_small, nf_x_positions))
    print("MSE NF X-Pos: " + str(np.square(diff.mean())))
    plt.plot(nf_pos_time_stamps, diff, color="blue", label="Error NF X-Pos")

    gt_x_small = []
    index = 0
    for j in range(len(of_x_positions)):
        for i in range(0, len(gt_time_stamps)):
            if gt_time_stamps[i] == of_pos_time_stamps[j]:
                index = i
                break
        gt_x_small.append(gt_x_positions[index])
    diff = np.abs(np.subtract(gt_x_small, of_x_positions))
    print("MSE OF X-Pos: " + str(np.square(diff.mean())))
    plt.plot(of_pos_time_stamps, diff, color="orange", label="Error OF X-Pos")

    limit = []
    for i in range(0, len(nf_pos_time_stamps)):
        limit.append(0.5)
    plt.plot(
        nf_pos_time_stamps,
        limit,
        color="red",
        linestyle="dashed",
        label="Accepted error",
    )
    plt.legend()


def plot_gt_heading(subplotId):
    plt.subplot(subplotId)
    plt.title("Heading GT")
    plt.plot(gt_time_stamps, gt_headings, label="gt heading " + GT_FILE_NAME)
    plt.legend()


def plot_y_position():
    plt.subplot(322)
    plt.title("Filtervergleich Y-Position")
    plt.plot(
        nf_pos_time_stamps,
        nf_y_positions,
        label="nf y positions " + NEW_FILTER_FILE_NAME,
        color="blue",
    )
    plt.plot(
        of_pos_time_stamps,
        of_y_positions,
        label="of y positions " + OLD_FILTER_FILE_NAME,
        color="orange",
    )
    plt.plot(
        gt_time_stamps,
        gt_y_positions,
        label="gt y positions " + GT_FILE_NAME,
        color="green",
    )

    upper_limit = []
    lower_limit = []
    for i in range(len(gt_time_stamps)):
        upper_limit.append(gt_y_positions[i] + 0.5)
        lower_limit.append(gt_y_positions[i] - 0.5)
    plt.plot(
        gt_time_stamps,
        upper_limit,
        color="red",
        linestyle="dashed",
        label="upper limit",
    )
    plt.plot(
        gt_time_stamps,
        lower_limit,
        color="red",
        linestyle="dashed",
        label="lower limit",
    )
    plt.legend()
    plt.subplot(324)
    plt.title("Fehler Y-Position")
    gt_y_small = []
    index = 0
    for j in range(len(nf_y_positions)):

        for i in range(0, len(gt_time_stamps)):
            if gt_time_stamps[i] == nf_pos_time_stamps[j]:
                index = i
                break
        gt_y_small.append(gt_y_positions[index])

    diff = np.abs(np.subtract(gt_y_small, nf_y_positions))
    print("MSE NF Y-Pos: " + str(np.square(diff.mean())))
    plt.plot(nf_pos_time_stamps, diff, color="blue", label="Error NF Y-Pos")

    gt_y_small = []
    index = 0
    for j in range(len(of_y_positions)):
        for i in range(0, len(gt_time_stamps)):
            if gt_time_stamps[i] == of_pos_time_stamps[j]:
                index = i
                break
        gt_y_small.append(gt_y_positions[index])
    diff = np.abs(np.subtract(gt_y_small, of_y_positions))
    print("MSE OF Y-Pos: " + str(np.square(diff.mean())))
    plt.plot(of_pos_time_stamps, diff, color="orange", label="Error OF Y-Pos")

    limit = []
    for i in range(0, len(nf_pos_time_stamps)):
        limit.append(0.5)
    plt.plot(
        nf_pos_time_stamps,
        limit,
        color="red",
        linestyle="dashed",
        label="Accepted error",
    )
    plt.legend()

    plt.subplot(326)
    plt.title("Heading GT")
    plt.plot(gt_time_stamps, gt_headings, label="gt heading " + GT_FILE_NAME)
    plt.legend()
    plt.show()

    upper_limit = []
    lower_limit = []
    for i in range(len(gt_time_stamps)):
        upper_limit.append(gt_y_positions[i] + 0.5)
        lower_limit.append(gt_y_positions[i] - 0.5)
    plt.plot(
        gt_time_stamps,
        upper_limit,
        color="red",
        linestyle="dashed",
        label="upper limit",
    )
    plt.plot(
        gt_time_stamps,
        lower_limit,
        color="red",
        linestyle="dashed",
        label="lower limit",
    )
    plt.legend()


def plot_y_error(subplotId):
    plt.subplot(subplotId)
    plt.title("Fehler Y-Position")
    gt_y_small = []
    index = 0
    for j in range(len(nf_y_positions)):
        for i in range(0, len(gt_time_stamps)):
            if gt_time_stamps[i] == nf_pos_time_stamps[j]:
                index = i
                break
        gt_y_small.append(gt_y_positions[index])

    diff = np.abs(np.subtract(gt_y_small, nf_y_positions))
    print("MSE NF Y-Pos: " + str(np.square(diff.mean())))
    plt.plot(nf_pos_time_stamps, diff, color="blue", label="Error NF Y-Pos")

    gt_y_small = []
    index = 0
    for j in range(len(of_y_positions)):
        for i in range(0, len(gt_time_stamps)):
            if gt_time_stamps[i] == of_pos_time_stamps[j]:
                index = i
                break
        gt_y_small.append(gt_y_positions[index])
    diff = np.abs(np.subtract(gt_y_small, of_y_positions))
    print("MSE OF Y-Pos: " + str(np.square(diff.mean())))
    plt.plot(of_pos_time_stamps, diff, color="orange", label="Error OF Y-Pos")

    limit = []
    for i in range(0, len(nf_pos_time_stamps)):
        limit.append(0.5)
    plt.plot(
        nf_pos_time_stamps,
        limit,
        color="red",
        linestyle="dashed",
        label="Accepted error",
    )
    plt.legend()


"""
def plot_z_position():
    plt.plot(
        nf_pos_time_stamps,
        nf_z_positions,
        label="nf z positions " + NEW_FILTER_FILE_NAME,
    )
    plt.plot(
        of_pos_time_stamps,
        of_z_positions,
        label="of z positions " + OLD_FILTER_FILE_NAME,
    )
    plt.plot(gt_time_stamps, gt_z_positions, label="gt z positions " + GT_FILE_NAME)
    plt.plot()
    plt.legend()


def plot_heading():
    plt.plot(
        nf_heading_time_stamps, nf_headings, label="nf heading " + NEW_FILTER_FILE_NAME
    )
    plt.plot(
        of_heading_time_stamps, of_headings, label="of heading " + OLD_FILTER_FILE_NAME
    )
    plt.plot(gt_time_stamps, gt_headings, label="gt heading " + GT_FILE_NAME)
    plt.plot()
    plt.legend()
"""


plot_x_position()
# plot_y_position()
# plot_z_position()
# plot_heading()
