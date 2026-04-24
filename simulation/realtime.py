# Real time plot test

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time

matplotlib.use('TkAgg')

# Data storage
x_data = []
y_data = []

# Create figure and axis
fig, ax = plt.subplots()
line, = ax.plot([], [], marker='o')

# Initialize plot limits
ax.set_xlim(0, 10)
ax.set_ylim(0, 1)

def update(frame):
    x_data.append(len(x_data))
    y_data.append(random.random())

    x_data_trim = x_data[-10:]
    y_data_trim = y_data[-10:]

    line.set_data(x_data_trim, y_data_trim)

    ax.set_xlim(max(0, len(x_data) - 10), len(x_data))

    return line,

ani = animation.FuncAnimation(fig, update, interval=1000)

plt.show()