import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from TimeTagger import TimeTagStream, createTimeTaggerNetwork
import time

TT_CLICK_CHANNEL = 1
NUM_HISTORY = 100

# Time Tagger initialization
tt = createTimeTaggerNetwork('localhost:41101')

tt.setTriggerLevel(TT_CLICK_CHANNEL, 0.25)

stream = TimeTagStream(tt, 1e6, [TT_CLICK_CHANNEL])

matplotlib.use('TkAgg')

# Data storage
x_data = []
y_data = []

# Create figure and axis
fig, ax = plt.subplots()
line, = ax.plot([], [], marker='o')


# Initialize plot limits
ax.set_xlim(0, NUM_HISTORY)
ax.set_ylim(0,1)

last_meas = datetime.now()

def update(frame):
    global last_meas
    x_data.append(len(x_data))
    
    # now = datetime.now()
    # diff_t = (now - last_meas).total_seconds()
    # last_meas = now
    # print(diff_t)
    
    data = stream.getData()
    diff_t = (data.tGetData - data.tStart) / 1e12
    if diff_t !=0:
        rate = data.size / diff_t
        
        y_data.append(rate)

    x_data_trim = x_data[-NUM_HISTORY:]
    y_data_trim = y_data[-NUM_HISTORY:]

    line.set_data(x_data_trim, y_data_trim)

    ax.set_xlim(max(0, len(x_data) - NUM_HISTORY), len(x_data))
    ax.set_ylim(0, max(y_data_trim) * 1.1)

    return line,

ani = animation.FuncAnimation(fig, update, interval=10)

plt.show()