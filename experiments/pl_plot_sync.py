import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from TimeTagger import CountBetweenMarkers, createTimeTaggerNetwork, CHANNEL_UNUSED
import time

TT_CLICK_CHANNEL = 1
TT_MARKER_CHANNEL = 2

# Time Tagger initialization
tt = createTimeTaggerNetwork('localhost:41101')

tt.setTriggerLevel(TT_MARKER_CHANNEL, 0.5)
tt.setTriggerLevel(TT_CLICK_CHANNEL, 0.25)

cbm = CountBetweenMarkers(tt, TT_CLICK_CHANNEL, TT_MARKER_CHANNEL, CHANNEL_UNUSED, 3*160)

cbm.start();

while not cbm.ready():
    time.sleep(0.2)

data = cbm.getData()

plt.plot(data)
plt.show()

# laser:
# wave h = marker(1024, 1);
# wave l = marker(1024, 0);

# const PERIOD = 1;

# var i;
# while (true) {
#     playWave(h);
#     playHold(2000000000);
#     waitWave();
    
#     playWave(l);
#     playHold(2000000000);
#     waitWave();
# }

# MW:
# wave m = marker(16, 1);

# var i;
# while (true) {
#   playWave(m);
#   wait(2500000);
# }