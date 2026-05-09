 /*
To perform a rabi measurement the pulse length of the microwave drive should be drive for multiple mw pulse lengths.

 This is done by first exiciting the electrons with a laser pulse. After the laser a wait duration for the nv-centers to stabilize.
 This is followed by the mw drive for a specific pulse duration. Another wait duration for the nv-center to stabilize followed by a measurment period.
 To increase SNR multiple measurements per pulse duration are used.

 Since the marker channel is used to synchronize both the timetagger and laser 2 channels need to be used to send marker signals to both devices.

 * Required constants on Sequence property constants:
 * LASER_ON             - Time laser is on 
 * SHORTEST_PERIOD      - Period length with shortest pulse duration s
 * PULSE_INCR           - Pulse increment for every n_sweep step
 * N_SWEEP              - Number of sweep steps
 * N_MEAS               - Number of measurements at each time delay

*/


wave w1 = zeros(100);
wave m1 = marker(100, 0);
wave laser_on = w1 + m1;

wave w1 = zeros(100);
wave m2 = marker(100, 1);
wave laser_off = w1 + m2;

assignWaveIndex(1, 2, laser_on, 0);
assignWaveIndex(1, 2, laser_off, 1);

curr_laser_off=SHORTEST_PERIOD - LASER_ON;

for (i = 0; i < N_SWEEP; i++) { 
    resetOscPhase();

    repeat(N_MEAS) {
        executeTableEntry(0);
        executeTableEntry(2);
        waitWave();
        executeTableEntry(1);
        playHold(curr_laser_off-100)
        waitWave()
    }

    curr_laser_off += PULSE_INCR;
}





