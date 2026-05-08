 /*
 * Required constants on Sequence property constants:
 * LASER_ON             - Time laser is on 
 * SHORTEST_PERIOD      - Period length with shortest pulse duration 
 * OSC                  - Oscillator being swept
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

laser_off=SHORTEST_PERIOD - LASER_ON;

for (i = 0; i < N_SWEEP; i++) { 
    resetOscPhase();

    repeat(N_MEAS) {
        executeTableEntry(0);
        executeTableEntry(2);
        waitWave();
        executeTableEntry(1);
        playHold(laser_off-100)
        waitWave()
    }

    laser_off+= PULSE_INCR;
}





