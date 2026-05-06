/*
 * Performs a CW ODMR frequency sweep.
 * Includes marker that stays low for MEAS_DELAY samples, to allow system to stabilize.
 *
 * Marker is only toggled in the first measurement (of N_MEAS), staying high afterwards since the pulse doesn't change.
 *
 * Required constants on Sequence property constants:
 *  - PULSE_LENGTH  - number of samples to output the pulse for - must be less than 49 kSa and a multiple of 16
 *  - MEAS_DELAY    - number of delay samples before marker is set high - must be less than PULSE_LENGTH and a multiple of 16
 *  - OSC           - oscillator to sweep
 *  - START_FREQ    - starting frequency relative to center
 *  - FREQ_INCR     - sweep frequency increment amount
 *  - N_SWEEP       - number of sweep steps to perform
 *  - N_MEAS        - number of measurements to perform at each frequency
 */

wave w1 = ones(MEAS_DELAY);
wave m1 = marker(MEAS_DELAY, 1);

wave w2 = ones(1024);

wave wm1 = w1 + m1;

assignWaveIndex(1, 2, wm1, 0);
assignWaveIndex(1, 2, w2, 1);

configFreqSweep(OSC, START_FREQ, FREQ_INCR);

var i;
for (i = 0; i < N_SWEEP; i++) {
    setSweepStep(OSC, i);

    resetOscPhase();

    executeTableEntry(0);
    waitWave();

    repeat (N_MEAS) {
        executeTableEntry(1);
        executeTableEntry(2);
        waitWave();
    }
}

// Finish off with marker reset for TT
executeTableEntry(0);