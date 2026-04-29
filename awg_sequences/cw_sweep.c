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
wave m11 = marker(MEAS_DELAY, 0);

wave w2 = ones(16);
wave m2 = marker(16, 1);

wave wm1 = w1 + m1;
wave wm2 = w2 + m2;

assignWaveIndex(0, wm1);
assignWaveIndex(1, wm2);

// Sample count to hold after the meas delay, minus 16 as it's triggered with playHold from a previous.
const PULSE_HOLD = PULSE_LENGTH - MEAS_DELAY - 16;

configFreqSweep(OSC, START_FREQ, FREQ_INCR);

var i;
for (i = 0; i < N_SWEEP; i++) {
    setSweepStep(OSC, i);

    resetOscPhase();

    executeTableEntry(0);
    executeTableEntry(1);
    playHold(PULSE_HOLD);

    // Wait until completion to not setSweepStep during waveform
    waitWave();
}