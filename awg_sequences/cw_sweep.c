/*
 * Performs a CW ODMR frequency sweep.
 * Includes marker that stays high during the second half of the pulse, to allow system to stabilize.
 *
 * In general multiple measurements per frequency (through N_MEAS) should not be needed. One longer pulse is preferred.
 *
 * Required constants on Sequence property constants:
 *  - PULSE_LENGTH  - number of samples to output the pulse for
 *  - MEAS_DELAY    - number of delay samples before measurement marker is set high
 *  - OSC           - oscillator to sweep
 *  - START_FREQ    - starting frequency relative to center
 *  - FREQ_INCR     - sweep frequency increment amount
 *  - N_SWEEP       - number of sweep steps to perform
 *  - N_MEAS        - number of measurements to perform at each frequency
 */

wave w = sine(PULSE_LENGTH, 1, 0, 1);

wave m1 = marker(MEAS_DELAY, 0);
wave m2 = marker(PULSE_LENGTH - MEAS_DELAY, 1);
wave m = join(m1, m2);

wave wm = w + m;

configFreqSweep(OSC, START_FREQ, FREQ_INCR);

var s;
for (s = 0; s < N_SWEEP; s++) {
    setSweepStep(OSC, s);

    resetOscPhase();

    repeat (N_MEAS) {
        playWave(1, wm);
    }

    // Wait until completion to not setSweepStep during waveform
    waitWave();
}