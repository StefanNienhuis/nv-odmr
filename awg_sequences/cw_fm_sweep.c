/*
 * Performs a square wave FM modulated CW ODMR frequency sweep. The frequency will alternate between +- FREQ_DEV.
 * Includes marker that stays high during the second half of the pulse, to allow system to stabilize.
 *
 * Square FM modulation done by sending two pulses identical to cw_sweep.c, but switching between two oscillators with
 * different frequencies.
 *
 * In contrast to cw_sweep.c, multiple measurements must be taken instead of a long pulse, as it should alternate
 * between two frequencies.
 *
 * Required constants on Sequence property constants:
 *  - PULSE_LENGTH  - number of samples to output the pulse for
 *  - MEAS_DELAY    - number of delay samples before measurement marker is set high
 *  - OSC1          - the first oscillator to sweep
 *  - OSC2          - the second oscillator to sweep
 *  - START_FREQ    - starting frequency relative to center
 *  - FREQ_DEV      - the FM frequency deviation
 *  - FREQ_INCR     - sweep frequency increment amount
 *  - N_SWEEP       - number of sweep steps to perform
 *  - N_MEAS        - number of measurements to perform at each frequency
 */

wave w = ones(PULSE_LENGTH);

wave m1 = marker(MEAS_DELAY, 0);
wave m2 = marker(PULSE_LENGTH - MEAS_DELAY, 1);
wave m = join(m1, m2);

wave wm = w + m;

// Assign same wave to two indexes, oscillator selection done in Python
assignWaveIndex(0, wm);
assignWaveIndex(1, wm);

configFreqSweep(OSC1, START_FREQ - FREQ_DEV, FREQ_INCR);
configFreqSweep(OSC2, START_FREQ + FREQ_DEV, FREQ_INCR);

var i;
for (i = 0; i < N_SWEEP; i++) {
    setSweepStep(OSC1, i);
    setSweepStep(OSC2, i);

    resetOscPhase();

    repeat (N_MEAS) {
        executeTableEntry(0);
        executeTableEntry(1);
    }

    // Wait until completion to not setSweepStep during waveform
    waitWave();
}