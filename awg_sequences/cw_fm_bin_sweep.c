/*
 * Performs a square wave AM modulated CW ODMR frequency sweep.
 * Includes marker for bins.
 *
 * Square AM modulation done by first sending a pulse identical to cw_sweep.c, but afterwards also sending a zero pulse
 * representing the 0 amplitude part of the modulation.
 *
 * In contrast to cw_sweep.c, multiple measurements must be taken instead of a long pulse, as it should alternate
 * between sine/zero pulses.
 *
 * Required constants on Sequence property constants:
 *  - BIN_LENGTH    - number of samples per bin
 *  - N_BINS        - number of bins to output high/low each for
 *  - OSC1          - the first oscillator to sweep
 *  - OSC1          - the second oscillator to sweep
 *  - START_FREQ    - starting frequency relative to center
 *  - FREQ_DEV      - the FM frequency deviation
 *  - FREQ_INCR     - sweep frequency increment amount
 *  - N_SWEEP       - number of sweep steps to perform
 *  - N_MEAS        - number of measurements to perform at each frequency
 */

wave w = ones(BIN_LENGTH);

wave m1 = marker(16, 1);
wave m2 = marker(BIN_LENGTH - 16, 0);
wave m = join(m1, m2);

assignWaveIndex(1, 2, w + m, 0);

configFreqSweep(OSC1, START_FREQ - FREQ_DEV, FREQ_INCR);
configFreqSweep(OSC2, START_FREQ + FREQ_DEV, FREQ_INCR);

var i;
for (i = 0; i < N_SWEEP; i++) {
    setSweepStep(OSC, i);

    resetOscPhase();

    repeat (N_MEAS) {
        repeat(N_BINS) {
            executeTableEntry(0);
        }

        repeat(N_BINS) {
            executeTableEntry(1);
        }
    }

    waitWave();
}

// Finish off with marker reset for TT
executeTableEntry(0);