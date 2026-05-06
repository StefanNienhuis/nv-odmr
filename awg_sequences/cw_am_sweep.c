/*
 * Performs a square wave AM modulated CW ODMR frequency sweep.
 * Includes marker that stays high during the second half of the pulse, to allow system to stabilize.
 *
 * Square AM modulation done by first sending a pulse identical to cw_sweep.c, but afterwards also sending a zero pulse
 * representing the 0 amplitude part of the modulation.
 *
 * In contrast to cw_sweep.c, multiple measurements must be taken instead of a long pulse, as it should alternate
 * between sine/zero pulses.
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

wave wh1 = ones(MEAS_DELAY);
wave wh2 = ones(1024);

wave wl1 = zeros(MEAS_DELAY);
wave wl2 = zeros(1024);

wave m1 = marker(MEAS_DELAY, 1);
wave m2 = marker(1024, 0);

wave wh1m = wh1 + m1;
wave wh2m = wh2 + m2;
wave wl1m = wl1 + m1;
wave wl2m = wl2 + m2;

assignWaveIndex(1, 2, wh1m, 0);
assignWaveIndex(1, 2, wh2m, 1);
assignWaveIndex(1, 2, wl1m, 2);
assignWaveIndex(1, 2, wl2m, 3);

configFreqSweep(OSC, START_FREQ, FREQ_INCR);

var i;
for (i = 0; i < N_SWEEP; i++) {
    setSweepStep(OSC, i);

    resetOscPhase();

    repeat (N_MEAS) {
        executeTableEntry(0);
        waitWave();
        executeTableEntry(1);
        executeTableEntry(4);
        waitWave();

        executeTableEntry(2);
        waitWave();
        executeTableEntry(3);
        executeTableEntry(4);
        waitWave();
    }
}

// Finish off with marker reset for TT
executeTableEntry(2);