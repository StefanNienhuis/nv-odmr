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

wave s = sine(PULSE_LENGTH, 1, 0, 1);
wave z = zeros(PULSE_LENGTH);

wave m1 = marker(MEAS_DELAY, 0);
wave m2 = marker(PULSE_LENGTH - MEAS_DELAY, 1);
wave m = join(m1, m2);

wave sm = s + m;
wave zm = z + m;

configFreqSweep(OSC, START_FREQ, FREQ_INCR);

var i;
for (i = 0; i < N_SWEEP; i++) {
    setSweepStep(OSC, i);

    resetOscPhase();

    repeat (N_MEAS) {
        playWave(1, sm);
        playWave(1, zm);
    }

    // Wait until completion to not setSweepStep during waveform
    waitWave();
}