/*
 * Laser part of the rabi sweep.
 *
 * Required constants on Sequence property constants:
 *  - INIT_LENGTH       - number of samples to enable laser for during initialization
 *  - DARK_LENGTH       - number of samples to delay between laser and MW
 *  - READOUT_LENGTH    - number of samples to enable laser for during readout
 *  - START_TAU         - start MW pulse length
 *  - TAU_INCR          - increment MW pulse length each step
 *  - N_SWEEP           - number of sweep steps to perform
 *  - N_MEAS            - number of measurements to perform at each tau
 */

wave init = marker(INIT_LENGTH, 1);
wave dark = marker(DARK_LENGTH, 0);
wave meas = marker(READOUT_LENGTH, 1);

assignWaveIndex(1, 2, init, 0);
assignWaveIndex(1, 2, dark, 1);
assignWaveIndex(1, 2, meas, 2);

var i;
for (i = 0; i < N_SWEEP; i++) {
    var tau = START_TAU + (i * TAU_INCR);

    repeat (N_MEAS) {
        waitDigTrigger(1);

        // Init
        executeTableEntry(0);

        // Dark
        executeTableEntry(1);

        // Hold dark (0) during MW
        playHold(tau);

        // Measure
        executeTableEntry(2);

        waitWave();
    }
}
