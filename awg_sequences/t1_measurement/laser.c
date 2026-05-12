/*
 * Laser part of the T1 measurement.
 *
 * Required constants on Sequence property constants:
 *  - INIT_LENGTH       - number of samples to enable laser for during init
 *  - READOUT_LENGTH    - number of samples to enable laser for during readout
 *  - START_DARK        - start dark length
 *  - DARK_INCR         - increment dark length each step
 *  - N_SWEEP           - number of sweep steps to perform
 *  - N_MEAS            - number of measurements to perform at each tau
 */

wave init = marker(INIT_LENGTH, 1);
wave readout = marker(READOUT_LENGTH, 1);

assignWaveIndex(1, 2, init, 0);
assignWaveIndex(1, 2, readout, 1);

var i;
for (i = 0; i < N_SWEEP; i++) {
    var dark = START_DARK + (i * DARK_INCR);

    repeat (N_MEAS) {
        waitDigTrigger(1);

        // Init
        executeTableEntry(0);

        // Dark
        playZero(dark);

        // Measure
        executeTableEntry(1);

        waitWave();
    }
}
