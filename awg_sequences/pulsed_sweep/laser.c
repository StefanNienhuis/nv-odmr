/*
 * Laser part of the pulsed sweep.
 *
 * Required constants on Sequence property constants:
 *  - INIT_LENGTH   - number of samples to enable laser for
 *  - DARK_LENGTH   - number of samples to delay between laser and MW
 *  - MW_LENGTH     - number of samples to output microwave for
 *  - MEAS_LENGTH   - number of samples to enable measurement for
 *  - N_SWEEP       - number of sweep steps to perform
 *  - N_MEAS        - number of measurements to perform at each tau
 */

wave init = marker(INIT_LENGTH, 1);
wave dark = marker(DARK_LENGTH + MW_LENGTH, 0);
wave meas = marker(MEAS_LENGTH, 1);

assignWaveIndex(1, 2, init, 0);
assignWaveIndex(1, 2, mw, 1);
assignWaveIndex(1, 2, meas, 2);

var i;
for (i = 0; i < N_SWEEP; i++) {
    var tau = START_TAU + (i * TAU_INCR);

    repeat (N_MEAS) {
        waitDigTrigger(1);

        // Init
        executeTableEntry(0);

        // Dark + MW
        executeTableEntry(1);

        // Measure
        executeTableEntry(2);

        waitWave();
    }
}
