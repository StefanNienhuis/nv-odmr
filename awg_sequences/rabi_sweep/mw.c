/*
 * Microwave part of the rabi sweep.
 *
 * Required constants on Sequence property constants:
 *  - INIT_LENGTH   - number of samples to enable laser for
 *  - DARK_LENGTH   - number of samples to delay between laser and MW
 *  - MEAS_LENGTH   - number of samples to enable measurement for
 *  - START_TAU     - start MW pulse length
 *  - TAU_INCR      - increment MW pulse length each step
 *  - N_SWEEP       - number of sweep steps to perform
 *  - N_MEAS        - number of measurements to perform at each tau
 */

wave w_init = zeros(INIT_LENGTH + DARK_LENGTH);
wave w_mw = ones(16);
wave w_meas = zeros(MEAS_LENGTH);

wave m_ref0 = marker(INIT_LENGTH - MEAS_LENGTH, 0);
wave m_ref1 = marker(MEAS_LENGTH, 1);
wave m_ref2 = marker(DARK_LENGTH, 0);
wave m_ref = join(m_ref0, m_ref1, m_ref2);

wave m_meas = marker(MEAS_LENGTH, 1);

assignWaveIndex(1, 2, w_init + m_ref, 0);
assignWaveIndex(1, 2, w_mw, 1);
assignWaveIndex(1, 2, w_meas + m_meas, 2);

var i;
for (i = 0; i < N_SWEEP; i++) {
    var tau_hold = START_TAU + (i * TAU_INCR) - 16;

    if (tau_hold >= 0) {
        // tau >= 16 -> play wave once and hold for remainder
        repeat (N_MEAS) {
            waitDigTrigger(1);

            // Init + dark
            executeTableEntry(0);

            // MW
            executeTableEntry(1);
            playHold(tau_hold);

            // Measure
            executeTableEntry(2);

            waitWave();
        }
    } else if (tau_hold >= -16) {
        // 0 <= tau <= 16 -> play wave once
        repeat (N_MEAS) {
            waitDigTrigger(1);

            // Init + dark
            executeTableEntry(0);

            // MW
            executeTableEntry(1);

            // Measure
            executeTableEntry(2);

            waitWave();
        }
    } else {
        // tau == 0 -> play nothing
        repeat (N_MEAS) {
            waitDigTrigger(1);

            // Init + dark
            executeTableEntry(0);

            // No MW

            // Measure
            executeTableEntry(2);

            waitWave();
        }
    }
}
