/*
 * Microwave part of the rabi sweep.
 *
 * Required constants on Sequence property constants:
 *  - INIT_LENGTH       - number of samples to enable laser for during initialization
 *  - DARK_LENGTH       - number of samples to delay between laser and MW
 *  - READOUT_LENGTH    - number of samples to enable laser for during readout
 *  - MEAS_LENGTH       - number of samples to enable measurement for
 *  - REF_LENGTH        - number of samples to enable reference measurement for
 *  - START_TAU         - start MW pulse length
 *  - TAU_INCR          - increment MW pulse length each step
 *  - N_SWEEP           - number of sweep steps to perform
 *  - N_MEAS            - number of measurements to perform at each tau
 */

wave w_init_dark = zeros(INIT_LENGTH + DARK_LENGTH);
wave w_mw = ones(16);
wave w_readout = zeros(READOUT_LENGTH);

wave m_readout0 = marker(MEAS_LENGTH, 1);
wave m_readout1 = marker(READOUT_LENGTH - MEAS_LENGTH - REF_LENGTH, 0);
wave m_readout2 = marker(REF_LENGTH, 1);
wave m_readout = join(m_readout0, m_readout1, m_readout2);

assignWaveIndex(1, 2, w_init_dark, 0);
assignWaveIndex(1, 2, w_mw, 1);
assignWaveIndex(1, 2, w_readout + m_readout, 2);

var i;
repeat (N_MEAS) {
    for (i = 0; i < N_SWEEP; i++) {
        var tau_hold = START_TAU + (i * TAU_INCR) - 16;

        if (tau_hold > 0) {
            // tau >= 16 -> play wave once and hold for remainder
            waitDigTrigger(1);

            // Init + dark
            executeTableEntry(0);

            // MW
            executeTableEntry(1);
            playHold(tau_hold);

            // Measure
            executeTableEntry(2);

            waitWave();
        } else if (tau_hold > -16) {
            // 0 <= tau <= 16 -> play wave once
            waitDigTrigger(1);

            // Init + dark
            executeTableEntry(0);

            // MW
            executeTableEntry(1);
            playHold(0);

            // Measure
            executeTableEntry(2);

            waitWave();
        } else {
            // tau == 0 -> play nothing
            waitDigTrigger(1);

            // Init + dark
            executeTableEntry(0);

            // No MW
            playHold(0);
            playHold(0);

            // Measure
            executeTableEntry(2);

            waitWave();
        }
    }
}
