/*
 * Microwave part of the rabi sweep.
 *
 * Required constants on Sequence property constants:
 *  - INIT_LENGTH       - number of samples to enable laser for during init
 *  - READOUT_LENGTH    - number of samples to enable laser for during readout
 *  - MEAS_LENGTH       - number of samples to enable measurement for
 *  - REF_LENGTH        - number of samples to enable reference measurement for
 *  - START_DARK        - start dark length
 *  - DARK_INCR         - increment dark length each step
 *  - N_SWEEP           - number of sweep steps to perform
 *  - N_MEAS            - number of measurements to perform at each tau
 */

wave w_init = zeros(INIT_LENGTH);

wave m_readout0 = marker(MEAS_LENGTH, 1);
wave m_readout1 = marker(READOUT_LENGTH - MEAS_LENGTH - REF_LENGTH, 0);
wave m_readout2 = marker(REF_LENGTH, 1);
wave m_readout = join(m_readout0, m_readout1, m_readout2);

assignWaveIndex(1, 2, w_init, 0);
assignWaveIndex(1, 2, m_readout, 1);

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
