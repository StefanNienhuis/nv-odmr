/*
 * Microwave part of the pulsed sweep.
 *
 * Required constants on Sequence property constants:
 *  - INIT_LENGTH   - number of samples to enable laser for
 *  - DARK_LENGTH   - number of samples to delay between laser and MW
 *  - MW_LENGTH     - number of samples to output microwave for
 *  - MEAS_LENGTH   - number of samples to enable measurement for
 *  - OSC           - oscillator to sweep
 *  - START_FREQ    - start MW frequency
 *  - FREQ_INCR     - increment MW frequency each step
 *  - N_SWEEP       - number of sweep steps to perform
 *  - N_MEAS        - number of measurements to perform at each tau
 */

wave w_init = zeros(INIT_LENGTH + DARK_LENGTH);
wave w_mw = ones(MW_LENGTH);
wave w_meas = zeros(MEAS_LENGTH);

wave m_ref0 = marker(INIT_LENGTH - MEAS_LENGTH, 0);
wave m_ref1 = marker(MEAS_LENGTH, 1);
wave m_ref2 = marker(DARK_LENGTH, 0);
wave m_ref = join(m_ref0, m_ref1, m_ref2);

wave m_meas = marker(MEAS_LENGTH, 1);

assignWaveIndex(1, 2, w_init + m_ref, 0);
assignWaveIndex(1, 2, w_mw, 1);
assignWaveIndex(1, 2, w_meas + m_meas, 2);

configFreqSweep(OSC, START_FREQ, FREQ_INCR);

var i;
for (i = 0; i < N_SWEEP; i++) {
    setSweepStep(OSC, i);

    resetOscPhase();

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
}
