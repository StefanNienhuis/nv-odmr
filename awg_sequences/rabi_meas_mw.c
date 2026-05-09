 /*
 * Required constants on Sequence property constants:
 * SHORTEST_PULSE       - Shortest mw pulse duration
 * PULSE_INCR           - Pulse increment for every n_sweep step
 * N_SWEEP              - Number of sweep steps
 * N_MEAS               - Number of measurements at each time delay

*/
wave w_until_mw = zeros(100);
wave m_until_mw = marker(100, 1);
wave until_mw = w_until_mw + m_until_mw;

wave w_mw_on = ones(100);
wave m_mw_on = marker(100,1);
wave mw_on = m_mw_on+ w_mw_on;

wave w_meas_delay = zeros(100);
wave m_meas_delay = marker(100, 1);
wave meas_delay = w_meas_delay + m_meas_delay;

wave w_meas = zeros(100);
wave m_meas = marker (100,0);
wave meas = w_meas + m_meas;


assignWaveIndex(1, 2, until_mw, 0);
assignWaveIndex(1, 2, mw_on, 1);
assignWaveIndex(1, 2, meas_delay, 2);
assignWaveIndex(1, 2,  meas, 3);

curr_pulse = SHORTEST_PULSE;
for (i = 0; i < N_SWEEP; i++) { 
    resetOscPhase();

    repeat(N_MEAS) {
        executeTableEntry(0);
        executeTableEntry(1);
        waitWave();

        executeTableEntry(2);
        playHold(curr_pulse-100);
        waitWave();

        executeTableEntry(3);
        extecuteTableEntry(4);
        waitWave();

        executeTableEntry(5);
        extecuteTableEntry(6);
        waitWave()
    }

    curr_pulse+= PULSE_INCR;
}