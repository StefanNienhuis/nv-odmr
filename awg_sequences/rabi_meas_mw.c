 /* 
 To perform a rabi measurement the pulse length of the microwave drive should be drive for multiple mw pulse lengths.

 This implementation uses triggers to synchronize with the laser channel. The procedure can be described as follows:
1. Wait until rising edge trigger from laser gets received
2. Do nothing first half of initial laser time, last half of initial laser time readout.
3. When initial laser is finished, start microwave sequence.
4. When microwave is finished wait for rising edge trigger from laser.
5. When trigger received start readout process.
6. Repeat procedure.

 * Required constants on Sequence property constants:
 * INIT_LASER           - Time laser is on  
 * READOUT              - Time final readout is done
 * SHORTEST_PULSE       - Shortest mw pulse duration
 * PULSE_INCR           - Pulse increment for every n_sweep step
 * N_MEAS               - Number of measurements at each time delay

*/

wave w_init_laser = zeros(INIT_LASER);
wave m_l_init_laser = marker(INIT_LASER*0.5, 0);
wave m_r_init_laser = marker(INIT_LASER*0.5, 1);
wave m_init_laser = join(m_l_init_laser,m_r_init_laser);
wave init_laser = w_init_laser + m_init_laser;


wave w_mw = ones(128);

wave readout = marker(READOUT, 1);

assignWaveIndex(1, 2, init_laser, 0);
assignWaveIndex(1, 2, w_mw, 1);
assignWaveIndex(1, 2, readout, 2);

curr_pulse = SHORTEST_PULSE;
while(1) { 

    for (int i = 0; i < N_meas; i++) {
    waitDigTrigger(1);
    resetOscPhase();
    executeTableEntry(0);
    waitWave()

    executeTableEntry(1);
    playHold(curr_pulse-128)
    waitWave()

    waitDigTrigger(1);
    executeTableEntry(2);
    waitWave()
    }
    curr_pulse+= PULSE_INCR;
}