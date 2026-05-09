 /*
To perform a rabi measurement the pulse length of the microwave drive should be drive for multiple mw pulse lengths.

This implementation uses triggers to synchronize with the mw channel. The procedure can be described as follows:
1. Start initial laser
2. Wait for falling edge trigger from mw than proceed waiting for mw delay.
3. Shoot laser for readout time. 
4. wait for rising edge trigger from mw to repeate procedure.

 * Required constants on Sequence property constants:
 * INIT_LASER           - Time laser is on 
 * MW_DELAY             - Time after laser until mw is turned on 
 * READOUT              - Time final readout is done
 * N_SWEEP              - Number of sweep steps
 * N_MEAS               - Number of measurements at each time delay

*/


wave w_init_laser = zeros(INIT_LASER);
wave m_init_laser = marker(INIT_LASER, 1);
wave init_laser = w_init_laser + m_init_laser;

wave w_readout = zeros(READOUT);
wave m_readout = marker(READOUT, 1);
wave readout = w_readout + m_readout

assignWaveIndex(1, 2, init_laser, 0);
assignWaveIndex(1, 2, readout, 1);

for (i = 0; i < N_SWEEP; i++) { 
    resetOscPhase();

    repeat(N_MEAS) {
        executeTableEntry(0);
        waitWave();
        waitDigTrigger(1); 
        Wait(MW_DELAY);
        executeTableEntry(1);
        waitWave();
        waitDigTrigger(1);
    }
}





