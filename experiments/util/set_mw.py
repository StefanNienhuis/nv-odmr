from zhinst.toolkit import Session, CommandTable

AWG_SERVER_HOST = 'localhost'
AWG_SERVER_PORT = 8004
AWG_DEVICE = 'DEV12120'
AWG_CHANNEL = 2
AWG_SAMPLE_RATE = 2e9

def set_steady():
    awg_session = Session(AWG_SERVER_HOST, AWG_SERVER_PORT)
    awg_device = awg_session.connect_device(AWG_DEVICE)

    awg_device.check_compatibility()

    awg_channel = awg_device.sgchannels[AWG_CHANNEL]

    awg_channel.synchronization.enable(0)

    center_freq = 2.8e9

    awg_channel.configure_channel(
        enable=True,
        output_range=10,
        center_frequency=center_freq,
        rf_path=True
    )
    
    awg_channel.configure_sine_generation(
        enable=True,
        osc_index=0,
        osc_frequency=2.8e9,
        phase=0
    )

    awg_channel.configure_pulse_modulation(
        enable=False,
        osc_index=0,
        osc_frequency=2.8e9,
        global_amp=1,
        phase=0
    )
