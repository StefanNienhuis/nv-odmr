import pycobolt

LASER_SN = '31977'
LASER_CURRENT = 57
LASER_POWER = 1

laser = pycobolt.CoboltLaser(serialnumber=LASER_SN)

# laser.current_modulation_mode()
# laser.set_modulation_current(LASER_CURRENT)

# laser.power_modulation_mode()
# laser.set_modulation_power(LASER_POWER)
# print(f"Laser mode: {laser.get_mode()}")


laser.constant_current()
laser.set_current(LASER_CURRENT)

# laser.constant_power()
# laser.set_power(LASER_POWER)