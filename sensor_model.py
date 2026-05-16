def simuliere_ball_sensor_abstand(real_abstand_cm: float) -> float:
    real_abstand_cm = max(0.0, float(real_abstand_cm))

    if real_abstand_cm <= 7.0:
        return 55.0 - (real_abstand_cm / 7.0) * 15.0
    if real_abstand_cm <= 9.0:
        return real_abstand_cm
    return 9.0
