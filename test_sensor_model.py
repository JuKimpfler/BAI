import unittest

from sensor_model import simuliere_ball_sensor_abstand


class SensorModelTests(unittest.TestCase):
    def test_positive_distance_is_unchanged(self):
        self.assertEqual(simuliere_ball_sensor_abstand(50.0), 50.0)

    def test_decimal_distance_is_unchanged(self):
        self.assertAlmostEqual(simuliere_ball_sensor_abstand(3.5), 3.5)

    def test_negative_values_are_clamped(self):
        self.assertEqual(simuliere_ball_sensor_abstand(-2.0), 0.0)


if __name__ == "__main__":
    unittest.main()
