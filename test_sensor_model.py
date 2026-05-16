import unittest

from sensor_model import simuliere_ball_sensor_abstand


class SensorModelTests(unittest.TestCase):
    def test_distances_above_nine_are_capped(self):
        self.assertEqual(simuliere_ball_sensor_abstand(50.0), 9.0)

    def test_distances_between_seven_and_nine_are_real_values(self):
        self.assertEqual(simuliere_ball_sensor_abstand(8.0), 8.0)
        self.assertEqual(simuliere_ball_sensor_abstand(9.0), 9.0)

    def test_jump_at_seven_cm(self):
        self.assertEqual(simuliere_ball_sensor_abstand(7.0), 40.0)

    def test_linear_range_below_seven(self):
        self.assertEqual(simuliere_ball_sensor_abstand(0.0), 55.0)
        self.assertAlmostEqual(simuliere_ball_sensor_abstand(3.5), 47.5)

    def test_negative_values_are_clamped(self):
        self.assertEqual(simuliere_ball_sensor_abstand(-2.0), 55.0)


if __name__ == "__main__":
    unittest.main()
