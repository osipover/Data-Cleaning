import pandas as pd

from unittest import TestCase, main
from ds_info import find_attr_max_gain_ratio


class TestFindAttrMaxGainRatio(TestCase):
    ACCURACY = 0.001
    TARGET_NAME = 'Job Offer'

    df = pd.read_csv('test_data.csv')

    def test_find_attr_max_gain_ratio(self):
        attr_max = find_attr_max_gain_ratio(self.TARGET_NAME, self.df)
        self.assertEqual(attr_max, 'CGPA')