import pandas as pd

from unittest import TestCase, main
from ds_info import gain_ratio


class TestGainRatio(TestCase):
    ACCURACY = 0.001
    TARGET_NAME = 'Job Offer'

    df = pd.read_csv('test_data.csv')

    def test_cgpa_gain_ratio(self):
        cgpa_gain_ratio = gain_ratio(self.df['CGPA'], self.df[self.TARGET_NAME])
        self.assertTrue(0.3658 - self.ACCURACY <= cgpa_gain_ratio <= 0.3658 + self.ACCURACY)

    def test_interactive_gain_ratio(self):
        interactive_gain_ratio = gain_ratio(self.df['Interactive'], self.df[self.TARGET_NAME])
        self.assertTrue(0.0939 - self.ACCURACY <= interactive_gain_ratio <= 0.0939 + self.ACCURACY)

    def test_practical_knowledge_gain_ratio(self):
        practical_knowledge_gain_ratio = gain_ratio(self.df['Practical Knowledge'], self.df[self.TARGET_NAME])
        self.assertTrue(0.1648 - self.ACCURACY <= practical_knowledge_gain_ratio <= 0.1648 + self.ACCURACY)

    def test_comm_skills_gain_ratio(self):
        comm_skills_gain_ratio = gain_ratio(self.df['Comm Skills'], self.df[self.TARGET_NAME])
        self.assertTrue(0.3502 - self.ACCURACY <= comm_skills_gain_ratio <= 0.3502 + self.ACCURACY)


if __name__ == '__main__':
    main()
