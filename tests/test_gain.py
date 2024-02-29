import pandas as pd

from unittest import TestCase, main
from ds_info import gain


class TestGain(TestCase):
    ACCURACY = 0.001
    TARGET_NAME = 'Job Offer'

    df = pd.read_csv('test_data.csv')

    def test_cgpa_gain(self):
        cgpa_gain = gain(self.df[self.TARGET_NAME], self.df['CGPA'])
        self.assertTrue(0.5564 - self.ACCURACY <= cgpa_gain <= 0.5564 + self.ACCURACY)

    def test_interactive_gain(self):
        interactive_gain = gain(self.df[self.TARGET_NAME], self.df['Interactive'])
        self.assertTrue(0.0911 - self.ACCURACY <= interactive_gain <= 0.0911 + self.ACCURACY)

    def test_practical_knowledge_gain(self):
        practical_knowledge_gain = gain(self.df[self.TARGET_NAME], self.df['Practical Knowledge'])
        self.assertTrue(0.2448 - self.ACCURACY <= practical_knowledge_gain <= 0.2448 + self.ACCURACY)

    def test_comm_skills_gain(self):
        comm_skills_gain = gain(self.df[self.TARGET_NAME], self.df['Comm Skills'])
        self.assertTrue(0.5202 - self.ACCURACY <= comm_skills_gain <= 0.5202 + self.ACCURACY)


if __name__ == '__main__':
    main()
