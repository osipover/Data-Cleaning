import pandas as pd

from unittest import TestCase, main
from ds_info import split_info


class TestSplitInfo(TestCase):
    ACCURACY = 0.001
    TARGET_NAME = 'Job Offer'

    df = pd.read_csv('test_data.csv')

    def test_cgpa_split_info(self):
        cgpa_split_info = split_info(self.df[self.TARGET_NAME], self.df['CGPA'])
        self.assertTrue(1.5211 - self.ACCURACY <= cgpa_split_info <= 1.5211 + self.ACCURACY)

    def test_interactive_split_info(self):
        interactive_split_info = split_info(self.df[self.TARGET_NAME], self.df['Interactive'])
        self.assertTrue(0.9704 - self.ACCURACY <= interactive_split_info <= 0.9704 + self.ACCURACY)

    def test_practical_knowledge_split_info(self):
        practical_knowledge_split_info = split_info(self.df[self.TARGET_NAME], self.df['Practical Knowledge'])
        self.assertTrue(1.4853 - self.ACCURACY <= practical_knowledge_split_info <= 1.4853 + self.ACCURACY)

    def test_comm_skills_split_info(self):
        comm_skills_split_info = split_info(self.df[self.TARGET_NAME], self.df['Comm Skills'])
        self.assertTrue(1.4853 - self.ACCURACY <= comm_skills_split_info <= 1.4853 + self.ACCURACY)


if __name__ == '__main__':
    main()
