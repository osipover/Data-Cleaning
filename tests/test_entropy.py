import pandas as pd

from unittest import TestCase, main
from ds_info import entropy_info, entropy


class TestEntropy(TestCase):
    ACCURACY = 0.001
    TARGET_NAME = 'Job Offer'

    df = pd.read_csv('test_data.csv')

    def test_target_entropy(self):
        target_entropy = entropy(self.df[self.TARGET_NAME])
        self.assertTrue(0.8807 - self.ACCURACY <= target_entropy <= 0.8807 + self.ACCURACY)

    def test_cgpa_entropy(self):
        cgpa_entropy = entropy_info(self.df[self.TARGET_NAME], self.df['CGPA'])
        self.assertTrue(0.3243 - self.ACCURACY <= cgpa_entropy <= 0.3243 + self.ACCURACY)

    def test_interactive_entropy(self):
        interactive_entropy = entropy_info(self.df[self.TARGET_NAME], self.df['Interactive'])
        self.assertTrue(0.7896 - self.ACCURACY <= interactive_entropy <= 0.7896 + self.ACCURACY)

    def test_practical_knowledge_entropy(self):
        practical_knowledge = entropy_info(self.df[self.TARGET_NAME], self.df['Practical Knowledge'])
        self.assertTrue(0.6361 - self.ACCURACY <= practical_knowledge <= 0.6361 + self.ACCURACY)

    def test_comm_skills_entropy(self):
        comm_skills_entropy = entropy_info(self.df[self.TARGET_NAME], self.df['Comm Skills'])
        self.assertTrue(0.3609 - self.ACCURACY <= comm_skills_entropy <= 0.3609 + self.ACCURACY)


if __name__ == '__main__':
    main()
