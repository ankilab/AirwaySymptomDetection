import itertools
import numpy as np


class CrossValidation():
    def __init__(self, k):
        """
        Initializes Class "CrossValidation".
        Args:
            k: k for k-fold-cross-validation.
        """
        if k > 30:
            raise ValueError("k can't be higher than 30, since we only have 6 different participants and always one "
                             "participant as val data set and one as test data set.")
        self.k = k

        self.p1 = ["P1_1", "P1_2", "P1_3"]
        self.p2 = ["P2_1", "P2_2", "P2_3"]
        self.p3 = ["P3_1", "P3_2", "P3_3"]
        self.p4 = ["P4_1", "P4_2", "P4_3"]
        self.p5 = ["P5_1", "P5_2", "P5_3"]
        self.p6 = ["P6_1", "P6_2", "P6_3"]
        self.all_participants = ["p1", "p2", "p3", "p4", "p5", "p6"]

        self.p_used_for_val_and_test = []

    def __get_unique_data_split(self):
        """
        Generates an unique data split for cross-validation.
        Returns:
        A unique data split.
        """
        # get random participant for validation data
        rand_p_val, rand_p_test = np.random.choice(self.all_participants, 2, replace=False)

        while self.p_used_for_val_and_test.__contains__([rand_p_val, rand_p_test]):
            rand_p_val, rand_p_test = np.random.choice(self.all_participants, 2, replace=False)

        self.p_used_for_val_and_test.append([rand_p_val, rand_p_test])
        val_data = self.__get_participant_list(rand_p_val)
        test_data = self.__get_participant_list(rand_p_test)

        # get train data
        train_data = []
        p_training = []
        for p in self.all_participants:
            if p == rand_p_val or p == rand_p_test:
                continue
            else:
                p_training.append(p)
                train_data.append(self.__get_participant_list(p))
        train_data = list(itertools.chain(*train_data))

        return train_data, val_data, test_data, [p_training, rand_p_val, rand_p_test]

    def get_unique_data_split(self):
        train_data, val_data, test_data, p_for_table = self.__get_unique_data_split()
        return train_data, val_data, test_data, p_for_table

    def __get_participant_list(self, p):
        if p == "p1":
            return self.p1
        elif p == "p2":
            return self.p2
        elif p == "p3":
            return self.p3
        elif p == "p4":
            return self.p4
        elif p == "p5":
            return self.p5
        elif p == "p6":
            return self.p6