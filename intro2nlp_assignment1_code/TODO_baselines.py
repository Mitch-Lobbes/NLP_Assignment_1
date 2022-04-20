import os
from collections import Counter
from sklearn.metrics import accuracy_score
import random
from wordfreq import word_frequency
import csv


class Baseline:

    def __init__(self):
        self._directory = os.getcwd()
        self._train_path = f"{self._directory}/data/preprocessed/train/"
        self._dev_path = f"{self._directory}/data/preprocessed/val/"
        self._test_path = f"{self._directory}/data/preprocessed/test/"
        self._data_dict = {}
        self.predictions_dict = {}
        self._majority = None

        self._load_data()

        self._run()

    def _load_data(self):

        with open(self._train_path + "sentences.txt") as sent_file:
            self._data_dict["train_sentences"] = sent_file.readlines()

        with open(self._train_path + "labels.txt") as label_file:
            self._data_dict["train_labels"] = label_file.readlines()

        with open(self._dev_path + "sentences.txt") as dev_file:
            self._data_dict["dev_sentences"] = dev_file.readlines()

        with open(self._dev_path + "labels.txt") as dev_label_file:
            self._data_dict["dev_labels"] = dev_label_file.readlines()

        with open(self._test_path + "sentences.txt") as testfile:
            self._data_dict["test_sentences"] = testfile.readlines()

        with open(self._test_path + "labels.txt") as test_label_file:
            self._data_dict["test_labels"] = test_label_file.readlines()

    def _run(self):

        self._get_majority_class()
        m1_ac = self._majority_baseline(data=self._data_dict["test_sentences"], labels=self._data_dict["test_labels"])
        print(f"Majority Accuracy Test: {m1_ac}")
        m2_ac = self._majority_baseline(data=self._data_dict["dev_sentences"], labels=self._data_dict["dev_labels"])
        print(f"Majority Accuracy Dev: {m2_ac}")
        print("-----------------------------------------")
        r1_ac = self._random_baseline(data=self._data_dict["test_sentences"], labels=self._data_dict["test_labels"])
        print(f"Random Accuracy Test: {r1_ac}")
        r2_ac = self._random_baseline(data=self._data_dict["dev_sentences"], labels=self._data_dict["dev_labels"])
        print(f"Random Accuracy Dev: {r2_ac}")
        print("-----------------------------------------")
        print(f"Length Accuracy Test")
        for i in range(16):
            self._length_baseline(data=self._data_dict["test_sentences"],
                                  labels=self._data_dict["test_labels"], threshold=i)
        print("-------------------------")
        print(f"Length Accuracy Dev")
        for i in range(16):
            self._length_baseline(data=self._data_dict["dev_sentences"],
                                  labels=self._data_dict["dev_labels"], threshold=i)

        print("-------------------------")
        print(f"Freq Accuracy Dev")
        for i in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
            self._frequency_baseline(data=self._data_dict["dev_sentences"], labels=self._data_dict["dev_labels"],
                                     threshold=i)

        print("-------------------------")
        print(f"Freq Accuracy Test")
        for i in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
            self._frequency_baseline(data=self._data_dict["test_sentences"], labels=self._data_dict["test_labels"],
                                     threshold=i)

        self._length_baseline(data=self._data_dict["test_sentences"],
                              labels=self._data_dict["test_labels"], threshold=0)
        self._frequency_baseline(data=self._data_dict["dev_sentences"], labels=self._data_dict["dev_labels"],
                                 threshold=1e-05)
        self._majority_baseline(data=self._data_dict["test_sentences"], labels=self._data_dict["test_labels"])
        self._random_baseline(data=self._data_dict["test_sentences"], labels=self._data_dict["test_labels"])

        self._save_performance_file(dict_key='length')
        self._save_performance_file(dict_key='frequency')
        self._save_performance_file(dict_key='majority')
        self._save_performance_file(dict_key='random')

    def _save_performance_file(self, dict_key: str):
        flattened_predictions = self.predictions_dict[dict_key]

        test_words = self._data_dict['test_sentences']
        test_words = [i.strip() for i in test_words]

        test_labels = self._data_dict['test_labels']
        test_labels = [i.strip() for i in test_labels]

        flattened_words = [word for line in test_words for word in line.split()]
        flattened_labels = [word for line in test_labels for word in line.split()]

        combined_list = [list(a) for a in zip(flattened_words, flattened_labels, flattened_predictions)]

        with open(f'experiments/base_model/baselines{dict_key}.tsv', 'wt') as outfile:
            tsvwriter = csv.writer(outfile, delimiter='\t')
            tsvwriter.writerows(combined_list)

    def _get_majority_class(self):

        label_counter = Counter()
        train_labels = [sentence.strip("\n").split(" ") for sentence in self._data_dict["train_labels"]]

        for sentence in train_labels:
            label_counter.update(label for label in sentence)

        self._majority = label_counter.most_common(1)[0][0]

    def _majority_baseline(self, data: list, labels: list):
        predictions = []
        formatted_labels = []

        labels = [sentence.strip("\n").split(" ") for sentence in labels]
        for label in labels:
            formatted_labels.extend(label)

        for label in formatted_labels:
            predictions.append(self._majority)

        self.predictions_dict['majority'] = predictions
        return accuracy_score(formatted_labels, predictions, normalize=True), predictions

    def _random_baseline(self, data: list, labels: list):
        predictions = []
        formatted_labels = []

        labels = [sentence.strip("\n").split(" ") for sentence in labels]
        for label in labels:
            formatted_labels.extend(label)

        for label in formatted_labels:
            predictions.append(random.choice(["C", "N"]))

        self.predictions_dict['random'] = predictions
        return accuracy_score(formatted_labels, predictions, normalize=True)

    def _length_baseline(self, data: list, labels: list, threshold: int):

        predictions = []
        formatted_labels = []
        formatted_data = []

        data = [sentence.strip("\n").split(" ") for sentence in data]
        labels = [sentence.strip("\n").split(" ") for sentence in labels]

        for i in range(len(labels)):
            formatted_labels.extend(labels[i])
            formatted_data.extend(data[i])

        for token in formatted_data:
            length = len(token)
            assignment = "C" if length <= threshold else "N"
            predictions.append(assignment)

        self.predictions_dict['length'] = predictions
        accuracy = accuracy_score(formatted_labels, predictions, normalize=True)
        print(f"{accuracy}, threshold: {threshold}")

    def _frequency_baseline(self, data: list, labels: list, threshold: float):

        predictions = []
        formatted_labels = []
        formatted_data = []

        data = [sentence.strip("\n").split(" ") for sentence in data]
        labels = [sentence.strip("\n").split(" ") for sentence in labels]

        for i in range(len(labels)):
            formatted_labels.extend(labels[i])
            formatted_data.extend(data[i])

        for token in formatted_data:
            freq = word_frequency(token, 'en')
            assignment = "C" if freq <= threshold else "N"
            predictions.append(assignment)

        self.predictions_dict['frequency'] = predictions
        accuracy = accuracy_score(formatted_labels, predictions, normalize=True)
        print(f"{accuracy}, threshold: {threshold}")


baseline_1 = Baseline()
