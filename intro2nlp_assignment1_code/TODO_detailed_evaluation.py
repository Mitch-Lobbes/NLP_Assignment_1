# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class

import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
import json
import os
import plotly.express as px
import train
import evaluate
import utils
import torch
import logging
from model.data_loader import DataLoader
import model.net as net
import build_vocab


class Detailed_Eval:

    def __init__(self, filelist):
        self.filelist = filelist
        self.model_list = ["Random", "Majority", "Length", "Frequency", "LSTM"]
        self.table = pd.DataFrame(columns=["Model",
                                           "Class_N Precision",
                                           "Class_N Recall",
                                           "Class_N F1",
                                           "Class_C Precision",
                                           "Class_C Recall",
                                           "Class_C F1",
                                           "Weighted Average F1"])
        self.update_table()
        self.print_table()

    def update_table(self):
        for filepath in enumerate(self.filelist):
            file_output = pd.read_csv(filepath_or_buffer=filepath[1], encoding='latin-1',
                                      names=["Word", "Gold", "Prediction"],
                                      sep="\t")
            file_output = file_output.dropna(axis=0)

            C_metrics = precision_recall_fscore_support(file_output["Gold"], file_output["Prediction"],
                                                        average='binary', pos_label='C')
            N_metrics = precision_recall_fscore_support(file_output["Gold"], file_output["Prediction"],
                                                        average='binary', pos_label='N')
            weighted_avg_f1 = precision_recall_fscore_support(file_output["Gold"], file_output["Prediction"],
                                                              average='weighted')
            self.table.loc[len(self.table.index)] = [self.model_list[filepath[0]], round(N_metrics[0], 2),
                                                     round(N_metrics[1], 2), round(N_metrics[2], 2),
                                                     round(C_metrics[0], 2), round(C_metrics[1], 2),
                                                     round(C_metrics[2], 2), round(weighted_avg_f1[2], 2)]

    def print_table(self):
        print("Detailed Evaluation (2.5 points)")
        print(self.table.to_string())
        print("\n")


class Experiments:

    def __init__(self, epochs_values, filepath):
        self.f1_list = []
        self.filepath = filepath
        self.lr_values = epochs_values
        self.epoch_experiment()

    def change_lr(self, epoch: int):

        with open(self.filepath, 'r') as file:
            json_data = json.load(file)
            for item in enumerate(json_data):
                if item[1] == 'learning_rate':
                    json_data[item[1]] = epoch
        with open(self.filepath, 'w') as file:
            json.dump(json_data, file, indent=2)

    def build_vocab(self):
        build_vocab.start()

    def train_lstm(self):
        train.start()

    def evaluate_lstm(self, index):

        # Load the parameters
        args = evaluate.parser.parse_args()
        json_path = os.path.join(args.model_dir, 'params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = utils.Params(json_path)

        # use GPU if available
        params.cuda = torch.cuda.is_available()  # use GPU is available

        # Set the random seed for reproducible experiments
        torch.manual_seed(230)
        if params.cuda: torch.cuda.manual_seed(230)

        # Get the logger
        utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

        # Create the input data pipeline
        logging.info("Creating the dataset...")

        # load data
        data_loader = DataLoader(args.data_dir, params)
        data = data_loader.load_data(['test'], args.data_dir)
        test_data = data['test']

        # specify the test set size
        params.test_size = test_data['size']
        test_data_iterator = data_loader.data_iterator(test_data, params)

        logging.info("- done.")

        # Define the model
        model = net.Net(params).cuda() if params.cuda else net.Net(params)

        loss_fn = net.loss_fn
        metrics = net.metrics

        logging.info("Starting evaluation")

        # Reload weights from the saved file
        utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

        # Evaluate
        num_steps = (params.test_size + 1) // params.batch_size
        # MY ADJUSTMENTS
        # reverse the vocab and tag dictionary to be able to map back from ids to words and tags
        id2word = {v: k for k, v in data_loader.vocab.items()}
        tags = {v: k for k, v in data_loader.tag_map.items()}
        outfile = args.model_dir + f"/model_output_{index}.tsv"
        test_metrics = evaluate.evaluate_and_output(model, loss_fn, test_data_iterator, metrics, num_steps, id2word,
                                                    tags,
                                                    outfile)
        save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
        utils.save_dict_to_json(test_metrics, save_path)

    def epoch_experiment(self):

        self.build_vocab()
        for i in range(len(self.lr_values)):
            # changing the epoch parameter
            self.change_lr(self.lr_values[i])
            self.train_lstm()
            self.evaluate_lstm(i)

        for i in range(len(self.lr_values)):
            self.f1_calculation(f"experiments/base_model/model_output_{i}.tsv")

        print(f'f1 results are: {self.f1_list}')
        self.plot_hyperparameter()

    def f1_calculation(self, filepath):

        file_output = pd.read_csv(filepath_or_buffer=filepath, encoding='latin-1',
                                  names=["Word", "Gold", "Prediction"],
                                  sep="\t")
        file_output = file_output.dropna(axis=0)

        weighted_avg_f1 = precision_recall_fscore_support(file_output["Gold"], file_output["Prediction"],
                                                          average='weighted')
        self.f1_list.append(round(weighted_avg_f1[2], 2))

    def plot_hyperparameter(self):

        df = pd.DataFrame(dict(
            learning_rate=self.lr_values,
            F1_score=self.f1_list
        ))
        fig = px.line(df, x="learning_rate", y="F1_score", title='F1 score based on hyperparamter "learning rate"')
        fig.write_image(f"experiments/base_model/learning_rate_f1.png")
        fig.show()



# Run this part for exercise 12 to get the table
# !!! The file-paths need to be in the correct order --> "Random", "Majority", "Length", "Frequency", "LSTM" !!!

table12_output = ['experiments/base_model/baselinesrandom.tsv', 'experiments/base_model/baselinesmajority.tsv',
                  'experiments/base_model/baselineslength.tsv', 'experiments/base_model/baselinesfrequency.tsv',
                  'experiments/base_model/model_original_output.tsv']
ex12 = Detailed_Eval(table12_output)

# Run this part for Exercise 14

lr_values = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
json_path = "experiments/base_model/params.json"
x = Experiments(lr_values, json_path)
