import json
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from datetime import datetime
import time
from enum import Enum

# Use monospace to make formatting cleaner
plt.rcParams.update({'font.family':'monospace'})

# fix random seed for reproducibility
np.random.seed(7)

# =============================================================================
# Globals here
# =============================================================================

# Model parameters
KMER_LENGTH = 20
MAX_EPOCH_LENGTH = 500
NUM_LSTM_LAYERS = 512
FIRST_CONV_FILTERS = 128
FIRST_CONV_KERNEL_SIZE = 5
FIRST_DENSE_LAYER = 128
SECOND_DENSE_LAYER = 64
COST_FUNC = "categorical_crossentropy"
OPTIMIZER = "adam"
OUTPUT_ACTIVATION_FUNC = "softmax"
HIDDEN_LAYER_ACTIVATION_FUNC = "relu"
VALIDATION_PERCENT = 0.3
PATIENCE_THRESHOLD = 20

# I wanted to keep all the parameters in a dict so they would be easy to save when we call save_result.
parameters = {
    "kmer_length": KMER_LENGTH,
    "max_epoch_length": MAX_EPOCH_LENGTH,
    "num_lstm_layers": NUM_LSTM_LAYERS,
    "first_conv_filters": FIRST_CONV_FILTERS, 
    "first_conv_kernel_size": FIRST_CONV_KERNEL_SIZE,
    "first_dense_layer": FIRST_DENSE_LAYER,
    "second_dense_layer": SECOND_DENSE_LAYER,
    "cost_func": COST_FUNC,
    "optimizer": OPTIMIZER,
    "output_activation_func": OUTPUT_ACTIVATION_FUNC,
    "hidden_layer_activation_func": HIDDEN_LAYER_ACTIVATION_FUNC,
    "validation_percent": VALIDATION_PERCENT,
    "patience_threshold": PATIENCE_THRESHOLD,
}

# Get raw data, create mapping of characters to integers
with open("alemtuzumab_sequences.txt", "r") as input_file:
    sequences = [seq.split("\n") for seq in input_file.read().split(">") if seq]
    names_to_sequences = {name.replace(":", "").replace("|", ""): "".join(parts).strip() for name, *parts in sequences}

all_chars = set("".join(list(names_to_sequences.values())))
char_to_int = {c: i for i, c in enumerate(all_chars)}
num_classes = len(char_to_int)

# Used in formatting table of parameters
# pad_name = max(len(item) for item in parameters.keys())
# pad_value= max(len(str(item)) for item in parameters.values())

# =============================================================================
# FUNCTIONS START HERE
# =============================================================================

# =============================================================================
# Save what we want from the results
def save_result(run_number, approach, test_accuracy, history, model, secs_to_train):
    now = datetime.now().strftime("%H-%M-%S")
    epoch_length = len(history["epochs_to_patience"])
    epoch_axis = list(range(1, epoch_length + 1))
    pretty_approach_name = " ".join(item.capitalize() for item in approach.split("_"))

    # Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
    fig, ax = plt.subplots()
    ax.plot(epoch_axis, history["accuracy"], label='Training')
    ax.plot(epoch_axis, history["val_accuracy"], label='Validation')
    ax.set_xlabel('Epochs')  # Add an x-label to the axes.
    ax.set_ylabel('Accuracy')  # Add a y-label to the axes.
    ax.set_title(f'Accuracy on Run {run_number} Using Approach {pretty_approach_name}')  # Add a title to the axes.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # FIXME: Was trying to add the model parameters to the figure but fuck it
    # notes = {
    #     "test_accuracy": test_accuracy,
    #     "epochs_actually_trained": epoch_length,
    # }

    # notes.update(parameters)
    # note = "\n".join(f"{name:<{pad_name}}: {value:>{pad_value}}" for name, value in notes.items())

    # ax.annotate(note,
    #             xy = (1.1, 0.5),
    #             xycoords='axes fraction',
    #             ha='left',
    #             va="center",
    #             fontsize=10)

    ax.legend();  # Add a legend.
    filename = f"{approach}_{run_number}_{now}"

    # FIXME: This shit sucks. The actual method to call depends on the backend used for the UI. This only works if backend
    # is QT. Really stupid that it's written to be backend specific. Should be a universal call regardless of backend to
    # ask the screen to render it (and save it) maximized.
    # Since the size of the saved image is based on "maximizing" which is dependent on the particular machine that runs it,
    # this code produces different size images for different machines, which also sucks. I'm too lazy to figure out how to
    # Resize this thing appropriately
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    fig.savefig(f"{filename}_accuracy.png")

    result = {
        "test_accuracy": test_accuracy,
        "epochs_actually_trained": epoch_length,
        "num_trainable_vars": int(np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])),
        "secs_to_train": secs_to_train,
        "parameters": parameters,
        "history": history,
    }

    with open(f"{filename}_results.json", "w+") as f:
        f.write(json.dumps(result))

    with open(f"{filename}_model_summary.txt", "w+") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Used this to print out the epochs to patience, but don't need it all the time
    x = []
    y = []
    for x_item, y_item in history["epochs_to_patience"]:
        x.append(x_item)
        y.append(y_item)
    # Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
    fig, ax = plt.subplots()
    ax.plot(x, y, label='Patience')
    ax.set_xlabel('Epochs')  # Add an x-label to the axes.
    ax.set_ylabel('Patience')  # Add a y-label to the axes.
    ax.set_title(f'Epochs to Patience')  # Add a title to the axes.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend();  # Add a legend.
    fig.savefig(f"{filename}_patience.png")

# =============================================================================
class ReportPatience(Callback):
    """
    Callback to print out the patience, i.e. how many steps its been since
    We've found the minimum val_loss while fitting the data.

    Updates history returned by model.fit() to include "epochs_to_patience"
    key that holds a list of tuples, where first element is the epoch and second
    is the patience at that epoch
    """
    def __init__(self):
        super().__init__()
        self.current_best_loss = None
        self.current_patience = 0

    def on_epoch_end(self, epoch, logs):
        current_val_loss = logs["val_loss"]
        if self.current_best_loss is None:
            self.current_best_loss = current_val_loss

        if current_val_loss <= self.current_best_loss:
            print(f"Found a better loss in {self.current_patience} steps")
            self.current_best_loss = current_val_loss
            self.current_patience = 0
        else:
            self.current_patience += 1

        logs["epochs_to_patience"] = (epoch, self.current_patience)

# =============================================================================
def preprocess_data(dataset):
    """
    Preprocesses raw dataset and returns tuple (dataX, dataY)
    """
    input_as_lst = []
    output_as_lst = []
    for inp, out in dataset:
        input_as_lst.append(inp)
        output_as_lst.append(out)

    # reshape X to be [samples, time steps, features], normalize
    dataX = np.reshape(input_as_lst, (len(input_as_lst), KMER_LENGTH, 1))
    dataX = dataX / float(num_classes)

    # Convert output to categorical vector
    dataY = np_utils.to_categorical(output_as_lst, num_classes=num_classes)

    return dataX, dataY

# =============================================================================
def approach1_data():
    """
    For this approach, we choose each of our N sequences as a holdout test set and 
    train on the other sequences. We do N runs of this. 

    I didn't want to delete this work, so I moved my old approach into a function.
    Can easily switch back by updating commented out line below
    """
    runs = []
    for test_name in names_to_sequences:
        test_seq = names_to_sequences[test_name]
        training_seqs = [names_to_sequences[name] for name in names_to_sequences if name != test_name]

        training_data = []
        test_data = []

        # iterate over each sequence, starting at kmer_length to ignore first kmer_length characters
        for seq in training_seqs:
            for index in range(KMER_LENGTH, len(seq)):
                seq_in = seq[index - KMER_LENGTH:index]
                seq_out = seq[index]
                training_data.append([[char_to_int[char] for char in seq_in], char_to_int[seq_out]])

        for index in range(KMER_LENGTH, len(test_seq)):
            seq_in = test_seq[index - KMER_LENGTH:index]
            seq_out = test_seq[index]
            test_data.append([[char_to_int[char] for char in seq_in], char_to_int[seq_out]])

        runs.append((training_data, test_data))

    return runs

# =============================================================================
def approach2_data():
    """
    For this approach, we randomize all the sequence data, then do a usual training/validation/test split.
    We do one run
    """
    all_data = list(names_to_sequences.values())
    random.shuffle(all_data)
    test_threshold = int(0.10 * len(all_data))

    raw_test_data = all_data[:test_threshold]
    raw_training_data = all_data[test_threshold:]

    def convert_raw_to_processed_data(dataset):
        processed_data = []
        # iterate over each sequence, starting at kmer_length to ignore first kmer_length characters
        for seq in dataset:
            for index in range(KMER_LENGTH, len(seq)):
                seq_in = seq[index - KMER_LENGTH:index]
                seq_out = seq[index]
                processed_data.append([[char_to_int[char] for char in seq_in], char_to_int[seq_out]])

        return processed_data
    
    test_data = convert_raw_to_processed_data(raw_test_data)
    training_data = convert_raw_to_processed_data(raw_training_data)
    return [(training_data, test_data)]

# =============================================================================
def approach3_data():
    """
    The new and latest approach to handling the training/test data.
    In this approach, all sequences are used as training data and only the missing gaps 
    from our paper are used for model evaluation
    """
    target_sequence = "DIQMTQSPSSLSASVGDRVTITCKASQNIDKYLNWYQQKPGKAPKLLIYNTNNLQTGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCLQHISRPRTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    missing_indices = set([0, 1, 2, 24, 25, 26, 62, 63, 64, 66, 67, 68, 69, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 209, 210, 211, 212, 213])

    training_data = []
    test_data = []

    # Iterate over each sequence, starting at kmer_length to ignore first kmer_length characters
    training_seqs = names_to_sequences.values()
    for seq in training_seqs:
        for index in range(KMER_LENGTH, len(seq)):
            seq_in = seq[index - KMER_LENGTH:index]
            seq_out = seq[index]
            training_data.append([[char_to_int[char] for char in seq_in], char_to_int[seq_out]])

    for index in [i for i in missing_indices if i >= KMER_LENGTH]:
        seq_in = target_sequence[index - KMER_LENGTH:index]
        seq_out = target_sequence[index]
        test_data.append([[char_to_int[char] for char in seq_in], char_to_int[seq_out]])

    return [(training_data, test_data)]

approaches = {
    "withhold_one_sequence": approach1_data,
    "test_against_gaps": approach2_data,
    "randomize_split": approach3_data,
}

# =============================================================================
def main():

    # Swap these out to try different approaches to handling the data
    approach = "withhold_one_sequence"
    # approach = "test_against_gaps"
    # approach = "randomize_split"

    runs = approaches[approach]()
    for run_num, (training_data, test_data) in enumerate(runs, 1):

        # Shuffle the training data so we can split randomly into validation / training
        random.shuffle(training_data)

        # Split training set into validation set based on validation_percent
        validation_threshold = int(VALIDATION_PERCENT * len(training_data))
        validation_data = training_data[:validation_threshold]
        training_data = training_data[validation_threshold:]

        # Convert lists of lists to appropriate data structure complete with any necessary preprocessing
        trainX, trainY = preprocess_data(training_data)
        testX, testY = preprocess_data(test_data)
        validX, validY = preprocess_data(validation_data)

        # create the model
        model = Sequential()
        model.add(Conv1D(FIRST_CONV_FILTERS, FIRST_CONV_KERNEL_SIZE))
        model.add(LSTM(NUM_LSTM_LAYERS, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(FIRST_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC))
        model.add(Dense(SECOND_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC))
        model.add(Dense(trainY.shape[1], activation=OUTPUT_ACTIVATION_FUNC))
        model.compile(loss=COST_FUNC, optimizer=OPTIMIZER, metrics=['accuracy'])

        # fit the data, summarize performance of the model
        start = time.time()
        history = model.fit(
            trainX, 
            trainY, 
            epochs=MAX_EPOCH_LENGTH, 
            batch_size=1, 
            verbose=2, 
            validation_data=(validX, validY), 
            callbacks = [EarlyStopping(monitor="val_loss", patience=PATIENCE_THRESHOLD), ReportPatience()],
        )
        end = time.time()
        secs_to_train = int(end - start)

        _, accuracy = model.evaluate(testX, testY, verbose=0)
        save_result(run_num, approach, f"{accuracy:.2f}", history.history, model, secs_to_train)

if __name__ == "__main__":
    main()