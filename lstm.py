import json
import numpy as np
import random
import os
import time
from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback

# Use monospace to make formatting cleaner
plt.rcParams.update({'font.family':'monospace'})

# fix random seed for reproducibility
np.random.seed(7)

# =============================================================================
# Globals here
# =============================================================================

# Path to save results
RESULTS_PATH = os.path.join(os.getcwd(), "results")

# Model parameters
KMER_LENGTH = 20
MAX_EPOCH_LENGTH = 5
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

# =============================================================================
# FUNCTIONS START HERE
# =============================================================================

# =============================================================================
# Save what we want from the results
def save_result(test_accuracy, history, model, secs_to_train):
    now = datetime.now().strftime("%H-%M-%S")
    epoch_length = len(history["epochs_to_patience"])
    epoch_axis = list(range(1, epoch_length + 1))

    # Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
    fig, ax = plt.subplots()
    ax.plot(epoch_axis, history["accuracy"], label='Training')
    ax.plot(epoch_axis, history["val_accuracy"], label='Validation')
    ax.set_xlabel('Epochs')  # Add an x-label to the axes.
    ax.set_ylabel('Accuracy')  # Add a y-label to the axes.
    ax.set_title(f'Accuracy')  # Add a title to the axes.
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
    filename = f"{now}"

    # FIXME: This shit sucks. The actual method to call depends on the backend used for the UI. This only works if backend
    # is QT. Really stupid that it's written to be backend specific. Should be a universal call regardless of backend to
    # ask the screen to render it (and save it) maximized.
    # Since the size of the saved image is based on "maximizing" which is dependent on the particular machine that runs it,
    # this code produces different size images for different machines, which also sucks. I'm too lazy to figure out how to
    # Resize this thing appropriately
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    fig.savefig(os.path.join(RESULTS_PATH, f"{filename}_accuracy.png"))

    result = {
        "test_accuracy": test_accuracy,
        "epochs_actually_trained": epoch_length,
        "num_trainable_vars": int(np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])),
        "secs_to_train": secs_to_train,
        "parameters": parameters,
        "history": history,
    }

    path = os.path.join(RESULTS_PATH, f"{filename}_results.json")
    with open(path, "w+") as f:
        f.write(json.dumps(result))

    path = os.path.join(RESULTS_PATH, f"{filename}_model_summary.txt")
    with open(path, "w+") as f:
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
    fig.savefig(os.path.join(RESULTS_PATH, f"{filename}_patience.png"))

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
def main():

    all_data = list(names_to_sequences.values())
    random.shuffle(all_data) # FIXME: Make sure this randomization is working
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

    # Shuffle the training data so we can split randomly into validation / training
    # FIXME: Do we need to shuffle again?
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
    save_result(f"{accuracy:.2f}", history.history, model, secs_to_train)

if __name__ == "__main__":
    main()
