
import functools
import json
from this import d
from turtle import forward
from unicodedata import bidirectional
import numpy as np
import os
import time
from datetime import datetime
import sys

from keras.models import Sequential

# TESTING FUNCTIONAL API
# from keras.layers import Dense, LSTM, Conv1D, Bidirectional, Embedding, MaxPooling1D, Input
from tensorflow import keras
from tensorflow.keras import layers

from keras.utils import np_utils
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from pygments import highlight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback

from colors import bcolors

# Use monospace to make formatting cleaner
plt.rcParams.update({'font.family':'monospace'})

# fix random seed for reproducibility
np.random.seed(7)

# =============================================================================
# Globals here
# =============================================================================

# Path to save results
RESULTS_PATH = os.path.join(os.getcwd(), "new_approach_results")

# Model parameters
KMER_LENGTH = 5
MAX_EPOCH_LENGTH = 100
EMBEDDING_DIM = 128
NUM_LSTM_LAYERS = 256
FIRST_CONV_FILTERS = 128
FIRST_CONV_KERNEL_SIZE = 3
FIRST_DENSE_LAYER = 128
SECOND_DENSE_LAYER = 64
COST_FUNC = "categorical_crossentropy"
OPTIMIZER = "adam"
OUTPUT_ACTIVATION_FUNC = "softmax"
HIDDEN_LAYER_ACTIVATION_FUNC = "relu"
VALIDATION_PERCENT = 0.1
PATIENCE_THRESHOLD = 20
BATCH_SIZE = 64
POOL_SIZE = 2
NUM_CLASSES = -1

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

    parameters = {
        "kmer_length": KMER_LENGTH,
        "max_epoch_length": MAX_EPOCH_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "num_lstm_layers": NUM_LSTM_LAYERS,
        "first_conv_filters": FIRST_CONV_FILTERS,
        "first_conv_kernel_size": FIRST_CONV_KERNEL_SIZE,
        "first_dense_layer": FIRST_DENSE_LAYER,
        "second_dense_layer": SECOND_DENSE_LAYER,
        "cost_func": COST_FUNC,
        "optimizer": OPTIMIZER,
        "output_activation_func": OUTPUT_ACTIVATION_FUNC,
        "hidden_layer_activation_func": HIDDEN_LAYER_ACTIVATION_FUNC,
        # "training/validation/test split": f"{int(TRAINING_SPLIT * 100)}/{int(VALIDATION_SPLIT* 100)}/{int(TEST_SPLIT * 100)}",
        "patience_threshold": PATIENCE_THRESHOLD,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
    }

    result = {
        "test_accuracy": test_accuracy,
        "epochs_actually_trained": epoch_length,
        # "num_trainable_vars": int(np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])),
        "secs_to_train": secs_to_train,
        "parameters": parameters,
        # "history": history,
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
    ax.set_xlabel('Training Epoch')  # Add an x-label to the axes.
    ax.set_ylabel('Epochs Since Better Loss Discovered')  # Add a y-label to the axes.
    ax.set_title(f'Epochs Since Better Loss Discovered')  # Add a title to the axes.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend();  # Add a legend.
    fig.savefig(os.path.join(RESULTS_PATH, f"{filename}_epochs_loss.png"))

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
def generate_input_output_pairs(sequence, kmer_length):

    # Extract all input_output pairs from all sequences
    input_output_pairs = []
    for seq in sequence:
        for start in range(len(seq)-kmer_length):
            end = start + kmer_length
            seq_in = seq[start:end]
            seq_out = seq[end]
            input_output_pairs.append((seq_in, seq_out))

            # FIXME: Testing whether it's better to add reversed data?
            seq_in = seq[end:start:-1]
            seq_out = seq[start]
            input_output_pairs.append((seq_in, seq_out))

    return input_output_pairs

# =============================================================================
def preprocess_data(dataset, char_to_int):
    """
    Preprocesses raw dataset and returns tuple (dataX, dataY)
    """

    # First, convert the raw strings to integers
    input_as_lst = []
    output_as_lst = []
    for inp, out in dataset:
        input_as_lst.append([char_to_int[c] for c in inp])
        output_as_lst.append(char_to_int[out])

    # reshape X to be [samples, time steps, features], normalize
    dataX = np.reshape(input_as_lst, (len(input_as_lst), KMER_LENGTH, 1))
    dataX = dataX / float(NUM_CLASSES)

    # Convert output to categorical vector
    dataY = np_utils.to_categorical(output_as_lst, num_classes=NUM_CLASSES)

    return dataX, dataY

# =============================================================================
def predict_gaps(seq, model, kmer_length, char_to_int_mapping, int_to_char_mapping, gap_char="-"):

    # Convert to mutable object, e.g. list
    predicted_seq = list(seq)

    for index, output_char in enumerate(predicted_seq):
        if index > kmer_length and output_char == gap_char:

            # FIXME: Since this exactly copies the preprocessing we do on all the elements in our
            # training/validation/testing data, it should be its own function to avoid code dup
            input_seq = predicted_seq[index - kmer_length:index]
            input_seq = np.array([char_to_int_mapping[c] for c in input_seq])
            input_seq = input_seq / float(NUM_CLASSES)
            input_seq = np.reshape(input_seq, (1, KMER_LENGTH, 1))

            # Our output array is a probability distribution since we use softmax activation
            # So, we have to acquire the largest probability to determine the class
            output_arr = model.predict(input_seq).flatten()
            highest_probability_index = np.where(output_arr == np.amax(output_arr))[0][0]

            # Convert that integer back into the predicted character
            predicted_char = int_to_char_mapping[highest_probability_index]
            predicted_seq[index] = predicted_char

    # Convert back to a single string when finished
    return functools.reduce(lambda a, b: a+b, predicted_seq)

# =============================================================================
def predict_gaps_both_directions(seq, model, kmer_length, char_to_int_mapping, int_to_char_mapping, gap_char="-"):

    forward_seq = predict_gaps(seq, model, kmer_length, char_to_int_mapping, int_to_char_mapping, gap_char)
    reverse_seq = predict_gaps(seq[::-1], model, kmer_length, char_to_int_mapping, int_to_char_mapping, gap_char)

    full_seq = []
    for forward_pred, reverse_pred in zip(forward_seq, reverse_seq[::-1]):
        # FIXME: I guess just default to the forward prediction unless missing? IDK
        if forward_pred == gap_char:
            full_seq.append(reverse_pred)
        else:
            full_seq.append(forward_pred)

    # Convert back to a single string when finished
    return functools.reduce(lambda a, b: a+b, full_seq)

# =============================================================================
def get_nonmatching_indices(seq1, seq2):
    s = set()
    for i, (c1, c2) in enumerate(zip(seq1, seq2)):
        if c1 != c2:
            s.add(i)
    return s

# =============================================================================
def highlight_indices(seq, indices, color):
    seq = list(seq)
    for i, char in enumerate(seq):
        if i in indices:
            seq[i] = f"{color}{char}{bcolors.ENDC}"

    return functools.reduce(lambda a, b: a+b, seq)

# =============================================================================
def print_sequence(seq, header=None, indices_to_highlight=None):

    line_length = 40
    group_length = 10

    if header:
        print(header)
    print("=====================================================================")

    lines = [seq[begin:begin+line_length] for begin in range(0, len(seq), line_length)]
    for line_num, line in enumerate(lines):

        groups = [line[begin:begin+group_length] for begin in range(0, len(line), group_length)]
        if indices_to_highlight:

            # FIXME: I hate this shit so much. Figure out a more elegant solution
            for group_num, group in enumerate(groups):
                start_group_index = line_num * line_length + group_num * group_length
                group_indices = range(start_group_index, start_group_index + group_length)
                missing_group_indices = indices_to_highlight.intersection(group_indices)
                missing_group_indices = set([item - start_group_index for item in missing_group_indices])

                groups[group_num] = highlight_indices(group, missing_group_indices, bcolors.FAIL)

        print("\t".join(groups))

# =============================================================================
def get_sequences(fasta_file):
    sequences = []
    with open(fasta_file, "r") as input_file:
        sequences = [seq.split("\n") for seq in input_file.read().split(">") if seq]
        sequences = ["".join(parts).strip() for _, *parts in sequences]

    return sequences

# =============================================================================
def compute_accuracy(seq1, seq2, ignore_first=None):
    '''
    Very simple way of measuring accuracy according to matching indices
    Assumes seq1 and seq2 have the same length
    ignore_first optional argument to ignore comparing first N characters
    '''
    if len(seq1) != len(seq2):
        raise Exception("Dumbass, they need to be the same length!")

    if ignore_first:
        seq1 = seq1[ignore_first:]
        seq2 = seq2[ignore_first:]
    return sum(1 if c1 == c2 else 0 for c1, c2 in zip(seq1, seq2)) / len(seq1)

# =============================================================================
def main():

    training_sequences = get_sequences("training_sequences_10.txt")
    target_sequence = get_sequences("target_sequence.txt")[0]
    target_sequence_reversed = target_sequence[::-1]

    # extract all chars from all sequences to create our mappings and to determine classes
    all_chars = set("".join(training_sequences) + target_sequence)
    char_to_int = {c: i for i, c in enumerate(all_chars)}
    int_to_char = {v: k for k, v in char_to_int.items()}

    # Number of classes is based on the data, so update at runtime
    global NUM_CLASSES
    NUM_CLASSES = len(all_chars)

    training_pairs = generate_input_output_pairs(training_sequences, KMER_LENGTH)
    testing_pairs = generate_input_output_pairs([target_sequence], KMER_LENGTH)
    testing_pairs_reversed = generate_input_output_pairs([target_sequence_reversed], KMER_LENGTH)

    # Shuffle the training data so no bias is introduced when splitting for validation
    np.random.shuffle(training_pairs)

    # Determine indices to use to split randomized data into training/validation/test sets
    validation_threshold = int(VALIDATION_PERCENT * len(training_pairs))

    # Convert lists of lists to appropriate data structure complete with any necessary preprocessing
    trainX, trainY = preprocess_data(training_pairs[validation_threshold:], char_to_int)
    validX, validY = preprocess_data(training_pairs[:validation_threshold], char_to_int)
    testX, testY = preprocess_data(testing_pairs, char_to_int)
    testX_rev, testY_rev = preprocess_data(testing_pairs_reversed, char_to_int)

    print(trainX.shape)
    print(trainY.shape)
    # create the model
    # model = Sequential()
    # model.add(Conv1D(FIRST_CONV_FILTERS, FIRST_CONV_KERNEL_SIZE))
    # model.add(Bidirectional(LSTM(NUM_LSTM_LAYERS, return_sequences=True), input_shape=(trainX.shape[1], trainX.shape[2])))
    # # model.add(LSTM(NUM_LSTM_LAYERS, input_shape=(trainX.shape[1], trainX.shape[2])))
    # model.add(Dense(FIRST_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC))
    # model.add(Dense(SECOND_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC))
    # model.add(Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION_FUNC))
    # model.compile(loss=COST_FUNC, optimizer=OPTIMIZER, metrics=['accuracy'])

    # FIXME: Is the right way to train on just the forward data, or to create the
    # reverse data and train on that? If I don't create the reverse data, the bidirectional
    # does not perform well predicting reversed sequences. But if I do train on the reversed
    # Data, the accuracy is high regardless of what tweaks I make to the model.
    # I.e. the bidirectional does not seem to help much

    inputs = keras.Input(shape=(KMER_LENGTH, 1))
    outputs = layers.LSTM(NUM_LSTM_LAYERS)(inputs)
    # outputs = layers.Conv1D(FIRST_CONV_FILTERS, FIRST_CONV_KERNEL_SIZE)(inputs)
    # outputs = layers.LSTM(NUM_LSTM_LAYERS)(outputs)
    # outputs = layers.Bidirectional(layers.LSTM(NUM_LSTM_LAYERS, return_sequences=True))(outputs)
    # outputs = layers.Bidirectional(layers.LSTM(NUM_LSTM_LAYERS))(outputs)
    # outputs = layers.Dense(FIRST_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC)(outputs)
    # outputs = layers.Dense(SECOND_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC)(outputs)
    outputs = layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION_FUNC)(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=COST_FUNC, optimizer=OPTIMIZER, metrics=['accuracy'])

    # fit the data, summarize performance of the model
    start = time.time()
    history = model.fit(
        trainX,
        trainY,
        epochs=MAX_EPOCH_LENGTH,
        batch_size=BATCH_SIZE,
        verbose=2,
        validation_data=(validX, validY),
        callbacks = [EarlyStopping(monitor="val_loss", patience=PATIENCE_THRESHOLD), ReportPatience()],
    )

    end = time.time()
    secs_to_train = int(end - start)

    model.summary()
    _, accuracy = model.evaluate(testX, testY, verbose=0)
    print(f"accuracy: {accuracy:.4f}")
    # save_result(f"{accuracy:.2f}", history.history, model, secs_to_train)

    _, accuracy = model.evaluate(testX_rev, testY_rev, verbose=0)
    print(f"accuracy in reverse direction: {accuracy:.4f}")

    # For funsies, use the model to predict the gaps in the de novo sequence
    # Make sure target sequence is not in training data
    missing_indices = set([0, 1, 2, 24, 25, 26, 62, 63, 64, 66, 67, 68, 69, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 209, 210, 211, 212, 213])

    # The reversed string has missing indices which are the complement of the forward direction
    missing_indices_reversed = set([len(target_sequence) - 1 - i for i in missing_indices])

    de_novo_sequence = "".join(c if i not in missing_indices else "-" for i, c in enumerate(target_sequence))
    de_novo_sequence_reversed = de_novo_sequence[::-1]

    print(missing_indices)
    print(missing_indices_reversed)

    pred_sequence_forward = predict_gaps(de_novo_sequence, model, KMER_LENGTH, char_to_int, int_to_char, gap_char="-")
    pred_sequence_reverse = predict_gaps(de_novo_sequence_reversed, model, KMER_LENGTH, char_to_int, int_to_char, gap_char="-")
    pred_sequence_full = predict_gaps_both_directions(de_novo_sequence, model, KMER_LENGTH, char_to_int, int_to_char, gap_char="-")

    forward_incorrect = get_nonmatching_indices(target_sequence, pred_sequence_forward)
    reverse_incorrect = get_nonmatching_indices(target_sequence_reversed, pred_sequence_reverse)
    full_incorrect = get_nonmatching_indices(target_sequence, pred_sequence_full)

    print("Forward Incorrect: {}".format(forward_incorrect))
    print("Reverse Incorrect: {}".format(reverse_incorrect))
    print("Full Incorrect: {}".format(full_incorrect))

    print_sequence(target_sequence, "TARGET SEQUENCE")
    print_sequence(de_novo_sequence,"DE NOVO SEQUENCE", missing_indices)
    print_sequence(pred_sequence_full, "PREDICTED SEQUENCE FULL", full_incorrect)
    print_sequence(pred_sequence_forward, "PREDICTED SEQUENCE FORWARD", forward_incorrect)

    print_sequence(target_sequence_reversed, "TARGET SEQUENCE REVERSE")
    print_sequence(de_novo_sequence_reversed,"DE NOVO SEQUENCE REVERSE", missing_indices_reversed)
    print_sequence(pred_sequence_reverse, "PREDICTED SEQUENCE REVERSE", reverse_incorrect)

    # FIXME: Now that I'm taking the effor to tget the nonmatching indices, just use that for accuracy
    # forward_accuracy = compute_accuracy(pred_sequence_forward, target_sequence, ignore_first=KMER_LENGTH)
    # reverse_accuracy = compute_accuracy(pred_sequence_reverse, target_sequence[::-1], ignore_first=KMER_LENGTH)
    # full_accuracy = compute_accuracy(target_sequence, pred_sequence_full)

    target_len = len(target_sequence)
    forward_accuracy = (target_len - len(forward_incorrect)) / target_len
    reverse_accuracy = (target_len - len(reverse_incorrect)) / target_len
    full_accuracy = (target_len - len(full_incorrect)) / target_len

    print(f"Accuracy on De Novo Sequence in forward direction: {forward_accuracy}")
    print(f"Accuracy on De Novo Sequence in reverse direction: {reverse_accuracy}")
    print(f"Accuracy on De Novo Sequence when combining both: {full_accuracy}")

if __name__ == "__main__":
    main()
