
import functools
import json
import numpy as np
import os
from datetime import datetime

from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import np_utils
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
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
CHAR_TO_INT = {}
INT_TO_CHAR = {}

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
def generate_input_output_pairs(sequence):

    # Extract all input_output pairs from all sequences
    input_output_pairs = []
    for seq in sequence:
        for start in range(len(seq)-KMER_LENGTH):
            end = start + KMER_LENGTH
            seq_in = seq[start:end]
            seq_out = seq[end]
            input_output_pairs.append((seq_in, seq_out))

    return input_output_pairs

# =============================================================================
def preprocess_data(dataset):
    """
    Preprocesses raw dataset and returns tuple (dataX, dataY)
    """

    # First, convert the raw strings to integers
    input_as_lst = []
    output_as_lst = []
    for inp, out in dataset:
        input_as_lst.append([CHAR_TO_INT[c] for c in inp])
        output_as_lst.append(CHAR_TO_INT[out])

    # reshape X to be [samples, time steps, features], normalize
    dataX = np.reshape(input_as_lst, (len(input_as_lst), KMER_LENGTH, 1))
    dataX = dataX / float(NUM_CLASSES)

    # Convert output to categorical vector
    dataY = np_utils.to_categorical(output_as_lst, num_classes=NUM_CLASSES)

    return dataX, dataY

# =============================================================================
def get_sequence_predictions(model, seq, gap_char):

    # Characters that already exist have a probability of 1. Until gaps are filled, their probability is 0
    predictions_probabilities = [(c, 1 if c != gap_char else 0) for c in seq]

    for start in range(len(seq) - KMER_LENGTH):
        end = start+KMER_LENGTH
        # Only if we have a gap, do we need to update predictions_probabilities
        if seq[end] == gap_char:
            input_seq = [c for c, _ in predictions_probabilities[start:end]]
            input_seq = np.array([CHAR_TO_INT[c] for c in input_seq])
            input_seq = input_seq / float(NUM_CLASSES)
            input_seq = np.reshape(input_seq, (1, KMER_LENGTH, 1))

            output_arr = model.predict(input_seq).flatten()
            highest_probability = np.amax(output_arr)
            output_class = np.where(output_arr == highest_probability)[0][0]

            # Convert the output class integer back into the predicted character
            predicted_char = INT_TO_CHAR[output_class]
            predictions_probabilities[end] = (predicted_char, highest_probability)

    return predictions_probabilities

# =============================================================================
def predict_gaps(seq, forward_model, reverse_model, gap_char="-"):

    forward_preds = get_sequence_predictions(forward_model, seq, gap_char)
    reverse_preds = get_sequence_predictions(reverse_model, seq[::-1], gap_char)

    predicted_seq = ""
    for ((forward_pred, forward_prob), (reverse_pred, reverse_prob)) in zip(forward_preds, reverse_preds[::-1]):
        best_prediction = forward_pred if forward_prob >= reverse_prob else reverse_pred
        predicted_seq += best_prediction

    return predicted_seq

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
            seq[i] = f"{bcolors.BOLD}{color}{char}{bcolors.ENDC}"

    return functools.reduce(lambda a, b: a+b, seq)

# =============================================================================
def print_sequence(seq, header=None, incorrect_indices=None, correct_indices=None):

    line_length = 40
    group_length = 10

    if header:
        print(header)
    print("=====================================================================")

    lines = [seq[begin:begin+line_length] for begin in range(0, len(seq), line_length)]
    for line_num, line in enumerate(lines):

        groups = [line[begin:begin+group_length] for begin in range(0, len(line), group_length)]

        # FIXME: I hate this shit so much. Figure out a more elegant solution
        for group_num, group in enumerate(groups):
            start_group_index = line_num * line_length + group_num * group_length
            group_indices = range(start_group_index, start_group_index + group_length)

            if incorrect_indices:
                fail_indices = incorrect_indices.intersection(group_indices)
                fail_indices = set([item - start_group_index for item in fail_indices])
                group = highlight_indices(group, fail_indices, bcolors.FAIL)

            if correct_indices:
                success_indices = correct_indices.intersection(group_indices)
                success_indices = set([item - start_group_index for item in success_indices])
                group = highlight_indices(group, success_indices, bcolors.OKBLUE)

            groups[group_num] = group

        print("\t".join(groups))

# =============================================================================
def get_sequences(fasta_file):
    sequences = []
    with open(fasta_file, "r") as input_file:
        sequences = [seq.split("\n") for seq in input_file.read().split(">") if seq]
        sequences = ["".join(parts).strip() for _, *parts in sequences]

    return sequences

# =============================================================================
def build_model(training_seqs):

    training_pairs = generate_input_output_pairs(training_seqs)

    # Shuffle the training data so no bias is introduced when splitting for validation
    np.random.shuffle(training_pairs)

    # Determine indices to use to split randomized data into training/validation/test sets
    validation_threshold = int(VALIDATION_PERCENT * len(training_pairs))

    # Convert lists of lists to appropriate data structure complete with any necessary preprocessing
    trainX, trainY = preprocess_data(training_pairs[validation_threshold:])
    validX, validY = preprocess_data(training_pairs[:validation_threshold])

    # Build model
    inputs = keras.Input(shape=(KMER_LENGTH, 1))
    outputs = layers.LSTM(NUM_LSTM_LAYERS)(inputs)
    outputs = layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION_FUNC)(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=COST_FUNC, optimizer=OPTIMIZER, metrics=['accuracy'])

    history = model.fit(
        trainX,
        trainY,
        epochs=MAX_EPOCH_LENGTH,
        batch_size=BATCH_SIZE,
        verbose=2,
        validation_data=(validX, validY),
        callbacks = [EarlyStopping(monitor="val_loss", patience=PATIENCE_THRESHOLD), ReportPatience()],
    )
    # model.summary()
    return model, history

# =============================================================================
def main():

    training_sequences = get_sequences("training_sequences_10.txt")
    training_sequences_reversed = [item[::-1] for item in training_sequences]

    target_sequence = get_sequences("target_sequence.txt")[0]
    target_sequence_reversed = target_sequence[::-1]

    # extract all chars from all sequences to create our mappings and to determine classes
    all_chars = set("".join(training_sequences) + target_sequence)

    # These globals must be determined at runtime
    global NUM_CLASSES, CHAR_TO_INT, INT_TO_CHAR
    NUM_CLASSES = len(all_chars)
    CHAR_TO_INT = {c: i for i, c in enumerate(all_chars)}
    INT_TO_CHAR = {v: k for k, v in CHAR_TO_INT.items()}

    forward_model, _ = build_model(training_sequences)
    reverse_model, _ = build_model(training_sequences_reversed)

    # Forward model is trained on forward data, tested on forward data
    testing_pairs = generate_input_output_pairs([target_sequence])
    testX, testY = preprocess_data(testing_pairs)
    _, accuracy = forward_model.evaluate(testX, testY)
    print(f"Accuracy on Forward Model: {accuracy:.2f}")

    # Reverse model is trained on reverse data, tested on reverse data
    testing_pairs = generate_input_output_pairs([target_sequence_reversed])
    testX, testY = preprocess_data(testing_pairs)
    _, accuracy = reverse_model.evaluate(testX, testY)
    print(f"Accuracy on Reverse Model: {accuracy:.2f}")

    # Now use both models to predict a de novo sequence based on target sequence
    missing_indices = set([0, 1, 2, 24, 25, 26, 62, 63, 64, 66, 67, 68, 69, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 209, 210, 211, 212, 213])
    de_novo_sequence = "".join(c if i not in missing_indices else "-" for i, c in enumerate(target_sequence))

    pred_sequence_full = predict_gaps(de_novo_sequence, forward_model, reverse_model)
    full_incorrect = get_nonmatching_indices(target_sequence, pred_sequence_full)

    # Print the three different sequences for visual inspection
    print_sequence(target_sequence, "TARGET SEQUENCE")
    print_sequence(de_novo_sequence,"DE NOVO SEQUENCE", missing_indices)
    print_sequence(pred_sequence_full, "PREDICTED SEQUENCE FULL", full_incorrect, missing_indices)

    # Compute final accuracy on de novo sequence
    target_len = len(target_sequence)
    full_accuracy = (target_len - len(full_incorrect)) / target_len
    print(f"Accuracy on De Novo Sequence: {full_accuracy}")

if __name__ == "__main__":
    main()
