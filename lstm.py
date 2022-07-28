
import functools
import json
import os
from datetime import datetime
from termcolor import colored

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers
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

# Run 38 for bilstm was a really good run
# Run 1, 8 are decent for lstm

KMER_LENGTH = 5
MAX_EPOCH_LENGTH = 200
NUM_LSTM_LAYERS = 256
NUM_FILTERS = 128
FILTER_SIZE = 3
FIRST_DENSE_LAYER = 128
SECOND_DENSE_LAYER = 64
COST_FUNC = "categorical_crossentropy"
OPTIMIZER = "adam"
OUTPUT_ACTIVATION_FUNC = "softmax"
HIDDEN_LAYER_ACTIVATION_FUNC = "relu"
VALIDATION_PERCENT = 0.1
BATCH_SIZE = 64
# L2_PENALTY = 0.01 # too large
# L2_PENALTY = 0.0075 $ too large
# L2_PENALTY = 0.005 # too large
L2_PENALTY = 0.003 # too large
# L2_PENALTY = 0.001 # okay
# L2_PENALTY = 0.002 # try this next
# L2_PENALTY = (0.0050 + 0.0075) / 2
# L2_MIN = 0.0
# L2_MAX = 1.0E-1
# L2_NUM_ITERATIONS = 5

NUM_CLASSES = -1
CHAR_TO_INT = {}
INT_TO_CHAR = {}

# =============================================================================
# FUNCTIONS START HERE
# =============================================================================

# =============================================================================
def snake_case_prettify(s):
    return " ".join(w.capitalize() for w in s.split("_"))

# =============================================================================
def get_run_number() -> int:
    run_number_file = os.path.join(RESULTS_PATH, "run_number.txt")
    if not os.path.exists(run_number_file):
        with open(run_number_file, "w+") as f:
            f.write(json.dumps({"run_number": 1}))

    with open(run_number_file, "r") as f:
        return json.loads(f.read())["run_number"]

# =============================================================================
def update_run_rumber():
    run_number_file = os.path.join(RESULTS_PATH, "run_number.txt")
    with open(run_number_file, "r") as f:
        run_number = int(json.loads(f.read())["run_number"]) + 1

    with open(run_number_file, "w+") as f:
        f.write(json.dumps({"run_number": run_number}))

# =============================================================================
def mkdir_if_not_exists(path):
    try:
        os.mkdir(os.path.join(path))
    except FileExistsError:
        pass

# =============================================================================
# Save what we want from the results
def save_result(basedir, model_type, model_name, test_accuracy, history, model):

    subdir = os.path.join(basedir, model_type)
    mkdir_if_not_exists(subdir)

    save_metric_plot(subdir, model_name, "loss", history["loss"], history["val_loss"])
    save_metric_plot(subdir, model_name, "categorical_accuracy", history["categorical_accuracy"], history["val_categorical_accuracy"])

    parameters = {
        "kmer_length": KMER_LENGTH,
        "max_epoch_length": MAX_EPOCH_LENGTH,
        "first_dense_layer": FIRST_DENSE_LAYER,
        "second_dense_layer": SECOND_DENSE_LAYER,
        "num_filters": NUM_FILTERS,
        "filter_size": FILTER_SIZE,
        "num_lstm_layers": NUM_LSTM_LAYERS,
        "cost_func": COST_FUNC,
        "optimizer": OPTIMIZER,
        "output_activation_func": OUTPUT_ACTIVATION_FUNC,
        "hidden_layer_activation_func": HIDDEN_LAYER_ACTIVATION_FUNC,
        "validation_percent": VALIDATION_PERCENT,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "l2_penalty": L2_PENALTY,
    }

    result = {
        "test_accuracy": test_accuracy,
        "val_categorical_accuracy": history["val_categorical_accuracy"][-1],
        "categorical_accuracy": history["categorical_accuracy"][-1],
        "loss": history["loss"][-1],
        "val_loss": history["val_loss"][-1],
        "epochs": MAX_EPOCH_LENGTH,
        "parameters": parameters,
    }

    path = os.path.join(RESULTS_PATH, subdir, "results.json")
    with open(path, "w+") as f:
        f.write(json.dumps(result))

    path = os.path.join(RESULTS_PATH, subdir, "model_summary.txt")
    with open(path, "w+") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    update_run_rumber()

# =============================================================================
def save_metric_plot(filedir, model_name, metric, trainX, validX):
    metric_prettify = snake_case_prettify(metric)
    title = "{} for {}".format(metric_prettify, model_name)
    fig, ax = plt.subplots()
    epoch_axis = list(range(1, MAX_EPOCH_LENGTH + 1))
    ax.plot(epoch_axis, trainX, label='Training')
    ax.plot(epoch_axis, validX, label='Validation')
    ax.set_xlabel('Epochs')         # Add an x-label to the axes.
    ax.set_ylabel(metric_prettify)  # Add a y-label to the axes.
    ax.set_title(title)             # Add a title to the axes.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend();  # Add a legend.
    fig.savefig(os.path.join(RESULTS_PATH, filedir, f"{metric}.png"))

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

    if not model:
        return predictions_probabilities

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
def predict_gaps(seq, forward_model=None, reverse_model=None, gap_char="-"):

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
            seq[i] = colored(char, color=color, attrs=['bold'])

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
                group = highlight_indices(group, fail_indices, color='red')

            if correct_indices:
                success_indices = correct_indices.intersection(group_indices)
                success_indices = set([item - start_group_index for item in success_indices])
                group = highlight_indices(group, success_indices, color='green')

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
def build_lstm_model(l2_penalty):

    l2_regularizer = regularizers.L2(l2_penalty)
    regularizer_kwargs = {
        "kernel_regularizer": l2_regularizer,
        "bias_regularizer": l2_regularizer,
        # "activity_regularizer": l2_regularizer,
    }

    inputs = keras.Input(shape=(KMER_LENGTH, 1))
    outputs = layers.LSTM(NUM_LSTM_LAYERS, **regularizer_kwargs)(inputs)
    outputs = layers.Dense(FIRST_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(SECOND_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    return (keras.Model(inputs=inputs, outputs=outputs), "LSTM")

# =============================================================================
def build_cnn_lstm_model(l2_penalty):

    l2_regularizer = regularizers.L2(l2_penalty)
    regularizer_kwargs = {
        "kernel_regularizer": l2_regularizer,
        "bias_regularizer": l2_regularizer,
        # "activity_regularizer": l2_regularizer,
    }
    inputs = keras.Input(shape=(KMER_LENGTH, 1))
    outputs = layers.Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE, **regularizer_kwargs)(inputs)
    outputs = layers.LSTM(NUM_LSTM_LAYERS, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(FIRST_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(SECOND_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    return (keras.Model(inputs=inputs, outputs=outputs), "CNN LSTM")

# =============================================================================
def build_bilstm_model(l2_penalty):

    l2_regularizer = regularizers.L2(l2_penalty)
    regularizer_kwargs = {
        "kernel_regularizer": l2_regularizer,
        "bias_regularizer": l2_regularizer,
        # "activity_regularizer": l2_regularizer,
    }

    inputs = keras.Input(shape=(KMER_LENGTH, 1))
    outputs = layers.Bidirectional(layers.LSTM(NUM_LSTM_LAYERS, **regularizer_kwargs))(inputs)
    outputs = layers.Dense(FIRST_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(SECOND_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    return (keras.Model(inputs=inputs, outputs=outputs), "Bi-LSTM")

# =============================================================================
def build_cnn_bilstm_model(l2_penalty):

    l2_regularizer = regularizers.L2(l2_penalty)
    regularizer_kwargs = {
        "kernel_regularizer": l2_regularizer,
        "bias_regularizer": l2_regularizer,
        # "activity_regularizer": l2_regularizer,
    }
    inputs = keras.Input(shape=(KMER_LENGTH, 1))
    outputs = layers.Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE, **regularizer_kwargs)(inputs)
    outputs = layers.Bidirectional(layers.LSTM(NUM_LSTM_LAYERS, **regularizer_kwargs))(inputs)
    outputs = layers.Dense(FIRST_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(SECOND_DENSE_LAYER, activation=HIDDEN_LAYER_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    outputs = layers.Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION_FUNC, **regularizer_kwargs)(outputs)
    return (keras.Model(inputs=inputs, outputs=outputs), "CNN Bi-LSTM")

# =============================================================================
def get_train_valid_split(training_seqs):

    training_pairs = generate_input_output_pairs(training_seqs)

    # Shuffle the training data so no bias is introduced when splitting for validation
    np.random.shuffle(training_pairs)

    # Determine indices to use to split randomized data into training/validation/test sets
    validation_threshold = int(VALIDATION_PERCENT * len(training_pairs))

    # Convert lists of lists to appropriate data structure complete with any necessary preprocessing
    trainX, trainY = preprocess_data(training_pairs[validation_threshold:])
    validX, validY = preprocess_data(training_pairs[:validation_threshold])
    
    return trainX, trainY, validX, validY

# =============================================================================
def get_model_builder(model_type):
    model_types = {
        "lstm": build_lstm_model,
        "cnn_lstm": build_cnn_lstm_model,
        "bilstm": build_bilstm_model,
        "cnn_bilstm": build_cnn_bilstm_model,
    }
    if model_type not in model_types:
        raise Exception("Not a valid model type! Pick from {}".format(model_types.keys()))
    return model_types[model_type]

# =============================================================================
def compile_and_fit_model(model, trainX, trainY, validX, validY):
    model.compile(loss=COST_FUNC, optimizer=OPTIMIZER, metrics=['categorical_accuracy'])

    history = model.fit(
        trainX,
        trainY,
        epochs=MAX_EPOCH_LENGTH,
        batch_size=BATCH_SIZE,
        verbose=2,
        validation_data=(validX, validY)
    )
    return history

# =============================================================================
def tune_l2_penalty(filedir, model_type, training_seqs):

    trainX, trainY, validX, validY = get_train_valid_split(training_seqs)
    model_builder = get_model_builder(model_type)

    l2_possibilities = np.array([
        [0.0, 0.0],
        [1.0E-5, 0.0],
        [1.0E-4, 0.0],
        [1.0E-3, 0.0],
        [1.0E-2, 0.0],
        [1.0E-1, 0.0]
    ])
    for row in l2_possibilities:

        model, model_type_name = model_builder(row[0])
        history = compile_and_fit_model(model, trainX, trainY, validX, validY)
        row[1] = history.history["val_loss"][-1]
    
    title = f"Loss versus Lambda for {model_type_name}"
    fig, ax = plt.subplots()

    ax.plot(l2_possibilities[:, 0], l2_possibilities[:, 1], label='L2 Vals')
    ax.set_xlabel('L2 Values')
    ax.set_ylabel('Loss')
    ax.set_title(title)

    ax.legend();  # Add a legend.
    fig.savefig(os.path.join(filedir, "lambda_curve.png"))

    best_lambda, smallest_loss = l2_possibilities[l2_possibilities[:, 0] == np.min(l2_possibilities[:, 0])][0]
    result = {"best_lambda": best_lambda, "smallest_loss": smallest_loss}
    with open(os.path.join(filedir, f"best_lambda.txt"), "w+") as f:
        return f.write(json.dumps(result))

# =============================================================================
def get_l2_penalty(model_type, training_sequences):

    return L2_PENALTY
    # filedir = os.path.join(RESULTS_PATH, f"l2_{model_type}")

    # if not os.path.exists(filedir):
    #     os.mkdir(filedir)

    # filepath = os.path.join(filedir, "best_lambda.txt")
    # # Only tune the l2 norm once, i.e. write the file once
    # if not os.path.exists(filepath):
    #     tune_l2_penalty(filedir, model_type, training_sequences)
    
    # with open(filepath, "r+") as f:
    #     return json.loads(f.read())["best_lambda"]

# =============================================================================
def train_model(model_type, training_seqs, l2_penalty):

    trainX, trainY, validX, validY = get_train_valid_split(training_seqs)
    model_builder = get_model_builder(model_type)
    model, model_type_name = model_builder(l2_penalty)
    history = compile_and_fit_model(model, trainX, trainY, validX, validY)

    return model, history, model_type_name

# =============================================================================
def main():

    mkdir_if_not_exists(RESULTS_PATH)
    run_number = get_run_number()
    basedir = os.path.join(RESULTS_PATH, f"run {run_number}")
    mkdir_if_not_exists(basedir)

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

    model_types = [
        ("lstm", 1.0E-3),
        ("cnn_lstm", 1.0E-3),
        ("bilstm", 1.0E-3),
        ("cnn_bilstm", 1.0E-3)
    ]

    for model_type, l2_penalty in model_types:
        # l2_penalty = get_l2_penalty(model_type, training_sequences)
        forward_model, forward_history, model_name = train_model(model_type, training_sequences, l2_penalty)
        # reverse_model, _, _ = train_model(model_type, training_sequences_reversed, l2_penalty)

        # Forward model is trained on forward data, tested on forward data
        testing_pairs = generate_input_output_pairs([target_sequence])
        testX, testY = preprocess_data(testing_pairs)
        _, forward_accuracy = forward_model.evaluate(testX, testY)
        print(f"Accuracy on Forward Model: {forward_accuracy:.2f}")

        # # Reverse model is trained on reverse data, tested on reverse data
        # testing_pairs = generate_input_output_pairs([target_sequence_reversed])
        # testX, testY = preprocess_data(testing_pairs)
        # _, reverse_accuracy = reverse_model.evaluate(testX, testY)
        # print(f"Accuracy on Reverse Model: {reverse_accuracy:.2f}")

        # RUN HERE
        save_result(basedir, model_type, model_name, forward_accuracy, forward_history.history, forward_model)

        # # Now use both models to predict a de novo sequence based on target sequence
        # missing_indices = set([0, 1, 2, 24, 25, 26, 62, 63, 64, 66, 67, 68, 69, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 209, 210, 211, 212, 213])
        # de_novo_sequence = "".join(c if i not in missing_indices else "-" for i, c in enumerate(target_sequence))

        # pred_sequence_full = predict_gaps(de_novo_sequence, forward_model, reverse_model)
        # incorrect_indices = get_nonmatching_indices(target_sequence, pred_sequence_full)
        # correct_indices = missing_indices.difference(incorrect_indices)

        # # Print the three different sequences for visual inspection
        # print_sequence(target_sequence, "TARGET SEQUENCE")
        # print_sequence(de_novo_sequence,"DE NOVO SEQUENCE", missing_indices)
        # print_sequence(pred_sequence_full, "PREDICTED SEQUENCE", incorrect_indices, correct_indices)

        # # Compute final accuracy on de novo sequence
        # target_len = len(target_sequence)
        # full_accuracy = (target_len - len(incorrect_indices)) / target_len
        # print(f"Accuracy on De Novo Sequence: {full_accuracy}")

        # # Print the predictions in forward and reverse directions as well
        # forward_pred = predict_gaps(de_novo_sequence, forward_model=forward_model, reverse_model=None)
        # incorrect_indices = get_nonmatching_indices(target_sequence, forward_pred)
        # correct_indices = missing_indices.difference(incorrect_indices)
        # print_sequence(forward_pred, "FORWARD PREDICTIONS", incorrect_indices, correct_indices)

        # reverse_pred = predict_gaps(de_novo_sequence, forward_model=None, reverse_model=reverse_model)
        # incorrect_indices = get_nonmatching_indices(target_sequence, reverse_pred)
        # correct_indices = missing_indices.difference(incorrect_indices)
        # print_sequence(reverse_pred, "REVERSE PREDICTIONS", incorrect_indices, correct_indices)

if __name__ == "__main__":
    main()
