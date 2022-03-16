
import numpy
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D
from keras.utils import np_utils

# fix random seed for reproducibility
numpy.random.seed(7)

# import the data
def get_names_to_sequences(filename):
    with open(filename, "r") as input_file:
        sequences = [seq.split("\n") for seq in input_file.read().split(">") if seq]
        return {name: "".join(parts).strip() for name, *parts in sequences}

names_to_sequences = get_names_to_sequences("alemtuzumab_sequences.txt")

# create mapping of characters to integers and the reverse for all characters in all sequences
all_chars = set("".join(list(names_to_sequences.values())))
char_to_int = {c: i for i, c in enumerate(all_chars)}
int_to_char = {v: k for k, v in char_to_int.items()}

# Model parameters
kmer_length = 10
epoch_length = 5
num_lstm_layers = 256
first_conv_filters = 128
first_conv_kernel_size = 5
first_dense_layer = 128
second_dense_layer = 64
cost_func = "categorical_crossentropy"
optimizer = "adam"
output_activation_func = "softmax"
hidden_layer_activation_func = "relu"

for test_name in names_to_sequences:
    test_seq = names_to_sequences[test_name]
    training_seqs = [names_to_sequences[name] for name in names_to_sequences if name != test_name]

    train_as_lstX = []
    train_as_lstY = []
    test_as_lstX = []
    test_as_lstY = []

    # Iterate over each sequence, starting at kmer_length to ignore first kmer_length characters
    for seq in training_seqs:
        for index in range(kmer_length, len(seq)):
            seq_in = seq[index - kmer_length:index]
            seq_out = seq[index]
            train_as_lstX.append([char_to_int[char] for char in seq_in])
            train_as_lstY.append(char_to_int[seq_out])

    for index in range(kmer_length, len(test_seq)):
        seq_in = test_seq[index - kmer_length:index]
        seq_out = test_seq[index]
        test_as_lstX.append([char_to_int[char] for char in seq_in])
        test_as_lstY.append(char_to_int[seq_out])

    # reshape X to be [samples, time steps, features]
    trainX = numpy.reshape(train_as_lstX, (len(train_as_lstX), kmer_length, 1))
    testX = numpy.reshape(test_as_lstX, (len(test_as_lstX), kmer_length, 1))

    # normalize (leng)
    trainX = trainX / float(len(char_to_int))
    testX = testX / float(len(char_to_int))

    # one hot encode the output variable
    trainY = np_utils.to_categorical(train_as_lstY)
    testY = np_utils.to_categorical(test_as_lstY)

    # create the model
    model = Sequential()
    model.add(Conv1D(first_conv_filters, first_conv_kernel_size))
    model.add(LSTM(num_lstm_layers, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(first_dense_layer, activation=hidden_layer_activation_func))
    model.add(Dense(second_dense_layer, activation=hidden_layer_activation_func))
    model.add(Dense(trainY.shape[1], activation=output_activation_func))
    model.compile(loss=cost_func, optimizer=optimizer, metrics=['accuracy'])

    # fit the data, summarize performance of the model
    model.fit(trainX, trainY, epochs=epoch_length, batch_size=1, verbose=2, validation_data=(testX, testY))
    scores = model.evaluate(trainX, trainY, verbose=0)
    final_accuracy = "%.2f%%" % (scores[1]*100)
    print("Accuracy when testing: %s" % test_name)
    print(final_accuracy)
