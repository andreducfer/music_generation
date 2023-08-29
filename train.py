import keras as keras
import json
from preprocess import generate_training_sequences, SEQUENCE_LENGTH, MAPPING_PATH

NUM_UNITS_HIDDEN_LAYER = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.keras"


def output_layer_size():
    with open(MAPPING_PATH, "r") as fp:
        mapping = json.load(fp)

    return len(mapping)


def build_model(output_units, num_units, loss, learning_rate):
    # create the model architecture
    input_layer = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input_layer)
    x = keras.layers.Dropout(0.2)(x)

    output_layer = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input_layer, output_layer)

    # compile model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    model.summary()

    return model


def train(num_units=NUM_UNITS_HIDDEN_LAYER, loss=LOSS, learning_rate=LEARNING_RATE):
    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    output_units = output_layer_size()

    # build the network
    model = build_model(output_units, num_units, loss, learning_rate)

    # train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()
