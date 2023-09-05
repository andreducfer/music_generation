import json
import numpy as np
import keras as keras
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH


class MelodyGenerator:
    def __init__(self, model_path="model.keras"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        self._start_symbols = ["/"] * SEQUENCE_LENGTH

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)
        fp.close()

    def generate_melody(self, start_melody, num_steps, max_sequence_length, temperature):
        # create the seed to feed the network
        melody = start_melody.split()
        seed_initial = start_melody.split()
        seed_initial = self._start_symbols + seed_initial

        # map seed to int
        seed_int = [self._mappings[symbol] for symbol in seed_initial]

        for _ in range(num_steps):
            # limit the seed to max_sequence_length
            seed_int = seed_int[-max_sequence_length:]

            # one-hot encode the seed_int
            seed_onehot = keras.utils.to_categorical(seed_int, num_classes=len(self._mappings))
            # (1, max_sequence_length, number of symbols in the vocabulary)
            seed_onehot = seed_onehot[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(seed_onehot)[0]
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed_int.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check whether we are at the end of a melody
            if output_symbol == "/":
                break

            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilities, temperature):
        # closer to 0 more rigid / bigger more unpredictable
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self, melody, step_duration=0.25, file_format="midi", file_name="generated_melody.mid"):
        # create a music21 stream
        stream = m21.stream.Stream()

        # parse all symbols in the melody and create note/rest objects
        # 60 _ _ _ r _ 55 _
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            # handle case where we have a note/rest
            if symbol != "_" or i + 1 == len(melody):
                # ensure not dealing with the first element of array
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step_counter
                    step_counter = 1

                start_symbol = symbol
            # handle case where we have a prolongation sign "_"
            else:
                step_counter += 1

        # write de music21 stream to a midi file
        stream.write(file_format, file_name)


if __name__ == "__main__":
    melody_generator = MelodyGenerator()
    seed = "55 _ _ _ 60 _ _ _ 60 _ _ _ 64 _ _ _"
    generated_melody = melody_generator.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.7)
    print(generated_melody)

    melody_generator.save_melody(generated_melody)
