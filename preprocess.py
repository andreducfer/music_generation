import os
import json
import music21 as m21
import numpy as np
import keras as keras

KERN_DATASET_PATH = "data/deutsch/erk"
PREPROCESSED_DATASET_PATH = "data/preprocessed_dataset"
SINGLE_FILE_DATASET_PATH = "data/preprocessed_dataset_single_file/dataset"
MAPPING_PATH = "data/preprocessed_dataset_single_file/mapping.json"
ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]
SEQUENCE_LENGTH = 64

us = m21.environment.UserSettings()
us["musicxmlPath"] = "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"


def load_songs_in_kern(dataset_path):
    # load all files of the dataset in music21 format
    songs = []

    for path, _, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song_to_append = m21.converter.parse(os.path.join(path, file))
                songs.append(song_to_append)

    return songs


def has_acceptable_durations(song_to_verify, acceptable_durations):
    for note in song_to_verify.flatten().notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False

    return True


def transpose(song):
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    first_measure = parts[0].getElementsByClass(m21.stream.Measure)
    key = first_measure[0][4]

    # estimate key if necessary
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # get interval for transposition
    interval = m21.interval
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    transposed_song = song.transpose(interval)

    return transposed_song


def encode_song(song_to_encode, time_step=0.25):
    encoded_song = []

    for event in song_to_encode.flatten().notesAndRests:
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        else:
            continue

        # convert note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def preprocess(dataset_path):
    # load the folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):
        # filter out songs that have non-acceptable duration
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # transpose songs to C major/A minor
        song = transpose(song)

        # encode songs with time series representation
        encoded_song = encode_song(song)

        # save songs to text files
        save_path = os.path.join(PREPROCESSED_DATASET_PATH, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)
        fp.close()


def load_preprocessed_song(song_path):
    with open(song_path, "r") as fp:
        song = fp.read()
    fp.close()

    return song


def create_single_file_dataset(files_path, single_file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    single_file_songs = ""

    # load encoded songs and add delimiters
    for path, _, files in os.walk(files_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load_preprocessed_song(file_path)
            single_file_songs = single_file_songs + song + " " + new_song_delimiter

    single_file_songs = single_file_songs[:-1]

    # save string that contains all the dataset
    with open(single_file_dataset_path, "w") as fp:
        fp.write(single_file_songs)
    fp.close()

    return single_file_songs


def create_mapping(songs, mapping_path):
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save the vocabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)
    fp.close()


def convert_songs_to_int(songs):
    int_songs_list = []

    # load mappings
    with open(MAPPING_PATH, "r") as fp:
        mapping = json.load(fp)

    # cast songs string to a list
    songs_list = songs.split()

    # map songs to int
    for symbol in songs_list:
        int_songs_list.append(mapping[symbol])

    return int_songs_list


def generate_training_sequences(sequence_length):
    inputs = []
    targets = []

    # load songs and map them to int
    songs = load_preprocessed_song(SINGLE_FILE_DATASET_PATH)
    int_songs = convert_songs_to_int(songs)

    # generate the training sequences
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # one-hot encode the sequences
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets


def main():
    preprocess(KERN_DATASET_PATH)
    songs_single_file = create_single_file_dataset(PREPROCESSED_DATASET_PATH, SINGLE_FILE_DATASET_PATH, SEQUENCE_LENGTH)
    create_mapping(songs_single_file, MAPPING_PATH)
    # inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()
