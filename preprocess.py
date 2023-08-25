import os
import music21 as m21

KERN_DATASET_PATH = "data/deutschl/test"
SAVE_DIR = "data/preprocessed_dataset"
ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]

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
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


if __name__ == "__main__":
    # songs_list = load_songs_in_kern(KERN_DATASET_PATH)
    # print(f"Loaded {len(songs_list)} songs.")
    # song_test = songs_list[0]

    # transposed_song_test = transpose(song_test)
    # transposed_song_test.show()

    preprocess(KERN_DATASET_PATH)
