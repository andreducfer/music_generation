import os
import music21 as m21

KERN_DATASET_PATH = "data/deutschl/test"
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


def preprocess(dataset_path):
    # load the folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for song in songs:
        # filter out songs that have non-acceptable duration
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # transpose songs to C major/A minor
        song = transpose(song)

        # encode songs with time series representation

        # save songs to text files


if __name__ == "__main__":
    songs_list = load_songs_in_kern(KERN_DATASET_PATH)
    song_test = songs_list[0]

    print(f"Loaded {len(songs_list)} songs.")
    print(f"Has acceptable duration? {has_acceptable_durations(song_test, ACCEPTABLE_DURATIONS)}")

    # song_test.show()
    # transposed_song_test = transpose(song_test)
    # transposed_song_test.show()
