#!/usr/bin/env python
import hdf5_handler
import data_handler

def get_example(h5, song_id):
    return hdf5_handler.get_segments_pitches(h5, song_id).flatten()[:100]

def get_label(h5, song_id):
    return hdf5_handler.get_year(h5, song_id)

dataset, classes = data_handler.generate_dataset('/Users/justinsvegliato/Downloads/MillionSongSubset/data', 10000, get_example, get_label)
data_handler.save_dataset(dataset)