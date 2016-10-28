#!/usr/bin/env python
import hdf5_handler
import data_handler

def get_example(h5, song_id):
    return hdf5_handler.get_segments_pitches(h5).flatten()[:100]

def get_label(h5, song_id):
    return hdf5_handler.get_artist_name(h5)

dataset, classes = data_handler.generate_dataset('/Users/justinsvegliato/Downloads/MillionSongSubset/data', 5000, get_example, get_label)
data_handler.save_dataset(dataset)

print 'Saved dataset with %d classes' % len(classes)
