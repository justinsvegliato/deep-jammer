from mido import MidiFile, MetaMessage

mid = MidiFile('example.mid')

for i, track in enumerate(mid.tracks):
    types = {}
    for message in track:
        types[message.type] = message

    print types
    for type in types:
        print type, types[type]

    print types['control_change']