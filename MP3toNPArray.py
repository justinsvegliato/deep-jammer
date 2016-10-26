import subprocess as sp
import numpy as np
from scipy.io.wavfile import write


hz = 44100
start_time = 30
end_time = 40


# http://zulko.github.io/blog/2013/10/04/read-and-write-audio-files-in-python-using-ffmpeg/

FFMPEG_BIN = '/Users/samwitty/Desktop/Coursework/ffmpeg'
song_file = '/Users/samwitty/Downloads/WE_ARE_FM_-_Scene_5.mp3'


cmd = [ FFMPEG_BIN,
        '-i', song_file,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', '44100', # ouput will have 44100 Hz
        '-ac', '2', # stereo (set to '1' for mono)
        '-']

p = sp.Popen(cmd, stdout=sp.PIPE, bufsize=10**8)
output = p.communicate()[0]
p.wait()

audio_array = np.fromstring(output, dtype="int16")
audio_array = audio_array.reshape((len(audio_array)/2,2))
audio_array = audio_array[(start_time*hz):(end_time*hz),:]


#print audio_array.shape

# Write audio file from processed np array to demostrate that it is working correctly
write('test.wav', hz, audio_array)



