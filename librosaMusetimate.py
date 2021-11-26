
import librosa
import librosa.display
import numpy as np

# Codification:
# 0 - anger
# 1 - calm
# 2 - happiness
# 3 - sadness

# Prepares all data from the database folder
def get_data():
    # Get files from the first folder
    files = librosa.util.find_files('C:\\Users\\fightglory\\Desktop\\db\\anger')
    # Prepare final array (lister)
    lister = np.zeros((len(files) * 4, 5))
    # Assign the first label
    lister[0:12, 4] = 0
    j = 0
    for i in files:
        y, sr = librosa.load(i)
        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        dtempo = np.std(dtempo)
        print(i.rpartition('db')[-1])
        print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
        print("\n")
        lister[j,0] = tempo
        lister[j,1] = dtempo
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        times = librosa.times_like(f0)
        lister[j,2] = np.mean(times)
        lister[j,3] = np.std(times)
        j += 1
    print(lister)
    files = librosa.util.find_files('C:\\Users\\fightglory\\Desktop\\db\\calm')
    lister[12:24, 4] = 1
    for i in files:
        y, sr = librosa.load(i)
        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        dtempo = np.std(dtempo)
        print(i.rpartition('db')[-1])
        print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
        print("\n")
        lister[j,0] = tempo
        lister[j,1] = dtempo
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        times = librosa.times_like(f0)
        lister[j, 2] = np.mean(times)
        lister[j, 3] = np.std(times)
        j += 1
    print(lister)
    files = librosa.util.find_files('C:\\Users\\fightglory\\Desktop\\db\\happiness')
    lister[24:36, 4] = 2
    for i in files:
        y, sr = librosa.load(i)
        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        dtempo = np.std(dtempo)
        print(i.rpartition('db')[-1])
        print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
        print("\n")
        lister[j,0] = tempo
        lister[j,1] = dtempo
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        times = librosa.times_like(f0)
        lister[j, 2] = np.mean(times)
        lister[j, 3] = np.std(times)
        j += 1
    print(lister)
    files = librosa.util.find_files('C:\\Users\\fightglory\\Desktop\\db\\sadness')
    lister[36:48, 4] = 3
    for i in files:
        y, sr = librosa.load(i)
        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        dtempo = np.std(dtempo)
        print(i.rpartition('db')[-1])
        print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
        print("\n")
        lister[j,0] = tempo
        lister[j,1] = dtempo
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        times = librosa.times_like(f0)
        lister[j, 2] = np.mean(times)
        lister[j, 3] = np.std(times)
        j += 1
    print(lister)
    return lister


def singler(path, feeling):
    y, sr = librosa.load(path)
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    print("For "+path+" , analyzing tempo.")
    dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
    dtempo = np.std(dtempo)
    print("Tempo: "+str(tempo))
    print("DTempo: "+str(dtempo))
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0)
    f0avg = np.mean(times)
    f0std = np.std(times)
    return [tempo, dtempo, f0avg, f0std]


