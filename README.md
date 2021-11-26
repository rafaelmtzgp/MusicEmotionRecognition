# MusicEmotionRecognition
A simple python project using librosa and Keras to create a small neural network that attempts to identify a song based on ellicited emotion.

## How to use
Make sure you have Keras, Numpy, and Librosa installed!

### Processing music
In order to use the features proposed, these must be extracted from a particular song. Ensure you have FFMPEG and AudioRead installed if you use MP3 files (such as the ones included in the project). 

Place your songs in the respective emotion folders, then change the folder names to your respective path names in librosaMusetimate.py. You may run get_data() to process every song in every folder (returns to the variable lister, you may want to save this as a .csv with numpy), or singler(path, tag) ((Tag is unused at the moment)) to process a single song.

### Training
This project ships with two generated models, 626 and 895. To generate your own model, use musicEvaluator.py. It loads music_data.csv as its default database (it only uses a training set at the moment). Running this script will train a model and save it as musicModelXXX.h5, where XXX is its accuracy (626 = 62.6%). 

### Predicting
To predict a song, you may want to use musicPredict.py. It loads any two models, using singler(path, tag) to preprocess a track, and uses the models to evaluate that track, giving it a predicted category.
The categories are:
- Anger (0)
- Calm (1)
- Happiness (2)
- Sadness (3)
