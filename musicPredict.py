from librosaMusetimate import singler
import keras
import numpy as np
# 626 architecture: 8 (relu), 8 (relu), 4 (sigmoid) - Codename STITCH
# 895 architecture: 15 (relu), 30 (sigmoid), 4 (softmax) - Codename SPARKY
# --- ID ---  |  --------  Name -------  | label (626 vs 895) |  Expected |
# =========================================================================
# Test song   = Down with the sickness,  | anger \ happiness  |   anger   |
# Test song 1 = Traitor,                 | sad   \ happiness  |    sad    |
# Test song 2 = Lay all your love on me  | anger \ calm       |   anger   |
# Test song 3 = Super Trouper,           | calm  \ calm       |    calm   |
# Test song 4 = Nuestra cancion          | anger \ anger      |     sad   |
# Test song 5 = Now we are free          | anger \ happy      |    calm   |
# Test song 6 = Don't go breaking my hea | anger \ anger      |   happy   |
# Test song 7 = Sarek                    | happy \ calm       |    calm   |
# Test song 8 = Mi primer millon         | sad   \ anger      |   happy   |
# Test song 9 = Peace                    | sad   \  sad       |    calm   |
# Test song 10= Beast in black           | anger \  happy     |    anger  |
# Test song 11= Hey, soul sister         | anger \  anger     |   happy   |
# =========================================================================


model = keras.models.load_model('musicModel626.h5')
model2 = keras.models.load_model('musicModel825.h5')
topredict = singler('test_song11.mp3',0)
topredict = np.expand_dims(topredict,axis=0)
a = model.predict(topredict)
b = model2.predict(topredict)
classes = a.argmax(axis=-1)
classes2= b.argmax(axis=-1)
print("Model 626: "+str(classes))
print("Model 895: "+str(classes2))