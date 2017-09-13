import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    counter = len(series) - window_size
    i = 0

    while (i < counter):
        xArray = []
        j = i
        while (j < i + window_size):
            xArray.append(series[j])
            j += 1

        X.append(xArray)
        y.append(series[i + window_size])
        i += 1

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:window_size])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    #layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
    #layer 2 uses a fully connected module with one unit
    #the 'mean_squared_error' loss should be used (remember: we are performing regression here)
    
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))

    model.summary()

    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    to_be_removed = ['/','(','"','&','%','*',')','-','$','@',"'",'è','é','à','â','0','1','2','3','4','5','6','7','8','9']
    for i in range(0,len(to_be_removed)):
        text = text.replace(to_be_removed[i],' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    counter = len(text) - window_size
    i = 0
    limit = window_size

    while(i < counter):
        inputs.append(text[i:i + window_size])
        outputs.append(text[i + window_size])
        i += step_size

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    #layer 1 should be an LSTM module with 200 hidden units --> note this should have input_shape = (window_size,len(chars)) where len(chars) = number of unique characters in your cleaned text
    #layer 2 should be a linear module, fully connected, with len(chars) hidden units --> where len(chars) = number of unique characters in your cleaned text
    #layer 3 should be a softmax activation ( since we are solving a multiclass classification)
    
    # I must import Activation again in my function. Don't reaaly know why...
    from keras.layers import Activation

    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='linear'))
    model.add(Activation('softmax'))

    model.summary()

    return model

