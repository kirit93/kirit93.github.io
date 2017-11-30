---
layout: post
title: Text Generation with Keras
---

This is the ipython notebook for my tutorial at PyData NYC on generating text using neural networks with Keras in Python.
My github repo has the presentaion and code, you can find it [here](https://github.com/kirit93/PyDataNYC)
<br><br>



```python
import keras
```

    Using TensorFlow backend.



```python
from keras.models import Sequential
```


```python
from keras.layers import LSTM, Dense, Dropout
```


```python
from keras.callbacks import ModelCheckpoint
```


```python
from keras.utils import np_utils
```


```python
import numpy as np
```


```python
SEQ_LENGTH = 100
```

Now that we've imported everything we need form Keras, we're all set to go!

First, we load our data.

What np_utils.to_categorical does


```python
test_x = np.array([1, 2, 0, 4, 3, 7, 10])

# one hot encoding
test_y = np_utils.to_categorical(test_x)
print(test_x)
print(test_y)
```

    [ 1  2  0  4  3  7 10]
    [[ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]]


This functions returns an array of sequences from the input text file and the corresponding output for each sequence encoded as a one-hot vector.

Now we add a function to create our LSTM.


```python
# Using keras functional model
def create_functional_model(n_layers, input_shape, hidden_dim, n_out, **kwargs):
    drop        = kwargs.get('drop_rate', 0.2)
    activ       = kwargs.get('activation', 'softmax')
    mode        = kwargs.get('mode', 'train')
    hidden_dim  = int(hidden_dim)

    inputs      = Input(shape = (input_shape[1], input_shape[2]))
    model       = LSTM(hidden_dim, return_sequences = True)(inputs)
    model       = Dropout(drop)(model)
    model       = Dense(n_out)(model)
```


```python
# Using keras sequential model
def create_model(n_layers, input_shape, hidden_dim, n_out, **kwargs):
    drop        = kwargs.get('drop_rate', 0.2)
    activ       = kwargs.get('activation', 'softmax')
    mode        = kwargs.get('mode', 'train')
    hidden_dim  = int(hidden_dim)
    model       = Sequential()
    flag        = True 

    if n_layers == 1:   
        model.add( LSTM(hidden_dim, input_shape = (input_shape[1], input_shape[2])) )
        if mode == 'train':
            model.add( Dropout(drop) )

    else:
        model.add( LSTM(hidden_dim, input_shape = (input_shape[1], input_shape[2]), return_sequences = True) )
        if mode == 'train':
            model.add( Dropout(drop) )
        for i in range(n_layers - 2):
            model.add( LSTM(hidden_dim, return_sequences = True) )
            if mode == 'train':
                model.add( Dropout(drop) )
        model.add( LSTM(hidden_dim) )

    model.add( Dense(n_out, activation = activ) )

    return model
```

Now we train our model.


```python
def train(model, X, Y, n_epochs, b_size, vocab_size, **kwargs):    
    loss            = kwargs.get('loss', 'categorical_crossentropy')
    opt             = kwargs.get('optimizer', 'adam')
    
    model.compile(loss = loss, optimizer = opt)

    filepath        = "Weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint      = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
    callbacks_list  = [checkpoint]
    X               = X / float(vocab_size)
    model.fit(X, Y, epochs = n_epochs, batch_size = b_size, callbacks = callbacks_list)
```

The fit function will run the input batchwase n_epochs number of times and it will save the weights to a file whenever there is an improvement. This is taken care of through the callback. <br><br>
After the training is done or once you find a loss that you are happy with, you can test how well the model generates text.


```python
def generate_text(model, X, filename, ix_to_char, vocab_size):
    
    # Load the weights from the epoch with the least loss
    model.load_weights(filename)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    start   = np.random.randint(0, len(X) - 1)
    pattern = np.ravel(X[start]).tolist()

    # We seed the model with a random sequence of 100 so it can start predicting
    print ("Seed:")
    print ("\"", ''.join([ix_to_char[value] for value in pattern]), "\"")
    output = []
    for i in range(250):
        x           = np.reshape(pattern, (1, len(pattern), 1))
        x           = x / float(vocab_size)
        prediction  = model.predict(x, verbose = 0)
        index       = np.argmax(prediction)
        result      = index
        output.append(result)
        pattern.append(index)
        pattern = pattern[1 : len(pattern)]

    print("Predictions")
    print ("\"", ''.join([ix_to_char[value] for value in output]), "\"")
```

Now we're ready to either train or test our model.


```python
filename    = 'data/game_of_thrones.txt'
data        = open(filename).read()
data        = data.lower()
# Find all the unique characters
chars       = sorted(list(set(data)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
ix_to_char  = dict((i, c) for i, c in enumerate(chars))
vocab_size  = len(chars)

print("List of unique characters : \n", chars)

print("Number of unique characters : \n", vocab_size)

print("Character to integer mapping : \n", char_to_int)

```

    List of unique characters : 
     ['\n', ' ', '!', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    Number of unique characters : 
     51
    Character to integer mapping : 
     {'\n': 0, ' ': 1, '!': 2, '&': 3, "'": 4, '(': 5, ')': 6, ',': 7, '-': 8, '.': 9, '0': 10, '1': 11, '2': 12, '3': 13, '4': 14, '5': 15, '6': 16, '7': 17, '8': 18, '9': 19, ':': 20, ';': 21, '?': 22, '[': 23, ']': 24, 'a': 25, 'b': 26, 'c': 27, 'd': 28, 'e': 29, 'f': 30, 'g': 31, 'h': 32, 'i': 33, 'j': 34, 'k': 35, 'l': 36, 'm': 37, 'n': 38, 'o': 39, 'p': 40, 'q': 41, 'r': 42, 's': 43, 't': 44, 'u': 45, 'v': 46, 'w': 47, 'x': 48, 'y': 49, 'z': 50}



```python
list_X      = []
list_Y      = []

# Python append is faster than numpy append. Try it!
for i in range(0, len(data) - SEQ_LENGTH, 1):
    seq_in  = data[i : i + SEQ_LENGTH]
    seq_out = data[i + SEQ_LENGTH]
    list_X.append([char_to_int[char] for char in seq_in])
    list_Y.append(char_to_int[seq_out])

n_patterns  = len(list_X)
print("Number of sequences in data set : \n", n_patterns)
print(list_X[0])
print(list_X[1])

```

    Number of sequences in data set : 
     1605865
    [25, 1, 43, 39, 38, 31, 1, 39, 30, 1, 33, 27, 29, 1, 25, 38, 28, 1, 30, 33, 42, 29, 0, 0, 25, 1, 31, 25, 37, 29, 1, 39, 30, 1, 44, 32, 42, 39, 38, 29, 43, 0, 0, 40, 42, 39, 36, 39, 31, 45, 29, 0, 0, 47, 29, 1, 43, 32, 39, 45, 36, 28, 1, 43, 44, 25, 42, 44, 1, 26, 25, 27, 35, 7, 1, 31, 25, 42, 29, 28, 1, 45, 42, 31, 29, 28, 1, 25, 43, 1, 44, 32, 29, 1, 47, 39, 39, 28, 43, 1]
    [1, 43, 39, 38, 31, 1, 39, 30, 1, 33, 27, 29, 1, 25, 38, 28, 1, 30, 33, 42, 29, 0, 0, 25, 1, 31, 25, 37, 29, 1, 39, 30, 1, 44, 32, 42, 39, 38, 29, 43, 0, 0, 40, 42, 39, 36, 39, 31, 45, 29, 0, 0, 47, 29, 1, 43, 32, 39, 45, 36, 28, 1, 43, 44, 25, 42, 44, 1, 26, 25, 27, 35, 7, 1, 31, 25, 42, 29, 28, 1, 45, 42, 31, 29, 28, 1, 25, 43, 1, 44, 32, 29, 1, 47, 39, 39, 28, 43, 1, 26]



```python
X           = np.reshape(list_X, (n_patterns, SEQ_LENGTH, 1)) # (n, 100, 1)
# Encode output as one-hot vector
Y           = np_utils.to_categorical(list_Y)

print(X[0])
print(Y[0])
```

    [[25]
     [ 1]
     [43]
     [39]
     [38]
     [31]
     [ 1]
     [39]
     [30]
     [ 1]
     [33]
     [27]
     [29]
     [ 1]
     [25]
     [38]
     [28]
     [ 1]
     [30]
     [33]
     [42]
     [29]
     [ 0]
     [ 0]
     [25]
     [ 1]
     [31]
     [25]
     [37]
     [29]
     [ 1]
     [39]
     [30]
     [ 1]
     [44]
     [32]
     [42]
     [39]
     [38]
     [29]
     [43]
     [ 0]
     [ 0]
     [40]
     [42]
     [39]
     [36]
     [39]
     [31]
     [45]
     [29]
     [ 0]
     [ 0]
     [47]
     [29]
     [ 1]
     [43]
     [32]
     [39]
     [45]
     [36]
     [28]
     [ 1]
     [43]
     [44]
     [25]
     [42]
     [44]
     [ 1]
     [26]
     [25]
     [27]
     [35]
     [ 7]
     [ 1]
     [31]
     [25]
     [42]
     [29]
     [28]
     [ 1]
     [45]
     [42]
     [31]
     [29]
     [28]
     [ 1]
     [25]
     [43]
     [ 1]
     [44]
     [32]
     [29]
     [ 1]
     [47]
     [39]
     [39]
     [28]
     [43]
     [ 1]]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]



```python
print("Shape of input data ", X.shape, "\nShape of output data ", Y.shape)
```

    Shape of input data  (1605865, 100, 1) 
    Shape of output data  (1605865, 51)



```python
model   = create_model(1, X.shape, 256, Y.shape[1], mode = 'train')
```


```python
train(model, X[:1024], Y[:1024], 2, 512, vocab_size)
```

    Epoch 1/2
     512/1024 [==============>...............] - ETA: 6s - loss: 3.9281Epoch 00000: loss improved from inf to 3.91150, saving model to Weights/weights-improvement-00-3.9115.hdf5
    1024/1024 [==============================] - 12s - loss: 3.9115    
    Epoch 2/2
     512/1024 [==============>...............] - ETA: 5s - loss: 3.8554Epoch 00001: loss improved from 3.91150 to 3.83877, saving model to Weights/weights-improvement-01-3.8388.hdf5
    1024/1024 [==============================] - 10s - loss: 3.8388    



```python
generate_text(model, X, "Weights/weights-improvement-36-1.7693.hdf5", ix_to_char, vocab_size)
```

    Seed:
    " fully.  heartsbane. lord randyll let me hold it a few times, but it always scared me. it was valyria "
    Predictions
    " n soaek of the tooes of the soon of the tooeses and the soon of the words and the soon of the words and the soon of the words and the soon of the words and the soon of the words and the soon of the words and the soon of the words and the soon of the  "



```python
generate_text(model, X, "Weights/weights-improvement-56-1.7114.hdf5", ix_to_char, vocab_size)
```

    Seed:
    " ible on the ramparts and at the gates.
    
    janos slynt met them at the door to the throne room, armored "
    Predictions
    "  the soon to the soon of the soot of the soot wht was a shalo oo the soow of the soot of the soot when they were so then and the soon of the soall sanears of the soadl of the soot of the soot when they were so then and the soon of the soall sanears o "

<br>

Well, the predictions aren't all too bad. But if you train this for even longer (100 or more epochs) and you can get the loss down to fractional values, you probably will see much better results. 
<br>
Try it out for yourself and ping me with your results!

Note - Try using word level encoding instead of characters just for fun.