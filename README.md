# DeepLearning-MusicGeneration
Generate music using LSTM networks from MIDI files

# Data Source:
https://www.freemidi.org/genre-jazz

# Reference link:
https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5

This is a great write up by Sigurður Skúli. It is a great place to learn about how you can read MIDI files and  then use that information in your networks.

# Objective:
Train a model on data and after training, the model should be able to generate music from an input seed.

# Reading MIDI files:
Music21 is the library used for reading files and getting notes and chord information from it.

Sample:

element:  <music21.chord.Chord C#4 F#2 F#3 F2>. 
Chord: [1, 5, 6] 

We store all these notes and chord information which becomes our data to be used in the model.
So in this case our value will be 1.5.6

# Data preparation:

We first create a dictionary which stores the unique notes/chord information as key and for values we give integral numbers.

e.g : {"0.1.2": 2, "0.1.2.6": 3}
The entire data is now mapped using this dictionary information.

# Batch creation:

To speed up the training process, we create batches of data to train. Each batch is of sequence length = 64 (time steps) and size 16.
When taking larger batch sizes, the accuracy was observed to be low. Since it is a many to many RNN, the input and output has to be created in similar way.

Input tensor: [16, 64]
Output tensor: [16,64,length of dictionary(unique chars] . 

The output tensor is of (n,m,p) , where p is the length of unique chars and it is encoded as one hot vector, with the value of the next char set to 1.

# Training:

If we train the entire data set at once, it may start taking a lot of time as the data volume increases. So we train on batches using the function train_on_batch function. The loss and accuracy are logged to file for each epoch. For 100 epochs, the accuracy reached in neighbourhood of 90 %. 
After every 10 epochs the model is saved, so that if something goes wrong, we dont have to start from scratch.

# Sampling:

To sample, we give number of chars to be generated as the output. The sample() method generates a random seed and uses that to make predictions or generating the next notes and so on.

# Runing the code:

Directories needed:

data : Put the MIDI files inside it

model : The models after epochs will be saved inside this directory

logs : Log directory

output : Output files are saved here

To train : python train.py --epochs=100 --freq=10

By default these values are 100 and 10.

To sample: python sample.py --chars=100

--chars is the number of chars to be generated

