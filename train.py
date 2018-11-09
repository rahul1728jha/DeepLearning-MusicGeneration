import os
import json
import argparse
from music21 import converter, instrument, note, chord
import numpy as np
import pickle
from model import build_model, save_weights
from datetime import datetime
import glob

DATA_DIR = './data'
LOG_DIR = './logs'
BATCH_SIZE = 16
SEQ_LENGTH = 64

class TrainLogger(object):
    def __init__(self, file):
        self.file = os.path.join(LOG_DIR, file)
        self.epochs = 0
        with open(self.file, 'w') as f:
            f.write('epoch,loss,acc\n')

    def add_entry(self, loss, acc):
        self.epochs += 1
        s = '{},{},{}\n'.format(self.epochs, loss, acc)
        with open(self.file, 'a') as f:
            f.write(s)

    
def log_notes_information(file, msg):
        with open(file, 'a') as f:
            f.write(msg)

def read_midi_files():
    start = datetime.now()
    notes=[]
    j=0
    print('Reading MIDI Files')
    for file in glob.glob(DATA_DIR +'/*.mid'):
        j += 1
        #if j % 5 == 0:
        #    print('%d  files processed ' %(j))

        print('Processing file : ' ,file)
        midi = converter.parse(file)

        notes_to_parse = None
        # file has instrument parts
        try: 
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        # file has notes in a flat structure
        except: 
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    
    print('All files read. Duming notes to file')
    with open(os.path.join(DATA_DIR, 'notes'), 'wb') as filepath:
        pickle.dump(notes, filepath)
    print('Total Time Taken : %s ' %(str(datetime.now() -start)))
    return notes
			
def char_index_char_mapping(notes):
    msg =''
    char_to_idx = { ch: i for (i, ch) in enumerate(sorted(list(set(notes)))) }

    with open(os.path.join(DATA_DIR, 'char_to_idx.json'), 'w') as f:
        json.dump(char_to_idx, f)

    idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }
    vocab_size = len(char_to_idx)

    uniqueNotesLen = len(set(notes))
    msg='\nTotal notes length : ' + str(len(notes))
    msg= msg + '\nTotal unique notes length : ' + str(len(char_to_idx))
    msg = msg + '\n*' *100
    msg = msg + '\nUnique notes : \n' + str(char_to_idx)
    msg = msg + '\n*' *100

    log_notes_information('NotesInfo.txt',msg)
    return char_to_idx,idx_to_char,vocab_size,uniqueNotesLen

def read_batches(T, vocab_size):
    length = T.shape[0]; 
    batch_chars = int(length / BATCH_SIZE);

    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH): 
        X = np.zeros((BATCH_SIZE, SEQ_LENGTH)) 
        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size)) 
        for batch_idx in range(0, BATCH_SIZE): 
            for i in range(0, SEQ_LENGTH): 
                X[batch_idx, i] = T[batch_chars * batch_idx + start + i] 
                Y[batch_idx, i, T[batch_chars * batch_idx + start + i + 1]] = 1
        yield X, Y

def train(notes,char_to_idx,uniqueNotesLen, epochs=100, save_freq=10):

    #model_architecture
    model = build_model(BATCH_SIZE, SEQ_LENGTH, vocab_size)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    #Train data generation
    T = np.asarray([char_to_idx[c] for c in notes], dtype=np.int32) #convert complete text into numerical indices
    #T_norm = T / float(uniqueNotesLen)
    print("Length of text:" + str(T.size)) 
    print("Length of unique test: ," ,uniqueNotesLen)

    steps_per_epoch = (len(notes) / BATCH_SIZE - 1) / SEQ_LENGTH   
    print('Steps per epoch : ' ,steps_per_epoch)

    log = TrainLogger('training_log.csv')

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        
        losses, accs = [], []
        msg = ""
        for i, (X, Y) in enumerate(read_batches(T, vocab_size)):
            
            print(X);

            loss, acc = model.train_on_batch(X, Y)
            print('Batch {}: loss = {}, acc = {}'.format(i + 1, loss, acc))
            losses.append(loss)
            accs.append(acc)

        log.add_entry(np.average(losses), np.average(accs))
        
        if (epoch + 1) % save_freq == 0:
            save_weights(epoch + 1, model)
            print('Saved checkpoint to', 'weights.{}.h5'.format(epoch + 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on some text.')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--freq', type=int, default=10, help='checkpoint save frequency')
    args = parser.parse_args()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
	
    notes = read_midi_files()
    char_to_idx,idx_to_char,vocab_size,uniqueNotesLen = char_index_char_mapping(notes)
    train(notes,char_to_idx,uniqueNotesLen, args.epochs, args.freq)
