1. RNN Play Generator
    https://www.google.com/url?q=https%3A%2F%2Fwww.tensorflow.org%2Ftutorials%2Ftext%2Ftext_generation
2. Dataset
    We need one peice of training data. In fact, we can write our own poem or play and pass that to the network for training if we'd like. However, to make things easy we'll use an extract from a shakesphere play.
3. Loading Own Data
4. Reading the Contents of File
5. Encoding
6. Create Training Example
    Remember our task is to feed the model a sequence and have it return to us the next character. This means we need to split our text data from above into many shorter sequences that we can pass to the model as training examples. 
    The training examples we will prepapre will use a *seq_length* sequence as input and a *seq_length* sequence as the output where that sequence is the original sequence shifted one letter to the right. For example:
        ```input: Hell | output: ello```
    Our first step will be to create a stream of characters from our text data.
7. Building the Model
    use an embedding layer a LSTM and one dense layer that contains a node for each unique character in our training data. The dense layer will give us a probability distribution over all nodes.
8. Creating a Loss Function
    our model will output a (64, sequence_length, 65) shaped tensor that represents the probability distribution of each character at each timestep for every sequence in the batch.
9. Compiling the Model
    our problem as a classification problem where the model predicts the probabillity of each unique letter coming next.
10. Creating Checkpoints
    setup and configure our model to save checkpoinst as it trains. This will allow us to load our model from a checkpoint and continue training it.
11. Training
    If this is taking a while go to Runtime > Change Runtime Type and choose "GPU" under hardware accelerator.
12. Loading the Model
    rebuild the model from a checkpoint using a batch_size of 1 so that we can feed one peice of text to the model and have it make a prediction.
    Once the model is finished training, we can find the lastest checkpoint that stores the models weights using the following line.
    can load any checkpoint we want by specifying the exact file to load.
13. Generating Text
    