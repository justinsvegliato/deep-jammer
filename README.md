# Deep Jammer

## Installation (Work in progress - Working on this right now.)

To set up Deep Jamer, we just need to do a few steps:

1. First, make sure that you're using Python2.7. This will not work with Python3 since I don't use the **print** command as a function. Yup, we totally regret that decision.

1. Before we can run Deep Jammer, we need to install a few Python packages. To get every package we need, run the command below. While we're pretty sure that we've included every package, let us know if we missed anything.

`sudo pip install theano`
`sudo pip install theano_lstm`
`sudo pip install python-midi`

2. Next, we need to create a **repository**. A repository is a collection of music in the representation understood by Deep Jammer. To do this, we need to run the command below. Note that **repository_handler.py** uses the all of the MIDI files stored in the **pieces** directory and saves the repository in the **repositories** directory. By default, we include (a) a few MIDI files in the **pieces** directory and (b) the corresponding repository named **test-repository** in the **repositories** directory. If you run into any permission issues, just add `sudo` to the front of the command. This should solve any problems. This probably happens since it needs to read MIDI files and write a repository file on your file system.

 `./repository-handler.py repository-name`
 
3. Now that we've built a repository, we can train Deep Jammer. All we need to do is run the command below. Again, add `sudo` to the front of the command if you run into any permission issues. Theano needs to access a few files in **~/.theano**.

`./deep_jammer.py repository-name`

4. If Deep Jammer runs properly, you should see the output below. At every 5 epochs, we print out a summary of the current loss. We also save the current configuration (i.e., the current weights) of the network and compose a sample piece that represents the current loss. If you want, you can adjust how often we save the current configuration or compose a sample piece by modifying the variables **SUMMARY_THRESHOLD** and **CHECKPOINT_THRESHOLD** in **deep_jammer.py**.

```
Generating Deep Jammer...
Training Deep Jammer...
Epoch 0
    Loss = 64153
    Pieces = 0
Epoch 0 -> Checkpoint
Epoch 5
    Loss = 19303
    Pieces = 25
Epoch 5 -> Checkpoint
Saving Deep Jammer...
Deep Jamming...
```

5. Now that we've trained Deep Jammer, let's adjust some of the parameters of our neural netwotrk just so you know how. At the top of **deep_jammer.py**, you'll see two variables: **TIME_MODEL_LAYERS** and **NOTE_MODEL_LAYERS**. Basically, the arrays represents the number of nodes in each time layer and each note layer. You can adjust these to whatever you want. We found that the current configuration is the sweet spot. While there are other variables that you can adjust, they aren't as important. Feel free to play around with them if you want though.

6. In addition to adjusting the parameters of our neural network, we can change the the number of epochs and the batch size. All we have to do is add two optional flags to the command we ran before. Check it out below:

`./deep_jammer.py --epochs epochs --batch-size batch-size repository-name`
