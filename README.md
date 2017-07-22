# Deep Jammer
Deep Jammer is a deep neural network can endlessly compose classical piano music by learning from a ton of music from Bach, Beethoven, Mozart, and other famous composers. If you want to read a blog post about this, check it out [here](https://medium.com/towards-data-science/can-a-deep-neural-network-compose-music-f89b6ba4978d)! And, if you're feeling really nerdy, check out [the report](https://www.justinsvegliato.com/pdf/deep-jammer-report.pdf) and the [the poster](https://www.justinsvegliato.com/pdf/deep-jammer-poster.pdf).

## Installation

To set up Deep Jamer, we just need to do a few steps:

1. First, make sure that you're using Python2.7 or else you'll get errors. This will not work with Python3 since we don't use the **print** command as a function (and probably other stuff). Yup, we totally regret that decision.

1. Before we can run Deep Jammer, we need to install a few Python packages, some of which you may already have. To get every package, we just run the command below. While we're pretty sure that we've included every package, let us know if we missed anything.

`sudo pip install theano`
`sudo pip install theano_lstm`
`sudo pip install python-midi`

2. Next, we need to create a **repository**. A repository is a collection of music in the representation understood by Deep Jammer. To do this, we need to run the command below. This could take a while just as a heads up. Note that **repository_handler.py** uses the all of the MIDI files stored in the **pieces** directory and saves the repository in the **repositories** directory. By default, we include (a) a a lot of MIDI files in the **pieces** directory and (b) the corresponding repository named **test-repository** in the **repositories** directory. If you run into any permission issues, just add `sudo` to the front of the command. This should solve any problems. I think this happens because it needs to read MIDI files and write a repository file on your file system.

 `./repository-handler.py repository-name`
 
4. If that command ran successfully, you should see the following output:

```
Loading the MIDI files...
Saving the repository
```
 
5. Now that we've built a repository, we can train Deep Jammer. All we need to do is run the command below. Again, add `sudo` to the front of the command if you run into any permission issues. Theano needs to access a few files in **~/.theano**.

`./deep_jammer.py repository-name`

6. If Deep Jammer runs properly, you should see the output below. At every 5 epochs, we print out a summary of the current loss. We also save the current configuration (i.e., the current weights) and compose a sample. If you want, you can adjust how often we save the current configuration or compose a sample piece by modifying the variables **SUMMARY_THRESHOLD** and **CHECKPOINT_THRESHOLD** in **deep_jammer.py**. More importantly, we store every loss in a file called **loss-history.txt**.

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

7. Now that we've trained Deep Jammer, let's adjust some of the parameters of our neural network just so you know how. At the top of **deep_jammer.py**, you'll see two variables: **TIME_MODEL_LAYERS** and **NOTE_MODEL_LAYERS**. Basically, these arrays represents the number of nodes in each time layer and each note layer. You can adjust them to whatever you want. We found that the current configuration is the sweet spot though. While there are other variables that you can adjust, they aren't as important. Feel free to play around with them if you want.

8. In addition to adjusting the parameters of our neural network, we can change the the number of epochs and the batch size. By default, the number of epochs is 200 and the batch size is 5. You'll probably need more epochs to generate good music. We used 5000 epochs. That said, all we have to do is add two optional flags to the command we ran before. Check it out below:

`./deep_jammer.py --epochs epochs --batch-size batch-size repository-name`
