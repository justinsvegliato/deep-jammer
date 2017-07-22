# Deep Jammer

## Installation (Work in progress --- I'm literally working on this right now.)

To set up Deep Jamer, we just need to do a few steps:

1. First, we need to install several Python packages. Run the commands below to get all of the packages we need:

`sudo pip install theano`
`sudo pip install theano_lstm`
`sudo pip install python-midi`

2. Next, we need to create a **repository**. A repository is a collection of music in the representation needed by Deep Jammer. To do this, we need to run the command below. Note that repository_handler.py uses the MIDI files stored in the pieces directory and saves the repository in the repository directory. By default, we included (1) a few MIDI files in the pieces directory and (2) the corresponding repository called test-repository in the repository directory. 

`./repository_handler.py repository-name`

3. After we've built a repository, we can train Deep Jammer. All we need to do is run the command below. If you run into any permission issues, just add `sudo` to the front of the command. This should solve any problems.

`python deep_jammer.py repository-name`

4. If Deep Jammer runs properly, you should see the output below. In the configurations directory, you can find the a collection of **configurations**. A configuration is a collection of weights 

Retrieving the repository...
Generating Deep Jammer...
Training Deep Jammer...
Epoch 0
    Loss = 25687
    Pieces = 0
Epoch 0 -> Checkpoint
Saving Deep Jammer...
Deep Jamming...

5. Now that we've can Deep Jammer, we can adjust some of the parameters in the file deep_jammer.py.
