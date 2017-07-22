# Deep Jammer

## Installation (Work in progress --- I'm literally working on this right now.)

* Mention Python version
* Mention other packages
* Update variables in Deep Jammer
* Add music to the pieces directory

To set up Deep Jamer, we just need to do a few steps:

1. Before we can run Deep Jammer, we need to install a few Python packages. To get every package we need, run the following command:

`sudo pip install theano`
`sudo pip install theano_lstm`
`sudo pip install python-midi`

2. Next, we need to create a **repository**. A repository is a collection of music in the representation understood by Deep Jammer. To do this, we need to run the command below. Note that **repository_handler.py** uses the all of the MIDI files stored in the **pieces** directory and saves the repository in the **repositories** directory. By default, we include (a) a few MIDI files in the **pieces** directory and (b) the corresponding repository named **test-repository** in the **repositories** directory. If you run into any permission issues, just add `sudo` to the front of the command. This should solve any problems. This probably happens since it needs to read and write files on your file system.

 ./repository-handler.py repository-name
 
3. Now that we've built a repository, we can train Deep Jammer. All we need to do is run the command below. Again, add `sudo` to the front of the command if you run into any permission issues. Theano needs to access a few files in **~/.theano.

`./deep_jammer.py repository-name`

4. If Deep Jammer runs properly, you should see the following output:

Retrieving the repository...
Generating Deep Jammer...
Training Deep Jammer...
Epoch 0
    Loss = 25687
    Pieces = 0
Epoch 0 -> Checkpoint
Saving Deep Jammer...
Deep Jamming...

5. Now that we've can Deep Jammer, we can adjust some of the parameters in the file **deep_jammer.py**.
