# blackboxchallenge
Code for the self reinformcement learning competition at blackboxchallenge.com

This is a self reinforcement learning submission using a neural network for the blackbox challenge at www.blackboxchallenge.com. Most of the tricks from DeepMinds Atari paper are included in some form. If there is structure in the data it should find it eventually. One trick I'm using (without success it seems) is to prime the network with both random exploration as well as optimal actions using checkpoints.

It uses a 100 neuron neural network:
* Has one value for a number of learning epochs with a high (but decreasing) exploration rate
* One value for a set number of learning epochs with a fixed exploration rate
* Stores neural network architecture and updates weights each epoch, so learning/training can be continued at will
* Repeats same action a specified number of times
* Updates the network at a specified interval

