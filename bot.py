# coding: utf-8
# Import the game simulator
from __future__ import print_function
import interface as bbox
import numpy as np
import copy
np.random.seed(1335)  # for reproducibility

 
n_features = n_actions = max_time = -1
 
 
def prepare_bbox():
    global n_features, n_actions, max_time
 
    # Reset environment to the initial state, just in case
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        # Load the game level
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()

def calc_best_action_using_checkpoint(action_range=50):
	
	# Pretty straightforward — we create a checkpoint and get it's ID 
	checkpoint_id = bbox.create_checkpoint()
 
	best_action = -1
	best_score = -1e9
 
	for action in range(n_actions):
		for _ in range(action_range): #random.randint(1,100)
			bbox.do_action(action)
		
		if bbox.get_score() > best_score:
			best_score = bbox.get_score()
			best_action = action
 
		bbox.load_from_checkpoint(checkpoint_id)
 
	return best_action
 
 
def run_bbox(verbose=False, epsilon=0.1, gamma=0.99, action_repeat=4, update_frequency=4, batchSize=32, buffer=100000, load_weights=False, save_weights=False):
    has_next = 1
    
    # Prepare environment - load the game level
    prepare_bbox()
    
    update_frequency_cntr = 0
    replay = []
    h=0
    if load_weights:
        model.load_weights('my_model_weights.h5')
        model_prim.load_weights('my_model_weights.h5')
    #stores tuples of (S, A, R, S')
 
    while has_next:
        # Get current environment state
        state = copy.copy(bbox.get_state())
        prev_reward = copy.copy(bbox.get_score())
        
        #Run the Q function on S to get predicted reward values on all the possible actions
        qval = model.predict(state.reshape(1,n_features), batch_size=1)
 
        # Choose an action to perform at current step
        if random.random() < epsilon: #choose random action or best action
            if random.random() < 0.5:
                action = np.random.randint(0,n_actions) #assumes 4 different actions
            else: # Use checkpoints to prime network with good actions
                action_range=50 #random.randint(1,200)
                action = calc_best_action_using_checkpoint(action_range=action_range)
                #for _ in range(action_range):
                #    has_next = bbox.do_action(action)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))


        # Perform chosen action, observe new state S'
        # Function do_action(action) returns False if level is finished, otherwise returns True.
        for a in range(action_repeat):
            has_next = bbox.do_action(action)
        new_state = copy.copy(bbox.get_state())
        reward = copy.copy(bbox.get_score()) - prev_reward
        #reward = 1.0 if reward > 0.0 else -1.0 #this gives better than random when combined with a small network

        #Experience replay storage
        if (len(replay) < buffer): #if buffer not filled, add to it
            replay.append((state, action, reward, new_state))
        else: #if buffer full, overwrite old values
            if (h < (buffer-1)):
                h += 1
            else:
                h = 0
            replay[h] = (state, action, reward, new_state)

            #randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            X_train = []
            y_train = []
            for memory in minibatch:
                #Get max_Q(S',a)
                old_state, action, reward, new_state = memory
                old_qval = model.predict(old_state.reshape(1,n_features), batch_size=1)
                newQ = model.predict(new_state.reshape(1,n_features), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,n_actions))
                y[:] = old_qval[:]
                if has_next == 1: #non-terminal state
                    update = (reward + (gamma * maxQ))
                else: #terminal state
                    update = reward
                y[0][action] = update
                X_train.append(old_state)
                y_train.append(y.reshape(n_actions,))

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            # update the weights of a copy of the network
            model_prim.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=0)
            if update_frequency_cntr >= update_frequency:
                prim_weights = model_prim.get_weights()
                print('model update')
                model.set_weights(prim_weights)
                update_frequency_cntr = 0
            update_frequency_cntr += 1

        if bbox.get_time() % 500000 == 0:
            print ("time = %d, score = %f" % (bbox.get_time(), bbox.get_score()))


    # Finish the game simulation, print earned reward and save weights
    if save_weights:
        model_prim.save_weights('my_model_weights.h5', overwrite=True)
    bbox.finish(verbose=1)

 
 
if __name__ == "__main__":

    from keras.models import Sequential
    from keras.models import model_from_json
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import RMSprop
    import random
    
    prepare_bbox()

    model = Sequential()
    model.add(Dense(n_features, init='lecun_uniform', input_shape=(n_features,)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    
    model.add(Dense(100, init='lecun_uniform')) #a 10 neuron network gives better than random result
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
#    model.add(Dense(10, init='lecun_uniform'))
#    model.add(Activation('relu'))
#    model.add(Dropout(0.2))

    model.add(Dense(n_actions, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

    rms = RMSprop(lr=0.00025)
    model.compile(loss='mse', optimizer=rms)
    
    json_string = model.to_json()
    open('my_model_architecture.json', 'w').write(json_string)
    # to load model architecture 'model = model_from_json(open('my_model_architecture.json').read())'
    model_prim = model_from_json(open('my_model_architecture.json').read())

    training = True
    exploration_epochs = 100
    learning_epochs = 100
    epsilon = 1 #1 is random
    gamma = 0.999 #a high gamma makes a long term reward more valuable
    action_repeat = 4 #repeat each action this many times
    update_frequency = 1000 #the number of time steps between each Q-net update
    batchSize = 128
    buffer = 300000
    load_weights = True
    
    if training:
        for i in range(exploration_epochs):
            print(i, epsilon, gamma, action_repeat, update_frequency, batchSize, buffer)
            run_bbox(verbose=0, epsilon=epsilon, gamma=gamma, action_repeat=action_repeat, update_frequency=update_frequency, batchSize=batchSize, buffer=buffer, load_weights=False, save_weights=True)
            if epsilon > 0.1:
                epsilon -= (1.0/exploration_epochs)

        for i in range(learning_epochs):
            epsilon = 0.1
            print(i, epsilon, gamma, action_repeat, update_frequency, batchSize, buffer)
            run_bbox(verbose=0, epsilon=epsilon, gamma=gamma, action_repeat=action_repeat, update_frequency=update_frequency, batchSize=batchSize, buffer=buffer, load_weights=load_weights, save_weights=True)
            load_weights = False

    else:
        has_next = 1
        # Prepare environment - load the game level
        prepare_bbox()
        model.load_weights('_my_model_weights.h5')
        while has_next:
            # Get current environment state
            state = copy.copy(bbox.get_state())
            #Run the Q function on S to get predicted reward values on all the possible actions
            qval = model.predict(state.reshape(1,n_features), batch_size=1)
            # Choose an action to perform at current step
            action = (np.argmax(qval))
            has_next = bbox.do_action(action)
        # Finish the game simulation
        bbox.finish(verbose=1)
