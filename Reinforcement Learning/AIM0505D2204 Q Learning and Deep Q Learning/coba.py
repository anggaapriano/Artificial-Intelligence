import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Inisialisasi lingkungan
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameter
learning_rate = 0.001
discount_factor = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory_size = 1000000
episodes = 1000
steps_per_episode = 300

# Inisialisasi model DQL
model = Sequential()
model.add(Dense(24, input_dim=state_size, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(action_size, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

# Inisialisasi pengalaman
memory = []
for i in range(memory_size):
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            break

# Melatih model
for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    score = 0
    for step in range(steps_per_episode):
        # Pilih tindakan
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values[0])
        
        # Ambil tindakan dan perbarui pengalaman
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        memory.append((state, action, reward, next_state, done))
        
        # Latih model
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = reward + discount_factor * np.amax(model.predict(next_state)[0])
                q_values = model.predict(state)
                q_values[0][action] = target
                model.fit(state, q_values, verbose=0)
        
        state = next_state
        score += reward
        
        if done:
            break
    
    # Kurangi nilai epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    # Cetak hasil
    print("Episode:", episode, "Score:", score, "Epsilon:", epsilon)
