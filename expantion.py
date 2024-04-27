import numpy as np
import random
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
import networkx as nx
import matplotlib.pyplot as plt
import pickle

class VocabNode:
    def __init__(self, token_id, initial_state, initial_r, learning_rate):
        self.token_id = token_id
        self.state = initial_state
        self.r = initial_r
        self.learning_rate = learning_rate
        
    def update_state(self):
        self.state = self.r * self.state * (1 - self.state)
        
    def update_r(self, reward):
        self.r += self.learning_rate * reward
        self.r = np.clip(self.r, 0, 4)

class PyramidNetwork:
    def __init__(self, vocab_size, min_r, max_r, learning_rate):
        self.nodes = [VocabNode(i, np.random.rand(), np.random.uniform(min_r, max_r), learning_rate) for i in range(vocab_size)]
        self.edges = np.zeros((vocab_size, vocab_size))
        
    def update_network(self):
        prev_states = [node.state for node in self.nodes]
        for node in self.nodes:
            node.update_state()
        curr_states = [node.state for node in self.nodes]
        rewards = []
        for i in range(len(self.nodes)):
            ripple_reward = np.mean([abs(prev_states[j] - curr_states[i]) * self.edges[i][j] for j in range(len(self.nodes)) if i != j])
            chaos_penalty = abs(curr_states[i] - prev_states[i])
            reward = ripple_reward - chaos_penalty
            rewards.append(reward)
        for i, node in enumerate(self.nodes):
            node.update_r(rewards[i])
        stability = np.mean(np.abs(np.array(curr_states) - np.array(prev_states)))
        return stability

def save_model(network, tokenizer, file_path):
    model_data = {
        'network': network,
        'tokenizer': tokenizer
    }
    with open(file_path, 'wb') as file:
        pickle.dump(model_data, file)

def load_model(file_path):
    with open(file_path, 'rb') as file:
        model_data = pickle.load(file)
    return model_data['network'], model_data['tokenizer']

def generate_text(network, tokenizer, prompt, max_length=100):
    generated_text = prompt
    prev_token_id = tokenizer.encode(prompt).ids[-1]
    for _ in range(max_length):
        next_token_probs = network.edges[prev_token_id]
        next_token_probs /= np.sum(next_token_probs)
        next_token_id = np.random.choice(range(len(next_token_probs)), p=next_token_probs)
        generated_text += tokenizer.decode([next_token_id])
        if tokenizer.decode([next_token_id]) != ' ':
            generated_text += ' '
        prev_token_id = next_token_id
    return generated_text

# Load the saved model
model_file_path = 'pyramid_network_model.pkl'
network, tokenizer = load_model(model_file_path)

# Load and tokenize the new corpus
with open('shakespeare.txt', 'r', encoding='utf-8') as file:
    new_corpus = file.read()

# Train the tokenizer on the new corpus
tokenizer.train_from_iterator(new_corpus.split('\n'), trainer)

# Encode the new corpus
new_encoded = tokenizer.encode(new_corpus)

# Get the updated vocabulary size
updated_vocab_size = tokenizer.get_vocab_size()

# Get the original vocabulary size
vocab_size = len(network.nodes)

# Define the hyperparameters
min_r = 3.86580
max_r = 4.0
learning_rate = 0.01

# Add new nodes to the network for the new tokens
for i in range(vocab_size, updated_vocab_size):
    new_node = VocabNode(i, np.random.rand(), np.random.uniform(min_r, max_r), learning_rate)
    network.nodes.append(new_node)

# Expand the edges matrix to accommodate the new nodes
network.edges = np.pad(network.edges, ((0, updated_vocab_size - vocab_size), (0, updated_vocab_size - vocab_size)), mode='constant')

# Update edge weights for the new tokens
print("Updating edge weights for new tokens...")
num_new_tokens = len(new_encoded.ids)
for i in range(num_new_tokens - 1):
    src_token_id = new_encoded.ids[i]
    dst_token_id = new_encoded.ids[i + 1]
    network.edges[src_token_id][dst_token_id] += 1
    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1} / {num_new_tokens} new tokens")

# Normalize the updated edge weights
network.edges /= np.max(network.edges)

# Train the updated network
num_epochs = 200
print(f"Training the updated network for {num_epochs} epochs...")
stability_scores = []
for epoch in range(1, num_epochs + 1):
    stability = network.update_network()
    stability_scores.append(stability)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Stability = {stability}")

# Visualize the graph after training
G = nx.DiGraph(network.edges)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=100)
nx.draw_networkx_edges(G, pos, width=0.5, arrows=True)
plt.axis('off')
plt.title('PyramidNetwork Graph')
plt.show()

# Save the updated model
updated_model_file_path = 'updated_pyramid_network_model.pkl'
save_model(network, tokenizer, updated_model_file_path)
