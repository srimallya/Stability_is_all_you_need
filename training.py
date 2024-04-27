import numpy as np
import random
import pickle
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

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

# Load and tokenize the corpus
with open('Ontology.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.train_from_iterator(corpus.split('\n'), trainer)

encoded = tokenizer.encode(corpus)
vocab_size = tokenizer.get_vocab_size()

# Create the PyramidNetwork
min_r = 3.86580
max_r = 4.0
learning_rate = 0.01
network = PyramidNetwork(vocab_size, min_r, max_r, learning_rate)

# Map related nodes and update edge weights
print("Mapping related nodes and updating edge weights...")
num_tokens = len(encoded.ids)
for i in range(num_tokens - 1):
    src_token_id = encoded.ids[i]
    dst_token_id = encoded.ids[i + 1]
    network.edges[src_token_id][dst_token_id] += 1
    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1} / {num_tokens} tokens")

# Normalize edge weights
network.edges /= np.max(network.edges)

# Print the number of nodes
print(f"Number of nodes: {vocab_size}")

# Train the network
num_epochs = 10000
print(f"Training the network for {num_epochs} epochs...")
stability_scores = []
for epoch in range(1, num_epochs + 1):
    stability = network.update_network()
    stability_scores.append(stability)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Stability = {stability}")

# Plot stability over time
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), stability_scores)
plt.xlabel('Epoch')
plt.ylabel('Stability Measure')
plt.title('Network Stability over 10000 Epochs')
plt.grid(True)
plt.show()

# Save the trained model
def save_model(network, tokenizer, file_path):
    model_data = {
        'network': network,
        'tokenizer': tokenizer
    }
    with open(file_path, 'wb') as file:
        pickle.dump(model_data, file)

# Load the saved model
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model_data = pickle.load(file)
    return model_data['network'], model_data['tokenizer']

# Save the model after training
model_file_path = 'pyramid_network_model.pkl'
save_model(network, tokenizer, model_file_path)

# Load the saved model for inference
loaded_network, loaded_tokenizer = load_model(model_file_path)

# Inference function
def generate_text(network, tokenizer, prompt, max_length=100):
    generated_text = prompt
    prev_token_id = tokenizer.encode(prompt).ids[-1]
    for _ in range(max_length):
        next_token_probs = network.edges[prev_token_id]
        # Normalize the probabilities
        next_token_probs /= np.sum(next_token_probs)
        next_token_id = np.random.choice(range(len(next_token_probs)), p=next_token_probs)
        generated_text += tokenizer.decode([next_token_id])
        # Add a space token after each generated word
        if tokenizer.decode([next_token_id]) != ' ':
            generated_text += ' '
        prev_token_id = next_token_id
    return generated_text

# Example usage
prompt = "The boy looked at the machine "
generated_text = generate_text(network, tokenizer, prompt)
print("Generated text:", generated_text)
