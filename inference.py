import pickle
import numpy as np

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

# Load the saved model
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model_data = pickle.load(file)
    return model_data['network'], model_data['tokenizer']

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

# Load the saved model for inference
model_file_path = 'chat_pyramid_network_model.pkl'
loaded_network, loaded_tokenizer = load_model(model_file_path)

# Example usage of the loaded model
prompt = "who are you ? "
generated_text = generate_text(loaded_network, loaded_tokenizer, prompt)
print("Generated text:", generated_text)
