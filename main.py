import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import random
import os
import pickle  
import argparse

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
GRID_SIZE = 10         
NUM_EPISODES = 10000  
MAX_STEPS = 150        
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99 
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9998 

OUTPUT_DIR = "rl_output_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. THE ENVIRONMENT (Warehouse with Obstacles)
# ==========================================
class WarehouseEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        
        # Agent 1 (Blue) - Start Top-Left -> Goal Bottom-Right
        self.start_1 = (0, 0)
        self.goal_1 = (9, 9)

        # Agent 2 (Red) - Start Bottom-Left -> Goal Top-Right
        self.start_2 = (9, 0)
        self.goal_2 = (0, 9)

        self.obstacles = []
        for r in range(GRID_SIZE):
            if r != 4 and r != 5: 
                self.obstacles.append((r, 5))
        
        self.obstacles.extend([(2, 2), (7, 7), (2, 7), (7, 2)])

        self.reset()

    def reset(self):
        self.agent_1_pos = self.start_1
        self.agent_2_pos = self.start_2
        
        self.agent_1_arrived = False
        self.agent_2_arrived = False
        
        self.done = False
        return self.get_state()

    def get_state(self):
        return (self.agent_1_pos, self.agent_2_pos)

    def step(self, action_1, action_2):
        moves = {
            0: (-1, 0), 1: (1, 0),  # Up, Down
            2: (0, -1), 3: (0, 1),  # Left, Right
            4: (0, 0)               # Stay
        }

        # Calculate proposed positions
        prop_pos_1 = self._get_proposed_pos(self.agent_1_pos, moves[action_1])
        prop_pos_2 = self._get_proposed_pos(self.agent_2_pos, moves[action_2])

        reward_1 = -0.1
        reward_2 = -0.1

        # --- OBSTACLE COLLISION LOGIC ---
        # If an agent hits a wall, it stays where it is and gets a penalty

        if prop_pos_1 in self.obstacles:
            reward_1 -= 1.0  # Collision penalty
            prop_pos_1 = self.agent_1_pos # Bounce back
        
        if prop_pos_2 in self.obstacles:
            reward_2 -= 1.0
            prop_pos_2 = self.agent_2_pos

        # --- AGENT COLLISION LOGIC ---
        collision = False
        if prop_pos_1 == prop_pos_2:
            collision = True
        if prop_pos_1 == self.agent_2_pos and prop_pos_2 == self.agent_1_pos:
            collision = True 

        if collision:
            reward_1 -= 10
            reward_2 -= 10
            # In case of collision, they don't move
            new_pos_1 = self.agent_1_pos
            new_pos_2 = self.agent_2_pos
        else:
            new_pos_1 = prop_pos_1
            new_pos_2 = prop_pos_2

        # Update positions
        self.agent_1_pos = new_pos_1
        self.agent_2_pos = new_pos_2

        # Agent 1
        if self.agent_1_pos == self.goal_1:
            if not self.agent_1_arrived:
                reward_1 += 50      # Reward only the first time
                self.agent_1_arrived = True
            else:
                reward_1 += 0       # If already there, no extra reward (or small bonus 0.1 for staying still)
        
        # Agente 2
        if self.agent_2_pos == self.goal_2:
            if not self.agent_2_arrived:
                reward_2 += 50
                self.agent_2_arrived = True
            else:
                reward_2 += 0

        # Episode ends if BOTH flags are True
        if self.agent_1_arrived and self.agent_2_arrived:
            self.done = True

        return self.get_state(), (reward_1, reward_2), self.done

    def _get_proposed_pos(self, pos, move):
        new_r = max(0, min(self.grid_size - 1, pos[0] + move[0]))
        new_c = max(0, min(self.grid_size - 1, pos[1] + move[1]))
        return (new_r, new_c)

# ==========================================
# 3. THE AGENT (Standard Q-Learning)
# ==========================================
class QLearningAgent:
    def __init__(self, action_space_size=5):
        self.agent_1_arrived = False
        self.agent_2_arrived = False
        self.q_table = {} 
        self.action_space_size = action_space_size
        self.lr = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EPSILON_START

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            q_values = [self.get_q(state, a) for a in range(self.action_space_size)]
            max_q = max(q_values)
            actions_with_max_q = [i for i, q in enumerate(q_values) if q == max_q]
            return random.choice(actions_with_max_q)

    def learn(self, state, action, reward, next_state):
        current_q = self.get_q(state, action)
        next_q_values = [self.get_q(next_state, a) for a in range(self.action_space_size)]
        max_next_q = max(next_q_values)
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def save_model(self, filename):
        """Save Q-Table on a file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"💾 Model saved to {filename}")

    def load_model(self, filename):
        """Load Q-Table from a file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"📂 Model loaded from {filename}")
        else:
            print(f"⚠️ Warning: Model file {filename} not found! Starting from scratch.")

# ==========================================
# 4. TRAINING LOOP
# ==========================================

def update_live_plot(ax, env, state, episode, step):
    """Updates the plot in real-time during training."""
    ax.clear()
    ax.set_title(f"Training... Ep: {episode} | Step: {step}")
    
    # Disegna Griglia e Muri
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.invert_yaxis()
    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
    
    for obs in env.obstacles:
        rect = patches.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1, color='black')
        ax.add_patch(rect)

    # Draw Goals
    ax.text(env.goal_1[1], env.goal_1[0], "G1", ha='center', va='center', color='blue', fontweight='bold')
    ax.text(env.goal_2[1], env.goal_2[0], "G2", ha='center', va='center', color='red', fontweight='bold')

    # Draw Agents
    p1, p2 = state
    circle_1 = patches.Circle((p1[1], p1[0]), 0.3, color='blue', alpha=0.8)
    circle_2 = patches.Circle((p2[1], p2[0]), 0.3, color='red', alpha=0.8)
    ax.add_patch(circle_1)
    ax.add_patch(circle_2)

    plt.draw()
    plt.pause(0.001)

def train(render_interval=1000):
 # Default: show 1 episode every 1000
    env = WarehouseEnv()
    agent_1 = QLearningAgent()
    agent_2 = QLearningAgent()
    
    # Live plot preparation
    if render_interval > 0:
        plt.ion()  # Turn on Interactive Mode
        fig, ax = plt.subplots(figsize=(6, 6))
    
    print("🤖 Training Started...")
    rewards_history = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        
        # Decide whether to show this episode
        show_this_episode = (render_interval > 0) and (episode % render_interval == 0)

        for step in range(MAX_STEPS):
            action_1 = agent_1.choose_action(state)
            action_2 = agent_2.choose_action(state)
            next_state, rewards, done = env.step(action_1, action_2)
            
            agent_1.learn(state, action_1, rewards[0], next_state)
            agent_2.learn(state, action_2, rewards[1], next_state)
            
            # --- LIVE VISUALIZATION ---
            if show_this_episode:
                update_live_plot(ax, env, state, episode, step)
            # ----------------------------

            state = next_state
            total_reward += sum(rewards)
            if done: break

        agent_1.decay_epsilon()
        agent_2.decay_epsilon()
        rewards_history.append(total_reward)

        if episode % 1000 == 0:
            print(f"Episode {episode}: Reward {total_reward:.1f} | Epsilon: {agent_1.epsilon:.3f}")

    if render_interval > 0:
        plt.ioff() # Turn off interactive mode at the end
        plt.close()

    return agent_1, agent_2, rewards_history

# ==========================================
# 5. VISUALIZATION
# ==========================================
def run_demo(agent_1, agent_2):
    print("🎥 Generating Complex Scenario GIF...")
    env = WarehouseEnv()
    state = env.reset()
    agent_1.epsilon = 0 # Pure greedy
    agent_2.epsilon = 0

    frames_data = []
    for _ in range(MAX_STEPS):
        frames_data.append(state)
        action_1 = agent_1.choose_action(state)
        action_2 = agent_2.choose_action(state)
        state, _, done = env.step(action_1, action_2)
        if done:
            frames_data.append(state)
            break
            
    if len(frames_data) == MAX_STEPS:
        print("⚠️ Warning: Agents did not reach the goal in the demo. Training might need more episodes.")

    fig, ax = plt.subplots(figsize=(7, 7))

    def draw_grid(state_data):
        ax.clear()
        ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1))
        ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1))
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, GRID_SIZE - 0.5)
        ax.invert_yaxis()

        # Draw Obstacles (Black Blocks)
        for obs in env.obstacles:
            rect = patches.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='black')
            ax.add_patch(rect)

        p1, p2 = state_data

        # Draw Goals
        rect_g1 = patches.Rectangle((env.goal_1[1]-0.4, env.goal_1[0]-0.4), 0.8, 0.8, linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
        ax.add_patch(rect_g1)
        ax.text(env.goal_1[1], env.goal_1[0], "G1", ha='center', va='center', color='blue', fontweight='bold')

        rect_g2 = patches.Rectangle((env.goal_2[1]-0.4, env.goal_2[0]-0.4), 0.8, 0.8, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect_g2)
        ax.text(env.goal_2[1], env.goal_2[0], "G2", ha='center', va='center', color='red', fontweight='bold')

        # Draw Agents
        circle_1 = patches.Circle((p1[1], p1[0]), 0.3, color='blue', label='Agent 1', alpha=0.8)
        ax.add_patch(circle_1)
        circle_2 = patches.Circle((p2[1], p2[0]), 0.3, color='red', label='Agent 2', alpha=0.8)
        ax.add_patch(circle_2)

        ax.set_title(f"Warehouse 10x10: Obstacles & Bottleneck")

    ani = animation.FuncAnimation(fig, draw_grid, frames=frames_data, interval=200, repeat=False)
    ani.save(os.path.join(OUTPUT_DIR, "warehouse_obstacles.gif"), writer='pillow')
    print(f"✨ GIF saved to {OUTPUT_DIR}")
    plt.close()

def plot_learning(history):
    plt.figure(figsize=(10, 5))
    # Moving average every 500 episodes
    window = 500
    if len(history) >= window:
        avg_rewards = np.convolve(history, np.ones(window)/window, mode='valid')
        plt.plot(avg_rewards, color='green')
        plt.title("Training Learning Curve (Moving Average)")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve.png"))
        print("📈 Learning curve saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Warehouse Multi-Agent RL')
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'demo'], 
                        help='Scegli "train" per addestrare o "demo" per vedere il risultato')
    
    parser.add_argument('--live', action='store_true', 
                        help='Se attivo, mostra il training in tempo reale ogni 500 episodi')
    
    args = parser.parse_args()

    model_file_1 = os.path.join(OUTPUT_DIR, "agent_1_qtable.pkl")
    model_file_2 = os.path.join(OUTPUT_DIR, "agent_2_qtable.pkl")

    if args.mode == 'train':
        render_interval = 500 if args.live else 0
        
        print(f"⚙️  Mode: TRAINING (Live View: {'ON' if args.live else 'OFF'})")
        if args.live:
            print("ℹ️  Premi Ctrl+C nel terminale se vuoi interrompere prima.")

        trained_a1, trained_a2, history = train(render_interval=render_interval)

        print("\n💾 Saving models...")
        trained_a1.save_model(model_file_1)
        trained_a2.save_model(model_file_2)
        
        plot_learning(history)
        print("✅ Training pipeline finished.")

    elif args.mode == 'demo':
        print(f"⚙️  Mode: DEMO")
        
        if not os.path.exists(model_file_1) or not os.path.exists(model_file_2):
            print("❌ Error: Model files not found! Run training first: python main.py --mode train")
            exit()

        print("🚀 Loading trained models...")
        
        # Create empty agents and inject memory (Q-Table)
        demo_agent_1 = QLearningAgent()
        demo_agent_2 = QLearningAgent()
        
        demo_agent_1.load_model(model_file_1)
        demo_agent_2.load_model(model_file_2)
        
        # Set epsilon to 0 for demo (exploitation only, no exploration)
        demo_agent_1.epsilon = 0
        demo_agent_2.epsilon = 0
        
        run_demo(demo_agent_1, demo_agent_2)



# python main.py --mode train
# python main.py --mode train --live
# python main.py --mode demo