import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import argparse

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
GRID_SIZE = 10
MAX_STEPS = 150
BATCH_SIZE = 128
LR = 0.0001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9992
TARGET_UPDATE = 50  # Update target network every N episodes
MEMORY_SIZE = 50000
NUM_EPISODES = 3000  # Fewer episodes needed with DQN than Q-Table usually

OUTPUT_DIR = "rl_deep_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. THE NEURAL NETWORK (DQN)
# ==========================================
# This replaces the Q-Table. It maps State -> Q-Values for all actions.
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),  
            nn.ReLU(),
            nn.Linear(256, 256),      
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. GYMNASIUM ENVIRONMENT
# ==========================================
class WarehouseGymEnv(gym.Env):
    def __init__(self, grid_size=GRID_SIZE):
        super(WarehouseGymEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4) 
        
        # [A_r, A_c, G_r, G_c, O_r, O_c, Wall_U, Wall_D, Wall_L, Wall_R]
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        self.obstacles = self._generate_obstacles()

    def _generate_obstacles(self):
        obs = []
        for r in range(self.grid_size):
            if r != 4 and r != 5: 
                obs.append((r, 5))
        obs.extend([(2, 2), (7, 7), (2, 7), (7, 2)])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        possible_points = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) 
                           if (r, c) not in self.obstacles]
        points = random.sample(possible_points, 4)
        self.agent_1_pos = points[0]
        self.agent_2_pos = points[1]
        self.goal_1 = points[2]
        self.goal_2 = points[3]
        self.agent_1_arrived = False
        self.agent_2_arrived = False
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        s = self.grid_size
        
        # Helper function to detect walls (Sensors)
        def get_sensors(pos):
            # Returns 1.0 if there's a wall or border in that direction, otherwise 0.0
            # Order: Up, Down, Left, Right
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            sensors = []
            for d in dirs:
                nr, nc = pos[0] + d[0], pos[1] + d[1]
                if nr < 0 or nr >= s or nc < 0 or nc >= s or (nr, nc) in self.obstacles:
                    sensors.append(1.0)
                else:
                    sensors.append(0.0)
            return sensors

        sens_1 = get_sensors(self.agent_1_pos)
        sens_2 = get_sensors(self.agent_2_pos)

        # Agent 1 State (10 values)
        obs_1 = np.array([
            self.agent_1_pos[0]/s, self.agent_1_pos[1]/s,
            self.goal_1[0]/s, self.goal_1[1]/s,
            self.agent_2_pos[0]/s, self.agent_2_pos[1]/s,
            *sens_1
        ], dtype=np.float32)

        # Agent 2 State (10 valori)
        obs_2 = np.array([
            self.agent_2_pos[0]/s, self.agent_2_pos[1]/s,
            self.goal_2[0]/s, self.goal_2[1]/s,
            self.agent_1_pos[0]/s, self.agent_1_pos[1]/s,
            *sens_2
        ], dtype=np.float32)
        
        return (obs_1, obs_2)

    def _get_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def step(self, action_1, action_2):
        self.step_count += 1
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)} # No Stay

        dist_1_old = self._get_distance(self.agent_1_pos, self.goal_1)
        dist_2_old = self._get_distance(self.agent_2_pos, self.goal_2)

        prop_1 = self._move(self.agent_1_pos, moves[action_1])
        prop_2 = self._move(self.agent_2_pos, moves[action_2])

        r1 = 0.0 if self.agent_1_arrived else -0.01
        r2 = 0.0 if self.agent_2_arrived else -0.01
        
        # Arrival Management
        if self.agent_1_arrived: prop_1 = self.goal_1
        if self.agent_2_arrived: prop_2 = self.goal_2

        # Wall Collision (Strong penalty but doesn't block learning)
        if prop_1 in self.obstacles and not self.agent_1_arrived:
            r1 -= 0.5
            prop_1 = self.agent_1_pos
        if prop_2 in self.obstacles and not self.agent_2_arrived:
            r2 -= 0.5
            prop_2 = self.agent_2_pos

        # Agent Collision
        if prop_1 == prop_2 and not (self.agent_1_arrived and self.agent_2_arrived):
            r1 -= 1.0
            r2 -= 1.0
            prop_1, prop_2 = self.agent_1_pos, self.agent_2_pos
        
        if prop_1 == self.agent_2_pos and prop_2 == self.agent_1_pos:
             r1 -= 1.0
             r2 -= 1.0
             prop_1, prop_2 = self.agent_1_pos, self.agent_2_pos

        self.agent_1_pos = prop_1
        self.agent_2_pos = prop_2

        # INTELLIGENT REWARD: Allow detours!
        if not self.agent_1_arrived:
            dist_1_new = self._get_distance(self.agent_1_pos, self.goal_1)
            if dist_1_new < dist_1_old:
                r1 += 0.1
            # If it moves away, it only gets the time penalty (-0.01).
            # This allows bypassing obstacles without "suffering" too much.

        if not self.agent_2_arrived:
            dist_2_new = self._get_distance(self.agent_2_pos, self.goal_2)
            if dist_2_new < dist_2_old:
                r2 += 0.1

        # Check Goals
        if self.agent_1_pos == self.goal_1 and not self.agent_1_arrived:
            r1 += 20.0
            self.agent_1_arrived = True
        
        if self.agent_2_pos == self.goal_2 and not self.agent_2_arrived:
            r2 += 20.0
            self.agent_2_arrived = True

        done = (self.agent_1_arrived and self.agent_2_arrived) or (self.step_count >= MAX_STEPS)
        return self._get_obs(), (r1, r2), done, False, {}

    def _move(self, pos, move):
        new_r = max(0, min(self.grid_size - 1, pos[0] + move[0]))
        new_c = max(0, min(self.grid_size - 1, pos[1] + move[1]))
        return (new_r, new_c)

# ==========================================
# 4. THE AGENT (Replay Buffer + Learning)
# ==========================================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.epsilon = EPSILON_START
        
        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, is_training=True):
        if is_training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Q(s, a)
        current_q = self.policy_net(states).gather(1, actions)

        # Target: r + gamma * max(Q_target(s', a'))
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (GAMMA * max_next_q * (1 - dones))

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

# ==========================================
# 5. TRAINING & VISUALIZATION
# ==========================================
def train_dqn(live_plot=False):
    env = WarehouseGymEnv()
    
    # Observation is a vector of size 6
    agent_1 = DQNAgent(state_dim=10, action_dim=4)
    agent_2 = DQNAgent(state_dim=10, action_dim=4)

    rewards_history = []
    
    if live_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))

    print("🧠 Deep RL Training Started (DQN)...")

    best_reward = -float('inf') 

    for episode in range(NUM_EPISODES):
        obs = env.reset() # returns (obs1, obs2)
        state_1, state_2 = obs
        total_reward = 0
        
        for step in range(MAX_STEPS):
            action_1 = agent_1.select_action(state_1)
            action_2 = agent_2.select_action(state_2)
            
            next_obs, rewards, done, _, _ = env.step(action_1, action_2)
            next_state_1, next_state_2 = next_obs
            
            # Store experience
            agent_1.store_transition(state_1, action_1, rewards[0], next_state_1, done)
            agent_2.store_transition(state_2, action_2, rewards[1], next_state_2, done)
            
            # Optimization step
            agent_1.train()
            agent_2.train()
            
            state_1, state_2 = next_state_1, next_state_2
            total_reward += sum(rewards)

            if live_plot and episode % 50 == 0:
                update_plot(ax, env)
            
            if done: break
        
        # Update Target Networks & Epsilon
        if episode % TARGET_UPDATE == 0:
            agent_1.update_target_network()
            agent_2.update_target_network()
        
        agent_1.decay_epsilon()
        agent_2.decay_epsilon()
        rewards_history.append(total_reward)

        EVAL_WINDOW = 100
        
        if len(rewards_history) >= EVAL_WINDOW:
            recent_rewards = rewards_history[-EVAL_WINDOW:]
            
            avg_reward = np.mean(recent_rewards)
            
            # 2. Calculate the SUCCESS RATE 
            # We assume that a reward > 10 means it reached the goal
            # (because the bonus is +20, so if you have >10 you almost certainly got the goal)
            success_count = sum(1 for r in recent_rewards if r > 10.0)
            success_rate = success_count / EVAL_WINDOW  # Es: 0.95 (95%)
            
            # Save ONLY if the success rate is high AND the average is good
            # This ignores "lucky" or unstable models
            if success_rate >= 0.90 and avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent_1.policy_net.state_dict(), os.path.join(OUTPUT_DIR, "best_agent1.pth"))
                torch.save(agent_2.policy_net.state_dict(), os.path.join(OUTPUT_DIR, "best_agent2.pth"))
                
                print(f"🌟 Ep {episode}: ROBUST MODEL! Success Rate: {success_rate*100:.1f}% | Avg Reward: {avg_reward:.2f}")

    # Save Models
    torch.save(agent_1.policy_net.state_dict(), os.path.join(OUTPUT_DIR, "dqn_agent1.pth"))
    torch.save(agent_2.policy_net.state_dict(), os.path.join(OUTPUT_DIR, "dqn_agent2.pth"))
    
    plt.figure()
    plt.plot(rewards_history)
    plt.title("DQN Learning Curve")
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve.png"))
    print("✅ Training Complete.")

def update_plot(ax, env):
    ax.clear()
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.invert_yaxis()
    ax.grid(True)
    
    for obs in env.obstacles:
        ax.add_patch(patches.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1, color='black'))
    
    # Agent 1
    ax.add_patch(patches.Circle((env.agent_1_pos[1], env.agent_1_pos[0]), 0.3, color='blue'))
    ax.text(env.goal_1[1], env.goal_1[0], "G1", color='blue', fontweight='bold', ha='center')

    # Agent 2
    ax.add_patch(patches.Circle((env.agent_2_pos[1], env.agent_2_pos[0]), 0.3, color='red'))
    ax.text(env.goal_2[1], env.goal_2[0], "G2", color='red', fontweight='bold', ha='center')
    
    plt.draw()
    plt.pause(0.001)

def run_demo():
    print("🎥 Running Demo with Trained Models (10 Test Runs)...")
    env = WarehouseGymEnv()
    
    agent_1 = DQNAgent(state_dim=10, action_dim=4)
    agent_2 = DQNAgent(state_dim=10, action_dim=4)
    
    try:
        path_a1 = os.path.join(OUTPUT_DIR, "best_agent1.pth")
        path_a2 = os.path.join(OUTPUT_DIR, "best_agent2.pth")
        
        agent_1.policy_net.load_state_dict(torch.load(path_a1, map_location=device))
        agent_2.policy_net.load_state_dict(torch.load(path_a2, map_location=device))
        
        agent_1.policy_net.eval()
        agent_2.policy_net.eval()
        print("✅ Models loaded successfully.")
    except FileNotFoundError:
        print("❌ No models found. Train first!")
        return

    plt.ion()
    fig, ax = plt.subplots(figsize=(6,6))
    
    success_count = 0
    total_runs = 100
    
    for i in range(1, total_runs + 1):
        print(f"\n🎬 TEST RUN {i}/{total_runs}")
        obs = env.reset()
        state_1, state_2 = obs
        
        done = False
        steps = 0
        max_steps_demo = 50  
        
        while not done and steps < max_steps_demo:
            action_1 = agent_1.select_action(state_1, is_training=False)
            action_2 = agent_2.select_action(state_2, is_training=False)
            
            obs, _, done, _, _ = env.step(action_1, action_2)
            state_1, state_2 = obs
            
            update_plot(ax, env)
            plt.title(f"Test {i}/{total_runs} | Step {steps}")
            plt.pause(0.1) 
            
            steps += 1
        
        if done:
            print(f"🏆 Run {i}: SUCCESS! (In {steps} steps)")
            success_count += 1
            plt.pause(1.0) 
        else:
            print(f"💀 Run {i}: FAILED (Max steps reached or stuck)")
            plt.pause(0.5)

    plt.ioff()
    plt.close()
    
    print(f"\n📊 FINAL RESULT: {success_count}/{total_runs} Successful Missions")

def run_fast_benchmark():
    print("🚀Fast Benchmark")
    
    env = WarehouseGymEnv()

    agent_1 = DQNAgent(state_dim=10, action_dim=4)
    agent_2 = DQNAgent(state_dim=10, action_dim=4)
    
    try:
        agent_1.policy_net.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_agent1.pth"), map_location=device))
        agent_2.policy_net.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_agent2.pth"), map_location=device))
        agent_1.policy_net.eval()
        agent_2.policy_net.eval()
    except FileNotFoundError:
        print("❌ Models not found")
        return

    NUM_EPISODES = 100
    success_count = 0
    collision_count = 0
    total_steps_cumulative = 0

    for i in range(NUM_EPISODES):
        obs = env.reset()
        state_1, state_2 = obs
        done = False
        steps = 0
        max_steps = 50 
        
        episode_has_collision = False
        
        while not done and steps < max_steps:
            action_1 = agent_1.select_action(state_1, is_training=False)
            action_2 = agent_2.select_action(state_2, is_training=False)
            
            next_obs, rewards, done, _, _ = env.step(action_1, action_2)
            state_1, state_2 = next_obs
            
            if rewards[0] <= -0.5 or rewards[1] <= -0.5:
                episode_has_collision = True
            
            steps += 1
        
        if env.agent_1_arrived and env.agent_2_arrived:
            success_count += 1
            total_steps_cumulative += steps
        
        if episode_has_collision:
            collision_count += 1
            
        if (i + 1) % 10 == 0:
            print(f"   ...Completed {i + 1}/{NUM_EPISODES} episodes")

    success_rate = (success_count / NUM_EPISODES) * 100
    avg_steps = total_steps_cumulative / success_count if success_count > 0 else 0
    
    print("\n" + "="*40)
    print("📊 Final Results")
    print("="*40)
    print(f"✅ Success Rate:     {success_rate:.1f}%")
    print(f"⚡ Avg Steps:        {avg_steps:.1f}")
    print(f"💥 Collision Rate:   {collision_count}%")
    print("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'demo','fast-demo'])
    args = parser.parse_args()

    if args.mode == 'train':
        train_dqn(live_plot=False) # Set True to watch it learn
    elif args.mode =='demo':
        run_demo()
    elif args.mode == 'fast-demo':
        run_fast_benchmark()


# python main_dqn.py --mode train
# python main_dqn.py --mode demo
# python main_dqn.py --mode fast-demo