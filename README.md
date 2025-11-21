# 🤖 Multi-Agent Deep Reinforcement Learning for Warehouse Logistics

Project Status: Active R&D Prototype for Autonomous Pathfinding using Deep Q-Networks (DQN).

## 📖 Overview

This project implements a Multi-Agent Reinforcement Learning (MARL) system to solve complex navigation tasks in a constrained warehouse environment.

Unlike traditional pathfinding algorithms (like A*), this system uses Deep Q-Networks (DQN) to allow agents to learn optimal policies through trial and error. The agents must navigate from random starting positions to dynamic goals while avoiding static obstacles and—crucially—avoiding collisions with each other.

This project serves as a proof-of-concept for decentralized autonomous robotic systems in logistics.

## 🎯 Key Features

- Deep Q-Learning (DQN): Implements neural networks (PyTorch) to approximate Q-values, replacing tabular methods for better generalization.
- Multi-Agent Environment: Custom Gymnasium environment managing state transitions, rewards, and collision logic for multiple entities.
- Stabilized Training: Uses Experience Replay and Target Networks to mitigate the instability of learning from correlated data.
- Dynamic Generalization: Agents are trained on randomized start/goal positions, preventing rote memorization of specific paths.

## 🎥 Demo

Blue and Red agents navigating a bottleneck to reach their respective goals (G1, G2) without colliding.

## 🛠️ Technical Architecture
1. The Environment (Gymnasium)

    The environment models a 10x10 grid with a central "bottleneck" wall. It observes the state of the world and returns it to the agents.

    - State Space (Normalized): [Agent_X, Agent_Y, Goal_X, Goal_Y, Other_Agent_X, Other_Agent_Y]
    - Action Space (Discrete): 0: Up, 1: Down, 2: Left, 3: Right, 4: Stay

2. The Agent (DQN)

    Each agent operates an independent Neural Network.
    - Architecture: 3-Layer Fully Connected Network (Input $\to$ 128 $\to$ 128 $\to$ Output).
    - Loss Function: Mean Squared Error (MSE) between Current Q and Target Q.
    - Optimizer: Adam ($\alpha=0.001$).

3. Reward Structure

    The reward function is designed to balance speed with safety:
    - Goal Reached: +10.0
    - Time Step: -0.01 (Incentivizes speed)
    - Wall Collision: -0.5
    - Agent Collision: -1.0

## 📊 Performance & Learning Curve

The agents utilize a Replay Buffer to store transitions $(s, a, r, s', d)$. This breaks the temporal correlation of data, leading to more stable convergence.(You can insert your learning_curve.png here)

## 🚀 Installation & Usage

Ensure you have Python 3.8+ installed.

<i> pip install numpy matplotlib gymnasium torch </i>

<b> Training the Agents </b>

To train the agents from scratch. This uses an Epsilon-Greedy strategy to balance exploration and exploitation. Training typically converges within 1000-2000 episodes.



<i>  python main_dqn.py --mode train </i>
    
<b> Running the Demo </b>

To visualize the trained models acting in the environment:

<i> python main_dqn.py --mode demo </i>


## 🔮 Future Improvements

CNN Integration: Replace coordinate inputs with a visual grid input (Convolutional Neural Networks) to handle arbitrary map shapes.

Centralized Training, Decentralized Execution (CTDE): Implement algorithms like MADDPG to improve cooperation in tighter spaces.

Curriculum Learning: Gradually increase obstacle density as the agents improve.