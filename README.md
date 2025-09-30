# Reinforcement Learning - QLearning: Taxi-v3🚕

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/notebooks/unit2/taxi.png" alt="Taxi Environment" width="400"/>

## Overview

This project implements the **Q-Learning algorithm** to solve the `Taxi-v3` environment from the [Gymnasium library](https://gymnasium.farama.org/environments/toy_text/taxi/). The `Taxi-v3` challenge involves a taxi navigating a 5x5 grid to pick up a passenger from one location and drop them off at a destination.

This is a step up in complexity from simpler grid worlds like FrozenLake, featuring:
- A larger state space (500 states).
- A more complex action space (including pickup and drop-off actions).
- A nuanced reward system with penalties for incorrect actions.

The included Jupyter Notebook provides a detailed, educational walkthrough of the entire process, from understanding the environment to training and publishing a high-performing agent.

## Core Learning Objectives

- **Applying Q-Learning to a More Complex Problem**: Adapting the algorithm to handle a larger state-action space.
- **Hyperparameter Tuning**: Experimenting with `learning_rate`, `gamma`, and exploration parameters to achieve better results.
- **Structured RL Implementation**: Using helper functions and a clean notebook structure for a scalable project.
- **Performance Evaluation**: Using a fixed evaluation seed to ensure reproducible and comparable results.
- **Community Contribution**: Sharing the final trained model on the Hugging Face Hub, complete with a model card and performance metrics.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/q-learning-taxi-v3.git
    cd q-learning-taxi-v3
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Open and execute the cells in the `taxi_v3_q_learning.ipynb` notebook. It contains all the code and explanations needed to train your agent. For the certification part of the Deep RL Course, the goal is to achieve a score of **>= 4.5** on the leaderboard.

## Results and Leaderboard

The performance of the trained agent is evaluated based on its mean reward over 100 episodes using a predefined seed. You can see how your model stacks up against others on the [Deep RL Leaderboard](https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard).

---
*This project is an assignment from the [Hugging Face Deep RL Course](https://huggingface.co/deep-rl-course).*
