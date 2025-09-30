import numpy as np
import random
import imageio
import json
import datetime
import pickle5 as pickle
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from tqdm.notebook import tqdm


def record_video(env, Qtable, out_directory, fps=1):
  """
  Generate a replay video of the agent.
  :param env: The environment.
  :param Qtable: The Q-table of our agent.
  :param out_directory: Directory to save the video.
  :param fps: Frames per second.
  """
  images = []
  terminated = False
  truncated = False
  state, info = env.reset(seed=random.randint(0, 500))
  img = env.render()
  images.append(img)
  while not terminated and not truncated:
    action = np.argmax(Qtable[state][:])
    state, reward, terminated, truncated, info = env.step(action)
    img = env.render()
    images.append(img)
  imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)


def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
  """
  Evaluate the agent for `n_eval_episodes` episodes and returns average reward and std of reward.
  :param env: The evaluation environment.
  :param max_steps: Maximum number of steps per episode.
  :param n_eval_episodes: Number of episodes to evaluate the agent.
  :param Q: The Q-table.
  :param seed: The evaluation seed array.
  """
  episode_rewards = []
  for episode in tqdm(range(n_eval_episodes)):
    if seed:
      state, info = env.reset(seed=seed[episode])
    else:
      state, info = env.reset()
    
    terminated = False
    truncated = False
    total_rewards_ep = 0

    for step in range(max_steps):
      action = np.argmax(Q[state][:])
      new_state, reward, terminated, truncated, info = env.step(action)
      total_rewards_ep += reward
      
      if terminated or truncated:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)
  return mean_reward, std_reward


def push_to_hub(repo_id, model, env):
    """
    Evaluate, Generate a video and Upload a model to Hugging Face Hub.
    :param repo_id: id of the model repository from the Hugging Face Hub.
    :param model: The model dictionary containing the Q-table and hyperparameters.
    :param env: The environment.
    """
    _, repo_name = repo_id.split("/")

    api = HfApi()

    # Step 1: Create the repo
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)

    # Step 2: Download files
    repo_local_path = Path(snapshot_download(repo_id=repo_id))

    # Step 3: Save the model
    if env.spec.kwargs.get("map_name"):
        model["map_name"] = env.spec.kwargs.get("map_name")
        if env.spec.kwargs.get("is_slippery", "") is False:
            model["slippery"] = False
    
    with open(repo_local_path / "q-learning.pkl", "wb") as f:
        pickle.dump(model, f)

    # Step 4: Evaluate the model and build JSON
    mean_reward, std_reward = evaluate_agent(
        env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"]
    )

    evaluate_data = {
        "env_id": model["env_id"],
        "mean_reward": mean_reward,
        "n_eval_episodes": model["n_eval_episodes"],
        "eval_datetime": datetime.datetime.now().isoformat()
    }

    with open(repo_local_path / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    # Step 5: Create the model card
    env_name = model["env_id"]
    if env.spec.kwargs.get("map_name"):
        env_name += "-" + env.spec.kwargs.get("map_name")
    if env.spec.kwargs.get("is_slippery", "") is False:
        env_name += "-no_slippery"

    metadata = {
        "tags": [env_name, "q-learning", "reinforcement-learning", "custom-implementation"]
    }

    eval_metadata = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_name,
        dataset_id=env_name,
    )
    
    metadata = {**metadata, **eval_metadata}

    model_card = f"""
# Q-Learning Agent for {env_name}

This is a trained model of a Q-Learning agent playing **{env_name}**.

## Usage

```python
import gymnasium as gym
import pickle

# Load the model from the Hub
# model = load_from_hub(repo_id="{repo_id}", filename="q-learning.pkl")

# Create the environment
# env = gym.make(model["env_id"], is_slippery=False)