#!/usr/bin/env python3 

import os
import rclpy
from ros2_env import (
    GazeboEnv,
    Odom_subscriber,
    Velodyne_subscriber,
    SetModelStateClient
)
from TD3 import td3
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayBuffer
import threading

def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        print(f"evaluating episode {_}", flush=True)
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            #env.get_logger().info(f"action : {action}")
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................", flush=True)
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: avg_reward %f, avg_col %f"
        % (eval_episodes, epoch, avg_reward, avg_col), flush=True
    )
    print("..............................................", flush=True)
    return avg_reward

if __name__ == '__main__':

    rclpy.init(args=None)

    seed = 0  # Random seed number
    eval_freq = 5e3  # After how many steps to perform the evaluation
    max_ep = 500  # maximum number of steps per episode
    eval_ep = 10  # number of episodes for evaluation
    max_timesteps = 5e6  # Maximum number of steps to perform
    expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
    expl_decay_steps = (
        500000  # Number of steps over which the initial exploration noise will decay over
    )
    expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
    batch_size = 40  # Size of the mini-batch
    discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
    tau = 0.005  # Soft target update variable (should be close to 0)
    policy_noise = 0.2  # Added noise for exploration
    noise_clip = 0.5  # Maximum clamping values of the noise
    policy_freq = 2  # Frequency of Actor network updates
    buffer_size = 1e6  # Maximum size of the buffer
    file_name = "td3_velodyne"  # name of the file to store the policy
    save_model = True  # Weather to save the model or not
    load_model = False  # Weather to load a stored model
    random_near_obstacle = True  # To take random actions near obstacles or not

    # Create the network storage folders
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_model and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Create the training environment
    env = GazeboEnv()
    environment_dim = 20
    robot_dim = 4
    time.sleep(5)

    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1

    #alpha = 0.9  # Smoothing factor for EMA
    #smoothed_reward = 0  # Initialize the smoothed reward

    writer = SummaryWriter(log_dir="./runs")
    # Create the network
    network = td3(state_dim, action_dim, max_action)
    # Create a replay buffer
    replay_buffer = ReplayBuffer(buffer_size, seed)
    if load_model:
        try:
            print("Will load existing model.", flush=True)
            network.load(file_name, "./pytorch_models")
        except:
            print("\033[33mCould not load the stored model parameters, initializing training with random parameters\033[0m", flush=True)

    # Create evaluation data store
    evaluations = []

    timestep = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    epoch = 1

    count_rand_actions = 0
    random_action = []

    
    odom_subscriber = Odom_subscriber()
    velodyne_subscriber = Velodyne_subscriber()
    set_model_state_client = SetModelStateClient()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(velodyne_subscriber)
    executor.add_node(set_model_state_client)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    rate = odom_subscriber.create_rate(2)
    try:
        while rclpy.ok():
            if timestep < max_timesteps:
                # On termination of episode
                if done:
                    print(f"Done. timestep : {timestep}", flush=True)
                    if timestep != 0:
                        #env.get_logger().info(f"train")
                        network.train(
                        replay_buffer,
                        episode_timesteps,
                        batch_size,
                        discount,
                        tau,
                        policy_noise,
                        noise_clip,
                        policy_freq,
                        )

                    if timesteps_since_eval >= eval_freq:
                        print("\033[33mValidating\033[0m", flush=True)
                        timesteps_since_eval %= eval_freq
                        evaluations.append(
                            evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
                        )

                        network.save(file_name, directory="./pytorch_models")
                        np.save("./results/%s" % (file_name), evaluations)
                        epoch += 1

                    state = env.reset()
                    done = False

                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1

                # add some exploration noise
                if expl_noise > expl_min:
                    expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

                action = network.get_action(np.array(state))
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                     -max_action, max_action
                )

                # If the robot is facing an obstacle, randomly force it to take a consistent random action.
                # This is done to increase exploration in situations near obstacles.
                # Training can also be performed without it
                if random_near_obstacle:
                    if (
                        np.random.uniform(0, 1) > 0.85
                        and min(state[4:-8]) < 0.6
                        and count_rand_actions < 1
                    ):
                        count_rand_actions = np.random.randint(8, 15)
                        random_action = np.random.uniform(-1, 1, 2)

                    if count_rand_actions > 0:
                        count_rand_actions -= 1
                        action = random_action
                        action[0] = -1

                # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
                a_in = [(action[0] + 1) / 2, action[1]]
                next_state, reward, done, target = env.step(a_in)
                done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
                done = 1 if episode_timesteps + 1 == max_ep else int(done)
                episode_reward += reward

                # Calculate EMA smoothing for reward
                #smoothed_reward = alpha * reward + (1 - alpha) * smoothed_reward

                writer.add_scalar("Baseline_TD3", reward, episode_num)
                # Save the tuple in replay buffer
                replay_buffer.add(state, action, reward, done_bool, next_state)

                # Update the counters
                state = next_state
                episode_timesteps += 1
                timestep += 1
                timesteps_since_eval += 1

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()