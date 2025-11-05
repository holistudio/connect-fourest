import numpy as np
import torch
from tqdm import tqdm
import imageio

from pettingzoo.classic import connect_four_v3

# TODO: import you agent1 model here
from agent import PPOAgent

# TODO: import you agent2 model here (optional)
# from agent import PPOAgent


def evaluator(n_episodes=100, seed=42, agent1=None, agent2=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = connect_four_v3.env()

    agent_list = [agent1, agent2]

    win_loss_draw = {
        "player_0": [0,0,0],
        "player_1": [0,0,0]
    }
    idx = 0

    for episode in tqdm(range(n_episodes)):
        env.reset(seed=42)
        for a in env.agent_iter():
            agent = agent_list[idx]

            observation, reward, termination, truncation, info = env.last()
            # invalid action masking is optional and environment-dependent
            if "action_mask" in info:
                mask = info["action_mask"]
            elif isinstance(observation, dict) and "action_mask" in observation:
                mask = observation["action_mask"]
            else:
                mask = None

            
            if termination or truncation:
                action = None
                if truncation:
                    _, value, _ = agent.step(o, mask)
                    win_loss_draw[a][2] += 1 
                else:
                    value = 0
                    if reward > 0:
                        win_loss_draw[a][0] += 1 
                    elif reward < 0:
                        win_loss_draw[a][1] += 1 
                    else:
                        win_loss_draw[a][2] += 1 
            else:
                if agent is None:
                    action = env.action_space(a).sample(mask)
                else:
                    # TODO: Modify any code for your agent's action function (no policy steps)
                    # The key is that your agent's act() function MUST take in `observation` 
                    # from environment as defined by PettingZoo
                    # to play nicely with other agent classes that do the same
                    action = agent.act(observation, mask)

            env.step(action)
            idx += 1
            if idx >= len(agent_list):
                idx = 0
    env.close()
    print(f"Player 0 Win Rate = {win_loss_draw['player_0'][0]*100/sum(win_loss_draw['player_0']):.1f}%")
    print(f"Player 1 Win Rate = {win_loss_draw['player_1'][0]*100/sum(win_loss_draw['player_1']):.1f}%")
    print(win_loss_draw)

def playback(agent1=None, agent2=None):
    env = connect_four_v3.env(render_mode="rgb_array")
    frames = []  # store frames here

    env.reset(seed=42)

    agent_list = [agent1, agent2]
    idx = 0

    
    for a in env.agent_iter():
        agent = agent_list[idx]

        observation, reward, termination, truncation, info = env.last()
        # Get the current frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        # invalid action masking is optional and environment-dependent
        if "action_mask" in info:
            mask = info["action_mask"]
        elif isinstance(observation, dict) and "action_mask" in observation:
            mask = observation["action_mask"]
        else:
            mask = None
        

        if termination or truncation:
            action = None
            env.reset()
            break
        else:
            if agent is None:
                action = env.action_space(a).sample(mask)
            else:

                # TODO: Modify any code for your agent's action function (no policy steps)
                # The key is that your agent's act() function MUST take in `observation` 
                # from environment as defined by PettingZoo
                # to play nicely with other agent classes that do the same
                action = agent.act(observation, mask)
                env.step(action)
    
    env.close()
    
    # Save video to file
    output_path = "agent_vs_agent.mp4"
    imageio.mimsave(output_path, frames, fps=1)  # adjust FPS as desired
    print()
    print(f"Watch the agents play in {output_path}")

if __name__ == '__main__':
    env = connect_four_v3.env()

    # TODO: initialize your agent1 however you want here
    agent1 = PPOAgent(env.observation_spaces['player_0']['observation'], 
                      env.action_spaces['player_0'], 
                      local_steps_per_epoch=100_000)
    agent1.load(module_filename='Conn4-P1_agent.pt.tar', buffer_filename="Conn4-P1_buffer.npz")
    
    agent2 = PPOAgent(env.observation_spaces['player_1']['observation'], 
                      env.action_spaces['player_1'], 
                      local_steps_per_epoch=100_000)
    agent1.load(module_filename='Conn4-P2_agent.pt.tar', buffer_filename="Conn4-P2_buffer.npz")
    
    evaluator(agent1=agent1)
    print()
    evaluator(agent2=agent2)
    print()
    
    if agent2 is not None:
        evaluator(agent1=agent1, agent2=agent2)
        print()
        playback(agent1=agent1, agent2=agent2)
        
        
