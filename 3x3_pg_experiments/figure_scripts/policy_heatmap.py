"""Produce a heatmap, given policy action distribution of one state."""
import sys

import torch
import seaborn as sns
from matplotlib import pyplot as plt

from corso.pg import PolicyNetwork
from corso.model import Corso, EMPTY_BOARD3X3

inspecting_state = Corso(EMPTY_BOARD3X3)


if __name__ == '__main__':
    policy_net_dir = sys.argv[1]
    agent_name = sys.argv[2]

    policy_net = PolicyNetwork.load(policy_net_dir)

    features = policy_net.state_features(inspecting_state)
    with torch.no_grad():
        _, distribution = policy_net(features)
    distribution = distribution.view(inspecting_state.height,
                                     inspecting_state.width)

    sns.heatmap(distribution, cmap='Blues', vmin=0, vmax=1, annot=True,
                linewidths=5)
    plt.title(f'First action distribution of {agent_name} agent')
    plt.savefig('agent_heatmap.svg', format='svg')
