![corso_banner_repo](https://github.com/Ball-Man/corso/assets/12080380/988af775-e6fd-4c3b-8f81-265979ffdd67)
# Corso

Virtual environment for the game of Corso. The game rules can be found [here](https://gist.github.com/Ball-Man/30c5cad7b906910522b84c1741ad9268).

This repository also serves as container for AI tools and experiments to obtain agents playing the game.

## Installation
Python >= 3.9 is required.

Automatic installation can be carried out as usual with pip:
```
pip install git+https://github.com/Ball-Man/corso
```
or by cloning the repository locally first (useful e.g. for editable/dev installations):
```
git clone https://github.com/Ball-Man/corso
cd corso
pip install .
```

## AI
Current AIs include:
* AlphaZero-like agent: superhuman player for the standard 5x5 game format (see `az_experiments/README.md`).
* Policy gradient agents (REINFORCE, AC, PPO): they can effectively learn the 3x3 version of the game (see `3x3_pg_experiments/README.md`).
* MinMax: classic approach based on minimax tree search. MinMax agents were mostly built for evaluation purposes. As such, they are not perfectly optimized. For example, they work on fixed depth, strategies like iterative deepening are entirely missing.

## Playing
After installation, it is possible to start a simple two-players session of the game:
```
python -m corso.cli
```
This will start a 5x5 game by default. For a 3x3 game run:
```
python -m corso.cli -H 3 -w 3
```

By deafult a fully manual game will start, requiring input for both players. Trivial AI players are available as well. For example, in order to challenge a random agent:
```
python -m corso.cli -p user random
```
More advanced agents, like neural ones, require custom scripts to setup a match (see [Custom game](#custom-game)).

For more info see `python -m corso.cli -h`.

### CLI game representation
When playing a CLI game, moves are represented by coordinates on the grid. In particular, coordinates shall be given as two positive integers separated by a space (row coordinate and column coordinate). Values start at `1`. For example:
* `1 2`
* `1 1`
* `2 3`
* `5 5`

Are valid moves

The board is represented as a matrix of characters. The meaning of the characters is:
* `O`: empty cell
* `A`: marble of player1
* `a`: dyed cell of player1
* `B`: marble of player2
* `b`: dyed cell of player2

Please refer to the [game rules](https://gist.github.com/Ball-Man/30c5cad7b906910522b84c1741ad9268) for the actual meaning of these tokens.

### Custom game
With the help of some code it is possible to instantiate a custom game, vs either a custom agent or an existing one provided by this repository. Here a few examples:

Play a 5x5 game against a MinMax agent with depth `3`. Human player starts first.
```py
from corso.cli import cli_game
from corso.minmax import MinMaxPlayer


if __name__ == '__main__':
    cli_game(player2=MinMaxPlayer(3))
```

Spectate a game against a MinMax player and a random player:
```py
from corso.cli import cli_game
from corso.model import RandomPlayer
from corso.minmax import MinMaxPlayer


if __name__ == '__main__':
    cli_game(player1=MinMaxPlayer(3), player2=RandomPlayer())
```

Play a game against a 3x3 policy gradient pretrained agent:
```py
from corso.cli import cli_game
from corso.model import Corso, EMPTY_BOARD3X3
from corso.pg import PolicyNetwork, PolicyNetworkPlayer


if __name__ == '__main__':
    starting_state = Corso(EMPTY_BOARD3X3)

    policy_net = PolicyNetwork.load('policy_net/')
    policy_player = PolicyNetworkPlayer(policy_net)

    cli_game(player1=policy_player,
             starting_state=starting_state)
```
Please note that `policy_net/` must be a pretrained policy trained and saved appropriately, which are not included in the library installation but can be downloaded from this repository. It is possible to download a variety of them from `3x3_pg_experiments/`. For example, try with `3x3_pg_experiments/ppo/policy_net_ppo_1749629150906108955`. A policy network must be a directory, containing two files: `model.pt`, `config.json`.

Similarly, to play against a 5x5 pretrained AlphaZero agent:
```py
from corso.cli import cli_game
from corso.model import Corso
from corso.az import AZConvNetwork, AZPlayer

NUM_PLAYOUTS = 100

if __name__ == '__main__':
    starting_state = Corso()

    policy_net = AZConvNetwork.load('policy_net/')
    policy_player = AZPlayer(policy_net, NUM_PLAYOUTS, 0.)

    cli_game(player1=policy_player,
             starting_state=starting_state)
```
Again, `policy_net/` must be a pretrained policy trained and saved appropriately, which can be downloaded from this repository. Try with the one in `az_experiments/5x5_64x4/az_network_5x5_64x4`.

## Code style
Code mostly follow the [PEP8](https://peps.python.org/pep-0008/) style guide. In particular, [Flake8](https://flake8.pycqa.org/en/latest/) with default settings is used as linter.

A simple equivalent VSCode setup (you will be prompted to install Flake):
```json
{
    "python.linting.flake8Enabled": true,
    "python.linting.enabled": true,
}
```
> [!WARNING]\
> Since October 2023, `python.linting` has been depreacted. To obtain the same result just install and enable the Flake8 extension.

## Aknowledgements
Corso was created to be featured in the videogame [Lone Planet](https://store.steampowered.com/app/1933170/Lone_Planet/).

AZ AI experiments were designed as project for the *Deep Learning* project work assignment, held at University
of Bologna. I would like to thank the professor of the related course, Andrea Asperti, for introducing me to Deep Learning.

PG AI experiments were designed as project for the *Automous and Adaptive Systems* course, held at University
of Bologna (2023). I would like to thank the professor of said course, Mirco Musolesi, for his
inspiring work.

I would also like to thank my colleague Michele Faedi for the great conversations on
related topics.
