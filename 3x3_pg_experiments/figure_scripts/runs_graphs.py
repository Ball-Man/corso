import sys
import os.path
import re

from tbparse import SummaryReader
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

agent_names_map = {'minmax1': 'MM-$1$', 'minmax2': 'MM-$2$',
                   'minmax3': 'MM-$3$', 'random': 'Random'}
learning_agent_names_map = {
    'vreinforce': 'REINFORCE',
    'wbaseline': 'REINFORCE with baseline',
    'ac': 'one step actor-critic',
    'ppo': 'PPO'}
prefix_to_learning_agent_re = re.compile(r'([a-zA-Z0-9]*).*')

sns.set_palette('muted')


def get_df(log_dir, prefix, df_name) -> pd.DataFrame:
    """Get (and cache) dataframe from tboard runs."""
    # Cache df due to the terrible performance of SummaryReader
    from_file = os.path.exists(df_name)
    if from_file:
        print('loading df from file:', df_name)
        df = pd.read_parquet(df_name)
    else:
        print('parsing summaries')
        reader = SummaryReader(log_dir, extra_columns={'dir_name'})
        df = reader.scalars

    # Remove draws as in this setup there are no draws
    df = df[~df.dir_name.str.contains('draw')]

    # Extract useful columns from dir_name
    extracted = df.dir_name.str.extract(
        prefix + r'(?P<seed>\d+)(?:/eval_vs_(?P<vs>.*)_wins_as_p(?P<as>\d))?')
    df = df.drop(columns=['dir_name'])
    df = pd.concat((df, extracted), axis=1)

    # Handle NaN values
    df['as'] = df['as'].fillna(0)
    df['vs'] = df['vs'].fillna('')

    # Correct typings
    df['as'] = pd.to_numeric(df['as'])
    df['vs'] = df['vs'].astype(str)

    # eval_groups = (
    #     df[df.vs != '']
    #     .groupby(['tag', 'step', 'as', 'vs'])
    #     .agg(value_mean=('value', np.mean), value_std=('value', np.std))
    # )

    if not from_file:
        print('dumping to file:', df_name)
        df.to_parquet(df_name)

    return df


if __name__ == '__main__':
    log_dir = sys.argv[1]
    prefix = sys.argv[2]
    df_name = log_dir + '.parquet'
    learning_agent_name = learning_agent_names_map.get(
        prefix_to_learning_agent_re.fullmatch(prefix).group(1),
        'Agent')

    df = get_df(log_dir, prefix, df_name)

    wins_vs_random_p1 = df[(df['as'] == 1) & (df['vs'] == 'random')]
    wins_vs_minmax1_p1 = df[(df['as'] == 1) & (df['vs'] == 'minmax1')]
    wins_vs_minmax2_p1 = df[(df['as'] == 1) & (df['vs'] == 'minmax2')]
    wins_vs_minmax3_p1 = df[(df['as'] == 1) & (df['vs'] == 'minmax3')]

    wins_p1 = (wins_vs_random_p1, wins_vs_minmax1_p1, wins_vs_minmax2_p1,
               wins_vs_minmax3_p1)

    wins_vs_random_p2 = df[(df['as'] == 2) & (df['vs'] == 'random')]
    wins_vs_minmax1_p2 = df[(df['as'] == 2) & (df['vs'] == 'minmax1')]
    wins_vs_minmax2_p2 = df[(df['as'] == 2) & (df['vs'] == 'minmax2')]
    wins_vs_minmax3_p2 = df[(df['as'] == 2) & (df['vs'] == 'minmax3')]

    wins_p2 = (wins_vs_minmax1_p2, wins_vs_minmax2_p2, wins_vs_minmax3_p2,
               wins_vs_minmax3_p2)

    # VS all, no error bars to avoid clutter
    for data in wins_p1:
        sns.lineplot(x=data.step, y=data.value, errorbar=None)

    plt.ylim(0, 1)
    plt.xlabel('train episodes')
    plt.ylabel('win rate')
    plt.legend(['Random', 'MM-$1$', 'MM-$2$', 'MM-$3$'])
    plt.title(f'Wins of {learning_agent_name} agent as first player')

    plt.savefig(f'{prefix}vs_all_as_p1.svg', format='svg')

    plt.figure()
    for data in wins_p2:
        sns.lineplot(x=data.step, y=data.value, errorbar=None)

    plt.ylim(0, 1)
    plt.xlabel('train episodes')
    plt.ylabel('win rate')
    plt.legend(['Random', 'MM-$1$', 'MM-$2$', 'MM-$3$'])
    plt.title(f'Wins of {learning_agent_name} agent as second player')

    plt.savefig(f'{prefix}vs_all_as_p2.svg', format='svg')

    # Individual wins with error bars (show one or twice in the report?)
    for data in wins_p1:
        plt.figure()
        sns.lineplot(x=data.step, y=data.value)

        plt.ylim(0, 1)
        plt.xlabel('train episodes')
        plt.ylabel('win rate')
        opponent = data.vs.unique()[0]
        plt.title(f'Wins of {learning_agent_name} agent as first player vs '
                  f'{agent_names_map[opponent]}')

        plt.savefig(f'{prefix}vs_{opponent}_as_p1.svg', format='svg')

    for data in wins_p2:
        plt.figure()
        sns.lineplot(x=data.step, y=data.value)

        plt.ylim(0, 1)
        plt.xlabel('train episodes')
        plt.ylabel('win rate')
        opponent = data.vs.unique()[0]
        plt.title(f'Wins of {learning_agent_name} agent as second player vs '
                  f'{agent_names_map[opponent]}')

        plt.savefig(f'{prefix}vs_{opponent}_as_p2.svg', format='svg')
