import json
import fire
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style


def main(json_file='pondering_stats.json', threshold=0.5):
    with open(json_file) as fh:
        j = json.load(fh)
    print(j.keys())
    max_length = len(j.keys()) - 1
    prev_survive_probs, last_survive_probs = [], []
    prev_exit_probs, last_exit_probs = [], []
    for iter_num, accept_length in enumerate(j['a']):
        survive = 1.0
        for i in range(accept_length):
            e_i = j[f'e{i}'][iter_num]
            survive *= (1 - e_i)
            prev_exit_probs.append(e_i)
            prev_survive_probs.append(survive)
            color = Fore.RED if e_i > threshold else Style.RESET_ALL
            print(f'{color}{round(e_i, 2)}{Style.RESET_ALL}', end=' ')
        print(Fore.YELLOW + '| ', end=Style.RESET_ALL)
        if accept_length > 0:
            prev_exit_probs.pop()
            prev_survive_probs.pop()
            last_exit_probs.append(e_i)
            last_survive_probs.append(survive)
        for i in range(accept_length, max_length):
            e_i = j[f'e{i}'][iter_num]
            color = Fore.RED if e_i > threshold else Style.RESET_ALL
            print(f'{color}{round(e_i, 2)}{Style.RESET_ALL}', end=' ')
        print()

    fig, ax = plt.subplots(1, 4, figsize=(9, 2))
    ax[0].hist(prev_exit_probs, bins=10)
    ax[0].set_xlabel("P(exit) on accept")
    ax[0].set_ylabel("Frequency")

    ax[1].hist(last_exit_probs, bins=10)
    ax[1].set_xlabel("P(exit) on reject")

    ax[2].hist(prev_survive_probs, bins=10)
    ax[2].set_xlabel("P(survival) on accept")

    ax[3].hist(last_survive_probs, bins=10)
    ax[3].set_xlabel("P(survival) on reject")

    plt.tight_layout()
    plt.savefig(f'{json_file}.png')


if __name__ == '__main__':
    fire.Fire(main)
