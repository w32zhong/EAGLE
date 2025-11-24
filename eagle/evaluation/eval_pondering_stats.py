import json
import fire
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from colorama import Fore, Style


def probs(json_file='pondering_stats.json', threshold=0.8):
    with open(json_file) as fh:
        j = json.load(fh)
    max_length = len(j.keys()) - 1
    for i in range(max_length):
        assert len(j[f'e{i}']) == len(j['a'])
    print(f'max(accept_length)={max(j['a'])}, max_length={max_length}')
    prev_survive_probs, last_survive_probs = [], []
    prev_exit_probs, last_exit_probs = [], []
    for iter_num, accept_length in enumerate(j['a']):
        if accept_length == 0: continue
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

    fig, ax = plt.subplots(1, 4, figsize=(16, 2))
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
    plt.savefig(f'{json_file}_probs.png')


def optimal(json_file='pondering_stats.json', use_linear_C=True):
    with open(json_file) as fh:
        j = json.load(fh)
    max_length = len(j.keys()) - 1

    if use_linear_C:
        C = [(1.16*(i+1) + 18.74) for i in range(max_length)]
    else:
        C = [42.07, 44.50, 46.58, 49.12, 51.28, 52.99, 55.59, 57.63, 59.75, 62.45, 64.18, 66.87, 68.46]
    print(C)

    data = []
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7,   0.8, 0.85, 0.90, 0.95, 1.0]
    for threshold in thresholds:
        speed_gain = []
        for iter_num, accept_length in enumerate(j['a']):
            if accept_length == 0: continue
            e = [j[f'e{i}'][iter_num] for i in range(max_length)]
            exit_length = next((i for i, e_i in enumerate(e) if e_i > threshold), max_length - 1) + 1

            static_speed = (accept_length + 1) / C[max_length - 1]
            dynamic_speed = (min(accept_length, exit_length) + 1) / C[exit_length - 1]
            speed_gain.append(dynamic_speed - static_speed)
        data.append(speed_gain)

    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    for i, speed_gain in enumerate(data):
        threshold = thresholds[i]
        ax[i // 5, i % 5].hist(speed_gain, bins=max_length)
        ax[i // 5, i % 5].set_title(f"threshold={threshold:.2f}")
        ax[i // 5, i % 5].set_xlabel("Speed Gain")

    plt.tight_layout()
    plt.savefig(f'{json_file}_optimal_linear{use_linear_C}.png')


def costs(json_file='pondering_stats.json'):
    with open(json_file) as fh:
        j = json.load(fh)
    max_exit_i = max(j['exit@'])

    C_hist = defaultdict(list)
    for iter_num, exit_at in enumerate(j['exit@']):
        C_hist[exit_at].append(j['C'][iter_num])

    C_hist = [np.array(C_hist[i]) for i in range(max_exit_i + 1)]
    C_mean = [h.mean().item() for h in C_hist]
    C_std = [h.std().item() for h in C_hist]

    x = np.arange(len(C_mean))
    y = np.array(C_mean)
    a, b = np.polyfit(x, y, 1) # degree 1 → linear
    interpolated = [(a * x + b).item() for x in range(len(C_mean))]

    print(C_mean)
    print(C_std)
    print(a, b)
    print(interpolated)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(x, y, color='red')
    ax.plot(x, interpolated)
    plt.tight_layout()
    plt.savefig(f'{json_file}_costs.png')


if __name__ == '__main__':
    fire.Fire(dict(probs=probs, optimal=optimal, costs=costs))
