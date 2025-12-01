import json
import fire
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style


def parse_data_lengths(j):
    exit_range, samples, = [], None
    keys = j.keys()
    for i in range(-1, len(keys)):
        key = f'e{i}'
        if key in keys:
            exit_range.append(i)
            if samples is not None:
                assert samples == len(j[key])
            else:
                samples = len(j[key])
        else:
            break
    max_accept_length = max(exit_range) + 1
    if max_accept_length != max(j['a']):
        print(Fore.YELLOW, """
              Warning: max_accept_length != max recorded accept_length!
              This may indicate the model is not performing as expected.
              """, Style.RESET_ALL)
    return exit_range, max_accept_length


def exit_condition(step, prob, threshold):
    if isinstance(threshold, tuple):
        threshold1, phase, threshold2 = threshold
        if step < phase:
            return (prob >= threshold1)
        else:
            return (prob >= threshold2)
    else:
        assert isinstance(threshold, float)
        return (prob >= threshold)


def calc_stats(j, exit_range, threshold, abort_on_first_exit=True):
    lengths, true_pos, false_pos, true_neg, false_neg = [], [], [], [], []
    for iter_num, accept_length in enumerate(j['a']):
        exit_length = None
        for i in exit_range:
            e_i = j[f'e{i}'][iter_num]
            color = Style.RESET_ALL
            if exit_condition(i, e_i, threshold):
                if i + 1 >= accept_length:
                    color += Fore.GREEN # good exit
                    true_pos.append(e_i)
                else:
                    color += Fore.RED # exit too early!
                    false_pos.append(e_i)

                if exit_length is None:
                    exit_length = i + 1
            else:
                if i + 1 >= accept_length:
                    false_neg.append(e_i)
                else:
                    true_neg.append(e_i)
            color += Back.MAGENTA if i >= accept_length else color
            print(f'{color}{e_i:.4f}{Style.RESET_ALL}',
                  end=' | ' if i == -1 else ' ')
            if abort_on_first_exit and exit_length is not None:
                break
        lengths.append((exit_length or accept_length, accept_length))
        print()
    return lengths, true_pos, false_pos, true_neg, false_neg


def calc_speed_gain(lengths, max_accept_length, C, bonus=1):
    speed_gain = []
    for exit_length, accept_length in lengths:
        static_speed = (accept_length + bonus) / C[max_accept_length]
        dynamic_speed = (exit_length + bonus) / C[exit_length]
        speed_gain.append(dynamic_speed - static_speed)
    return speed_gain


def probs(json_file='pondering_stats.json', threshold=0.9, A=1.365, B=23.618, abort_on_first_exit=False):
    with open(json_file) as fh:
        j = json.load(fh)
    exit_range, max_accept_length = parse_data_lengths(j)
    lengths, true_pos, false_pos, true_neg, false_neg = calc_stats(j, exit_range, threshold,
                                                          abort_on_first_exit=abort_on_first_exit)
    C = [A * (i+1) + B for i in exit_range]
    speed_gain = calc_speed_gain(lengths, max_accept_length, C)
    avg_speed_gain = sum(speed_gain) / (len(speed_gain) + 1e-5)
    print('avg_speed_gain', round(avg_speed_gain, 3))

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].hist(true_neg, bins=10, alpha=0.6, label="true neg", log=True)
    ax[0].hist(false_neg, bins=10, alpha=0.6,label="false neg", log=True)
    ax[0].set_xlabel("Non-Exit")
    ax[0].set_ylabel("Log Frequency")
    ax[0].legend()

    ax[1].hist(true_pos, bins=10, alpha=0.6, label="true pos", log=True)
    ax[1].hist(false_pos, bins=10, alpha=0.6, label="false pos", log=True)
    ax[1].set_xlabel("Exit")
    ax[1].legend()

    fig.suptitle(f'Model Predicted Probs (threshold={threshold}, speed_gain={avg_speed_gain:.2f})')
    plt.tight_layout()
    plt.savefig(f'{json_file}_probs_threshold{threshold}.png')


def optimal(json_file='pondering_stats.json', A=1.365, B=23.618):
    with open(json_file) as fh:
        j = json.load(fh)
    exit_range, max_accept_length = parse_data_lengths(j)
    C = [A * (i+1) + B for i in exit_range]

    data = []
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7,   0.8, 0.90, 0.95, 0.99, 1.0]
    for threshold in thresholds:
        lengths, *_ = calc_stats(j, exit_range, threshold)
        speed_gain = calc_speed_gain(lengths, max_accept_length, C)
        data.append(speed_gain)

    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    for i, speed_gain in enumerate(data):
        threshold = thresholds[i]
        avg_speed_gain = sum(speed_gain) / (len(speed_gain) + 1e-5)
        ax[i // 5, i % 5].hist(speed_gain, bins=max_accept_length)
        ax[i // 5, i % 5].set_title(f"threshold={threshold:.2f}")
        ax[i // 5, i % 5].set_xlabel(f"Speed Gain (avg={avg_speed_gain:.2f})")

    plt.tight_layout()
    plt.savefig(f'{json_file}_optimal_A{A:.2f}_B{B:.2f}.png')


def costs(json_file='pondering_stats.json'):
    with open(json_file) as fh:
        j = json.load(fh)
    max_exit_i = max(j['exit@'])
    print('max_exit_i', max_exit_i)

    C_hist = defaultdict(list)
    for iter_num, exit_at in enumerate(j['exit@']):
        C_hist[exit_at].append(j['C'][iter_num])

    print('keys', C_hist.keys())
    C_hist = [np.array(C_hist[i]) for i in range(-1, max_exit_i + 1)]
    C_mean = [h.mean().item() for h in C_hist]
    print('mean', C_mean)
    C_std = [h.std().item() for h in C_hist]
    print('std', C_std)

    X = np.arange(len(C_mean))
    Y = np.array(C_mean)
    A, B = np.polyfit(X, Y, 1) # degree 1 → linear iterpolation
    interpolated = [(A * x + B).item() for x in range(len(C_mean))]
    print('interpolated', interpolated)
    print('A,B', round(A, 2), round(B, 2))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(X, Y, color='red')
    ax.plot(X, interpolated)
    ax.set_ylabel("Cost")
    ax.set_xlabel("Exit Length")
    fig.suptitle(f'Model Cost Interpolation (A={A:.2f}, B={B:.2f})')
    plt.tight_layout()
    plt.savefig(f'{json_file}_costs.png')


if __name__ == '__main__':
    fire.Fire(dict(probs=probs, optimal=optimal, costs=costs))
