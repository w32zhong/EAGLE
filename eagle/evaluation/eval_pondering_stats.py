import json
import fire
import numpy as np
import matplotlib.pyplot as plt


def main(json_file='pondering_stats.json'):
    with open(json_file) as fh:
        j = json.load(fh)
    print(j.keys())
    prev_survive_probs, last_survive_probs = [], []
    for iter_num, accept_length in enumerate(j['a']):
        survive_to_the_next = 1.0
        for i in range(accept_length):
            e_i = j[f'e{i}'][iter_num]
            survive_to_the_next *= (1 - e_i)
            prev_survive_probs.append(survive_to_the_next)
            print(round(survive_to_the_next, 2), end=' ')
        if accept_length > 0:
            prev_survive_probs.pop()
            last_survive_probs.append(survive_to_the_next)
            print()

    fig, ax = plt.subplots(1, 2)
    ax[0].hist(prev_survive_probs, bins=10)
    ax[0].set_xlabel("Predicted survival prob when accepted")
    ax[0].set_ylabel("Frequency")

    ax[1].hist(last_survive_probs, bins=10)
    ax[1].set_xlabel("Predicted survival prob when rejected")

    plt.tight_layout()
    plt.savefig('./eval_pondering_stats.png')


if __name__ == '__main__':
    fire.Fire(main)
