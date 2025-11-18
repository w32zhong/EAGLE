import json
import fire
import numpy as np


def main(*jsonl_files):
    for jsonl_file in jsonl_files:
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                data.append(json_obj)

        speeds=[]
        for datapoint in data:
            qid=datapoint["question_id"]
            answer=datapoint["choices"][0]['turns']
            tokens= sum(datapoint["choices"][0]['new_tokens'])
            times = sum(datapoint["choices"][0]['wall_time'])
            speeds.append(tokens/times)

        avg_speed = np.array(speeds).mean()
        print(len(speeds), jsonl_file, avg_speed)


if __name__ == '__main__':
    fire.Fire(main)
