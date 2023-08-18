# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
import os
import random
import csv
import sys

from scipy.special import softmax

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=str)
    parser.add_argument('--out-csv', type=str)
    parser.add_argument('--data-type', type=str, choices=['rlcd', 'rlaif', 'rlcd_rescore'])
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=10000000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.out_csv, 'w') as wf:
        if args.data_type in ['rlaif', 'rlcd_rescore']:
            fieldnames = ['instruction', 'input', 'output_1', 'output_2', 'preference', 'win2_prob']
        else:
            fieldnames = ['instruction', 'input', 'output_1', 'output_2', 'preference']
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()
        for fname in os.listdir(args.in_dir):
            if fname.endswith('.json'):
                if int(fname[:-len('.json')]) < args.start_idx or int(fname[:-len('.json')]) >= args.end_idx:
                    continue
                with open(os.path.join(args.in_dir, fname), 'r') as rf:
                    for line in rf:
                        datum = json.loads(line)
                        if not datum['status']:
                            continue
                        if args.data_type == 'rlcd':
                            chosen = datum['chosen_result'].strip()
                            rejected = datum['rejected_result'].strip()
                            if random.random() < 0.5:
                                writer.writerow({'instruction': datum['prompt'].strip(), 'input': '', 'output_1': chosen, 'output_2': rejected, 'preference': 1}) # input field is unused downstream in our pipeline
                            else:
                                writer.writerow({'instruction': datum['prompt'].strip(), 'input': '', 'output_1': rejected, 'output_2': chosen, 'preference': 2})
                        else:
                            assert args.data_type in ['rlaif', 'rlcd_rescore']
                            A_logit = float(datum['A_logit'])
                            B_logit = float(datum['B_logit'])
                            win2_prob = softmax([A_logit, B_logit])[1]
                            preference = 2 if win2_prob > 0.5 else 1
                            out1 = datum['resultA'].strip()
                            out2 = datum['resultB'].strip()
                            writer.writerow({'instruction': datum['prompt'].strip(), 'input': '', 'output_1': out1, 'output_2': out2, 'preference': preference, 'win2_prob': win2_prob})