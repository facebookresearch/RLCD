# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import csv
import json

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str)
    parser.add_argument('--out-file', type=str)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=10000000)
    args = parser.parse_args()

    with open(args.in_file, 'r') as f:
        data = list(csv.DictReader(f))
            
    with open(args.out_file, 'w') as wf:
        writer = csv.DictWriter(wf, fieldnames=['instruction', 'input', 'output'])
        writer.writeheader()
        for i, d in enumerate(data):
            if i >= args.start_idx and i < args.end_idx:
                assert d['preference'] in ['1', '2']
                chosen = d['output_1'] if d['preference'] == '1' else d['output_2']
                instruction = d['instruction'].strip()
                output = chosen.strip()
                writer.writerow({'instruction': instruction, 'input': '', 'output': output})