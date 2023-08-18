# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
import os

def postprocess(response):
    bad_strings = ['Human:', 'Assistant:', 'User:', 'human:', 'assistant:', 'user:']
    for s in bad_strings:
        if response.strip().startswith(s):
            response = response.strip()[len(s):].strip()
    for s in bad_strings:
        if s in response:
            response = response[:response.index(s)].strip()
    return response.strip()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str)
    parser.add_argument('--out-file', type=str)
    args = parser.parse_args()

    with open(args.in_file, 'r') as rf:
        data = json.load(rf)

    print(args.in_file)
    print(args.out_file)
    
    with open(args.out_file, 'w') as wf:
        for line in data:
            line['output'] = postprocess(line['output'] if type(line['output']) == str else line['output'][0])
        wf.write(json.dumps(data) + '\n')
    
