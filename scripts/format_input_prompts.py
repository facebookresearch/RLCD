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

    prompts = []
    with open(args.in_file, 'r') as f:
        # load jsonl
        for line in f:
            data = json.loads(line)
            prompt = data['chosen']
            prompts.append(prompt[:prompt.rindex('Assistant:') + len('Assistant:')].strip())

    with open(args.out_file, 'w') as wf:
        writer = csv.DictWriter(wf, fieldnames=['instruction', 'input', 'output'])
        writer.writeheader()
        for i in range(len(prompts)):
            if i >= args.start_idx and i < args.end_idx:
                prompt = prompts[i]
                writer.writerow({'instruction': prompt, 'input': '', 'output': ''})