import argparse
import os
import json
import random
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


HARMLESS_ANTHROPIC_INSTRUCTIONS = [
    'Please choose the response that is the most helpful, honest, and harmless.',
    'Please choose the assistant response that is as harmless and ethical as possible. Do NOT choose responses that are toxic, racist, or sexist, or that encourage or support illegal, violent, or unethical behavior. Above all the assistant’s response should be wise, peaceful, and ethical.',
    'Compare the degree of harmfulness in the assistant responses and choose the one that’s less harmful. However, try to avoid choosing responses that are too preachy, obnoxious, or overly-reactive.',
    'Choose the assistant response that is as harmless, helpful, polite, respectful, and thoughtful as possible without sounding overly-reactive of accusatory.',
    'Please choose the assistant response that’s more ethical and moral. Do NOT choose responses that exhibit toxicity, racism, sexism or any other form of physical or social harm.',
    'Which of these assistant responses exhibits less harmful and more acceptable behavior? Choose the less harmful response.',
    'Choose the assistant response that answers the human’s query in a more friendly, amiable, conscientious, and socially acceptable manner.',
    'Which of these assistant responses is less harmful? Choose the response that a wise, ethical, polite and friendly person would more likely say.',
    'Choose the response that is less harmful, paying close attention to whether each response encourages illegal, unethical or immoral activity.',
    'Please consider these assistant responses and choose the one that is less harmful, more age-appropriate, and more family-friendly.',
    'Choose the response that answers the human in the most thoughtful, respectful and cordial manner.',
    'Compare the assistant responses in terms of toxicity, racism, sexism and social bias, and choose the less harmful response.',
    'Choose the response that sounds most similar to what a peaceful, ethical, and wise person like Martin Luther King Jr. or Mahatma Gandhi might say.',
    'Which of the assistant responses is most harmless, helpful, and honest? Choose the best response based on these criteria.',
    'Choose the assistant response that is more harmless, polite, pleasant, and least likely to offend a socially-aware audience.',
    'Choose the assistant response that demonstrates more ethical and moral awareness without sounding excessively condescending, reactive, annoying or condemnatory.',
]


HARMLESS_POSITIVE_AFFIXES = ['(helpful, honest, inoffensive response)',
                            '(law-abiding, ethical response)',
                            '(innocent response)',
                            '(harmless, helpful response)',
                            '(ethical and moral response)',
                            '(inoffensive response)',
                            '(correct, conscientious, socially acceptable response)',
                            '(simulating a wise, ethical person)',
                            '(legal, ethical, moral response)',
                            '(age-appropriate, family-friendly response)',
                            '(thoughtful response)',
                            '(politically correct response)',
                            '(simulating a very kind person)',
                            '(harmless, helpful, honest response)',
                            '(ethical response)',
                            '(ethically and morally aware response)',]
    

HARMLESS_NEGATIVE_AFFIXES = ['(unhelpful, dishonest, toxic response)',
                            '(illegal, unethical response)',
                            '(prejudiced response)',
                            '(harmful, unhelpful response)',
                            '(unethical and immoral response)',
                            '(offensive response)',
                            '(incorrect, unconscientious, socially unacceptable response)',
                            '(simulating a foolish, unethical person)',
                            '(illegal, unethical, immoral response)',
                            '(inappropriate, not-family-friendly response)',
                            '(thoughtless response)',
                            '(racist, sexist, socially biased response)',
                            '(simulating a really evil person)',
                            '(harmful, unhelpful, dishonest response)',
                            '(toxic response)',
                            '(unethical and immoral response)']


HELPFUL_ANTHROPIC_INSTRUCTIONS = ['Please choose the response that is more helpful.']


HELPFUL_POSITIVE_AFFIXES = ['(giving a helpful response)']


HELPFUL_NEGATIVE_AFFIXES = ['(giving an unhelpful response)']


def _parse_indices_helper(indices):
    for index in indices.split(','):
        if '-' in index:
            start, end = index.split('-')
            for i in range(int(start), int(end) + 1):
                yield i
        else:
            yield int(index)


def parse_indices(indices):
    return list(_parse_indices_helper(indices))


def process_filter_result(result):
    # some simple filtering + processing used for all methods (RLCD and baselines); feel free to improve based on your own use case
    if 'Human' not in result:
        return False, None
    result = result[:result.index('Human')]
    if '\n' not in result and '</s>' not in result:
        return False, None
    if '\n' in result:
        result = result[:result.rindex('\n')].strip()
    if '</s>' in result:
        result = result[:result.index('</s>')].strip()
    if 'Assistant:' in result: 
        # sometimes the response starts with an extra Assistant: delimiter or has one very early on
        # sometimes this will cut off the beginning of the response if there are lots of extra Assistant: tokens
        result = result[result.index('Assistant:') + len('Assistant:'):].strip()
    if len(result) == 0:
        return False, None
    return True, result


def generate_result(model, tokenizer, prompts, max_new_tokens=300):
    if type(prompts) == str:
        prompts = [prompts]
    bad_words = ['\\', '\\\\', '`', '```']
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids
    all_results = [None for _ in range(len(prompts))]
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')
        del inputs['token_type_ids']
        results = model.generate(**inputs, temperature=1, do_sample=True, max_new_tokens=max_new_tokens, bad_words_ids=bad_words_ids)
        for i in range(len(prompts)):
            result = results[i][len(inputs['input_ids'][i]):]
            result = tokenizer.decode(result)
            status, result = process_filter_result(result)
            all_results[i] = (status, result)
    return all_results


def create_prompts(args, prompt):
    if args.task == 'harmless':
        positive_affix_choices = HARMLESS_POSITIVE_AFFIXES
        negative_affix_choices = HARMLESS_NEGATIVE_AFFIXES
    elif args.task == 'helpful':
        positive_affix_choices = HELPFUL_POSITIVE_AFFIXES
        negative_affix_choices = HELPFUL_NEGATIVE_AFFIXES
    else:
        raise NotImplementedError
    assert len(positive_affix_choices) == len(negative_affix_choices)
    index = random.choice(list(range(len(positive_affix_choices))))
    positive_affix = positive_affix_choices[index]
    negative_affix = negative_affix_choices[index]
    # use the same-polarity affix at the end, but opposite-polarity affix for previous instances of Assistant:
    positive_prompt = (prompt+'PROMPT_END').replace('\n\nAssistant:PROMPT_END', '\n\nAssistant {positive_affix}:'.format(positive_affix=positive_affix))
    positive_prompt = positive_prompt.replace('\n\nAssistant:', '\n\nAssistant {negative_affix}:'.format(negative_affix=negative_affix))
    negative_prompt = (prompt+'PROMPT_END').replace('\n\nAssistant:PROMPT_END', '\n\nAssistant {negative_affix}:'.format(negative_affix=negative_affix))
    negative_prompt = negative_prompt.replace('\n\nAssistant:', '\n\nAssistant {positive_affix}:'.format(positive_affix=positive_affix))
    if args.method == 'rlcd':
        return ['chosen', 'rejected'], [positive_prompt, negative_prompt], index
    elif args.method == 'rlcd_rescore':
        prompts = [positive_prompt, negative_prompt]
        random.shuffle(prompts)
        return ['resultA', 'resultB'], prompts, index
    elif args.method == 'rlaif':
        return ['resultA', 'resultB'], [prompt, prompt], None


def ready_for_analysis(args, analysis_waiting_entry):
    # check that we generated both outputs for the conversation prefix before moving on to post hoc scoring as needed
    if args.method == 'rlcd':
        return all([key in analysis_waiting_entry for key in ['chosen', 'rejected']])
    elif args.method in ['rlaif', 'rlcd_rescore']:
        return all([key in analysis_waiting_entry for key in ['resultA', 'resultB']])


def analyze_results(args, batch, model, tokenizer):
    if args.method == 'rlcd':
        # no computation needed
        return [{
            'prompt': entry.data['prompt'],
            'actual_prompts': entry.data['actual_prompts'],
            'chosen_result': entry.data['chosen'],
            'rejected_result': entry.data['rejected'],
            'status': True
        } for entry in batch]
    elif args.method in ['rlaif', 'rlcd_rescore']:
        # post hoc scoring
        label_prompts = []
        if args.task == 'harmless':
            anthropic_instructions = HARMLESS_ANTHROPIC_INSTRUCTIONS
        elif args.task == 'helpful':
            anthropic_instructions = HELPFUL_ANTHROPIC_INSTRUCTIONS
        else:
            raise NotImplementedError
        for entry in batch:
            if entry.prompt_index is None:
                instruction = random.choice(anthropic_instructions)
            else:
                instruction = anthropic_instructions[entry.prompt_index]
            label_prompts.append('Consider the following conversation between a human and an assistant:\n\n\n\n' + entry.data['prompt'].strip() + '\n\n\n\n' + instruction.strip() + '\n\n\n\nOptions:\n\n\n\n(A) ' + entry.data['resultA'].strip() + '\n\n\n\n(B) ' + entry.data['resultB'].strip() + '\n\n\n\nThe answer is: (')
        label_prompt_inputs = tokenizer(label_prompts, return_tensors='pt', padding=True).to('cuda')
        del label_prompt_inputs['token_type_ids']
        logits = model(**label_prompt_inputs).logits[:, -1]
        A_token = tokenizer.encode('A')[-1]
        B_token = tokenizer.encode('B')[-1]
        A_logits = logits[:, A_token].cpu().tolist()
        B_logits = logits[:, B_token].cpu().tolist()
        data_dicts = []
        for i, entry in enumerate(batch):
            data_dicts.append({
                'prompt': entry.data['prompt'],
                'actual_prompts': entry.data['actual_prompts'],
                'label_prompt': label_prompts[i],
                'resultA': entry.data['resultA'],
                'resultB': entry.data['resultB'],
                'A_logit': A_logits[i],
                'B_logit': B_logits[i],
                'status': True,
            })
        return data_dicts


class GenerationQueueEntry:
    def __init__(self, idx, key, retries, prompt, original_prompt, prompt_index):
        self.idx = idx
        self.key = key
        self.retries = retries
        self.prompt = prompt
        self.original_prompt = original_prompt
        self.prompt_index = prompt_index

class AnalysisQueueEntry:
    def __init__(self, idx, data, prompt_index):
        self.idx = idx
        self.data = data
        self.prompt_index = prompt_index


def main(args):
    prompts = []
    with open(args.prompts_file, 'r') as f:
        # load jsonl
        for line in f:
            data = json.loads(line)
            prompt = data['chosen']
            prompts.append(prompt[:prompt.rindex('Assistant:') + len('Assistant:')].strip())

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_string, padding_side='left')
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error
    model = AutoModelForCausalLM.from_pretrained(args.model_string, device_map='auto', load_in_8bit=True, low_cpu_mem_usage=True).eval()

    indices = parse_indices(args.indices) if args.indices is not None else list(range(len(prompts)))

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            current_idx = -1
            failed_indices = set()
            generation_queue = []
            analysis_waiting = defaultdict(lambda: {})
            analysis_queue = []
            while current_idx < len(prompts) or len(generation_queue) > 0 or len(analysis_queue) > 0:

                # run a batch from the analysis queue (i.e., post hoc scoring for RLAIF or RLCD-Rescore) if it's full or we're at the end of data simulation
                if len(analysis_queue) >= args.batch_size or (len(analysis_queue) > 0 and current_idx >= len(prompts)):
                    batch = analysis_queue[:args.batch_size]
                    analysis_queue = analysis_queue[args.batch_size:]
                    assert all([x.idx not in failed_indices for x in batch])
                    save_dicts = analyze_results(args, batch, model, tokenizer)
                    for entry, save_dict in zip(batch, save_dicts):
                        with open(os.path.join(args.out_dir, f'{entry.idx}.json'), 'w') as f:
                            f.write(json.dumps(save_dict))
                
                # run a batch from the generation queue (i.e., data simulation) if it's full or we're at the end of data simulation
                elif len(generation_queue) >= args.batch_size or (len(generation_queue) > 0 and current_idx >= len(prompts)):
                    batch = generation_queue[:args.batch_size]
                    generation_queue = generation_queue[args.batch_size:]
                    batch = [x for x in batch if x.idx not in failed_indices]
                    batch_prompts = [x.prompt for x in batch]
                    batch_results = generate_result(model, tokenizer, batch_prompts)
                    for i, (status, result) in enumerate(batch_results):
                        if status:
                            analysis_waiting[batch[i].idx][batch[i].key] = result
                            analysis_waiting[batch[i].idx]['prompt'] = batch[i].original_prompt
                            if 'actual_prompts' not in analysis_waiting[batch[i].idx]:
                                analysis_waiting[batch[i].idx]['actual_prompts'] = []
                            analysis_waiting[batch[i].idx]['actual_prompts'].append(batch[i].prompt)
                            if ready_for_analysis(args, analysis_waiting[batch[i].idx]):
                                analysis_queue.append(AnalysisQueueEntry(batch[i].idx, analysis_waiting[batch[i].idx], batch[i].prompt_index))
                                del analysis_waiting[batch[i].idx]
                        else:
                            batch[i].retries += 1
                            if batch[i].retries >= args.max_retries:
                                failed_indices.add(batch[i].idx)
                                with open(os.path.join(args.out_dir, f'{batch[i].idx}.json'), 'w') as f:
                                    f.write(json.dumps({'status': False}))
                                if batch[i].idx in analysis_waiting:
                                    del analysis_waiting[batch[i].idx]
                            else:
                                generation_queue.append(batch[i])
                
                # add the next valid data index to the generation queue
                else:
                    current_idx += 1
                    if current_idx > len(prompts) or current_idx not in indices or current_idx % args.indices_mod != args.indices_remainder:
                        continue
                    if os.path.exists(os.path.join(args.out_dir, f'{current_idx}.json')):
                        continue
                    keys, new_prompts, prompt_index = create_prompts(args, prompts[current_idx])
                    for key, new_prompt in zip(keys, new_prompts):
                        generation_queue.append(GenerationQueueEntry(current_idx, key, 0, new_prompt, prompts[current_idx], prompt_index))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['harmless', 'helpful'])
    parser.add_argument('--method', type=str, choices=['rlcd', 'rlaif', 'rlcd_rescore'])
    parser.add_argument('--prompts-file', type=str)
    parser.add_argument('--model-string', type=str)
    parser.add_argument('--indices', type=str, default=None)
    parser.add_argument('--indices-remainder', type=int, default=0)
    parser.add_argument('--indices-mod', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-retries', type=int, default=5)
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    main(args)

    