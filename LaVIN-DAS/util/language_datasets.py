import datasets
import torch.utils.data as Data

import torch
import copy
import random
import re

from datasets import load_dataset, load_from_disk
from das import Tokenizer


ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n"
        "Instruction: {instruction}\nInput: {input}\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n"
        "Instruction: {instruction}\n"
    ),
}


def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"].format(**example)
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"].format(**example)
        
    response = "Response: {}".format(example['output'])
        
    return {'input': prompt_format, 'output': response}

def extract_csqa_dataset(example):
    question = "The following are multiple choice questions (with answers) about {}.\n\n".format(example['question_concept'])
    question += 'Question: ' + example['question'] + '\n'

    question += 'Options: '
    for i in range(len(example['choices']['label'])):
        question += example['choices']['label'][i] + '. ' + example['choices']['text'][i] + '\n'

    try:
        answer = 'Answer: The answer is {}. {}.'.format(example['answerKey'], example['choices']['text'][ord(example['answerKey']) - ord('A')])
    except:
        print(example)
        answer = 'Answer: The answer is {}. {}.'.format('C', example['choices']['text'][ord('C') - ord('A')])

    return {'input': question, 'output': answer}


def extract_boolq_detaset(example):
    question = "The following are yes/no questions (with answers) based on the given passage.\n\n"
    question += 'Passage: ' + example['passage'] + '\n\n'
    question += 'Question: ' + example['question'] + '\n\n'

    question += 'Options: A. True\nB. False\n\n'

    answer = 'Answer: The answer is {}.'.format("A. True" if example['answer'] else "B. False")
    
    return {'input': question, 'output': answer}


def extract_gsm8k_dataset(example):
    question = "The following are linguistically diverse grade school math word problems (with answers).\n\n"
    question += 'Question: ' + example['question'] + '\n\n'
    
    tmp = example['answer'].split('#### ')
    
    reason = tmp[0]
    answer = tmp[1]
    
    pattern = r'<<.*?>>'
    reason = re.sub(pattern, '', reason)
    
    # answer = 'Because: {}\nAnswer: The answer is {}.'.format(reason, answer)
    answer = 'Answer: The answer is {}.\n\nBecause: {}'.format(answer, reason)
    
    return {'input': question, 'output': answer}


def format_dataset(dataset, dataset_format):
    if 'alpaca' in dataset_format:
        dataset = dataset.map(extract_alpaca_dataset)
    elif 'commonsense_qa' in dataset_format:
        dataset = dataset.map(extract_csqa_dataset)
    elif 'boolq' in dataset_format:
        dataset = dataset.map(extract_boolq_detaset, load_from_cache_file=False)
    elif 'gsm8k' in dataset_format:
        dataset = dataset.map(extract_gsm8k_dataset, load_from_cache_file=False)
    
    # if (
    #     dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
    #     (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
    # ):
    #     dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
    # elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
    #     dataset = dataset.map(lambda x: {
    #         'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
    #         'output': x['text'].split('\n<bot>: ')[1],
    #     })
    # elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
    #     for old, new in [["prompt", "input"], ["completion", "output"]]:
    #         dataset = dataset.rename_column(old, new)
    # elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
    #     dataset = dataset.map(lambda x: {
    #         'input': '',
    #         'output': x['chosen']
    #     })
    # elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
    #     dataset = dataset.map(lambda x: {
    #         'input': '',
    #         'output': x['text'],
    #     })
    # elif dataset_format == 'input-output':
    #     # leave as is
    #     pass
    
    # Remove unused columns.
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
    )
    
    return dataset


def load_data(dataset_name, local_dataset):
    if local_dataset:
        return load_from_disk(dataset_name)
    
    if dataset_name == 'alpaca':
        return load_dataset("tatsu-lab/alpaca")
    elif dataset_name == 'commonsense_qa':
        return load_dataset("commonsense_qa")
    elif dataset_name == 'boolq':
        return load_dataset("boolq")
    elif dataset_name == 'gsm8k':
        return load_dataset("gsm8k", "main")
    # elif dataset_name == 'alpaca-clean':
    #     return load_dataset("yahma/alpaca-cleaned")
    # elif dataset_name == 'chip2':
    #     return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
    # elif dataset_name == 'self-instruct':
    #     return load_dataset("yizhongw/self_instruct", name='self_instruct')
    # elif dataset_name == 'hh-rlhf':
    #     return load_dataset("Anthropic/hh-rlhf")
    # elif dataset_name == 'longform':
    #     return load_dataset("akoksal/LongForm")
    # elif dataset_name == 'oasst1':
    #     return load_dataset("timdettmers/openassistant-guanaco")
    # elif dataset_name == 'vicuna':
    #     raise NotImplementedError("Vicuna data was not released.")
    else:
        raise NotImplementedError("To be continue")
        # if os.path.exists(dataset_name):
        #     try:
        #         args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
        #         full_dataset = local_dataset(dataset_name)
        #         return full_dataset
        #     except:
        #         raise ValueError(f"Error loading dataset from {dataset_name}")
        # else:
        #     raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")
        
        
class LanguageDataSet(Data.Dataset):
    def __init__(self, args, split, model_path, max_words=512, max_image_feats=0):
        super(LanguageDataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        dataset_train = load_data(args.language_dataset, args.local_dataset)
        dataset_train = format_dataset(dataset_train, args.language_dataset)
        self.data = dataset_train[split]
        
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.max_words = max_words
        self.max_image_feats = max_image_feats
        self.split = split

        print(f"number of problems in split {split}: {len(self.data['input'])}\n")

    def tokenize(self, prompt, answer):
        example = prompt + answer
        prompt = torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        return example, labels, example_mask, label_mask


    def __getitem__(self, idx):
        prompt_question = self.data['input'][idx]
        prompt_answer = self.data['output'][idx]
        
        example, labels, example_mask, label_mask = self.tokenize(prompt_question, prompt_answer)

        return example, labels, example_mask, 0, 0


    def __len__(self):
        return len(self.data['input'])


    def shuffle_list(self, list):
        random.shuffle(list)