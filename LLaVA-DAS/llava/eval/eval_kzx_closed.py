import argparse
import json
import collections
import random
import pandas as pd    
from nltk.translate.bleu_score import sentence_bleu
from eval_utils.evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization
from tabulate import tabulate
from eval_utils.glossary import *

import warnings
warnings.simplefilter('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--gt', type=str, default="test.json", help='path to groundtruth file', )
    parser.add_argument('--candidate', type=str, default="candidate.json", help='path to candidate answer file', )
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file', )
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

def evaluate(gt, pred, candidate, criterion=None):    
    closed_scores = collections.defaultdict(list)

    for gt_item, pred_item in zip(gt, pred):
        try:
            gt_results = gt_item['conversations']
        except:
            gt_results = gt_item['conversatons']
        gt_value = gt_results[1]['value'].lower()
        pred_value = pred_item['text'].lower()

        gt_value = normalize_word(gt_value)
        pred_value = normalize_word(pred_value)

        closed_scores['q_id'].append(pred_item['question_id'])
        # # 中文
        # if gt_value == pred_value:
        #     hit = 1.0
        # else:
        #     hit = 0.0
        # closed_scores['hit'].append(hit)
        # 中文
        if gt_value == pred_value:
            closed_scores['hit'].append(1)
        else:
            closed_scores['hit'].append(0)
    
    closed_score = sum(closed_scores['hit']) / len(closed_scores['hit']) if len(closed_scores['hit']) != 0 else 0.0

    num_open, num_close = 0,len(closed_scores['hit'])
    return tabulate(
        [
            ['yes/no accuracy', closed_score*100]
        ], 
        headers=['Metric', 'Performance']
    )

if __name__ == '__main__':
    args = parse_option()

    dataset = args.gt.split("/")[-2]
    print(f"\n========\n {dataset}")

    gt = json.load(open(args.gt, 'r'))
    # candidate = json.load(open(args.candidate, 'r'))
    candidate = None
    pred = load_jsonl(args.pred)

    gt_ids = [item['id'] for item in gt]
    pred_ids = [item['question_id'] for item in pred]
    num_gt_ids, num_pred_ids = len(gt_ids), len(pred_ids)
    print(f'num_gt_ids: {num_gt_ids} || num_pred_ids: {num_pred_ids}')
    # import pdb; pdb.set_trace()
    assert gt_ids == pred_ids, "please make sure pred and gt are exactly matched"

    # perform evaluation
    results = evaluate(gt, pred, candidate)
    print(results)