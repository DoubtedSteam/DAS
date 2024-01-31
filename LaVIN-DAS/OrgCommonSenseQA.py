import datasets
import random

commonsense_qa = datasets.load_dataset('commonsense_qa')

# trainset = []

# for example in commonsense_qa['train']:
#     question = 'Question: ' + example['question'] + '\n'
    
#     question += 'Options: '
#     for i in range(len(example['choices']['label'])):
#         question += example['choices']['label'][i] + '. ' + example['choices']['text'][i] + '\n'

#     try:
#         answer = 'Answer: The answer is {}. {}.\n\n'.format(example['answerKey'], example['choices']['text'][ord(example['answerKey']) - ord('A')])
#     except:
#         print(example)
#         answer = 'Answer: The answer is {}. {}.\n\n'.format('C', example['choices']['text'][ord('C') - ord('A')])

#     trainset.append(question + answer)

# count = len(trainset)

data_list = []

for example in commonsense_qa['validation']:
    question = "The following are multiple choice questions (with answers) about {}.\n\n".format(example['question_concept'])
    
    # for i in range(5):
    #     question += trainset[random.randint(0, len(trainset)-1)]
    
    question += 'Question: ' + example['question'] + '\n'

    question += 'Options: '
    for i in range(len(example['choices']['label'])):
        question += example['choices']['label'][i] + '. ' + example['choices']['text'][i] + '\n'
        
    data_list.append(
        {
            "subject": example['question_concept'],
            "input": question,
            "output": example['answerKey']
        }
    )

import json

# 指定输出的 JSON 文件路径
output_file_path = "commonsense_qa_0_shot_test.json"

# 将每个对象逐行写入文件
with open(output_file_path, 'w') as file:
    for data in data_list:
        json.dump(data, file)
        file.write('\n')  # 添加换行符，使得每个 JSON 对象占据文件的一行
    