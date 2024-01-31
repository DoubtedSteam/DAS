import datasets
import random

boolq = datasets.load_dataset('boolq')

# trainset = []

# for example in boolq['train']:
#     question = 'Question: ' + example['question'] + '\n'
#     question += 'Passage: ' + example['passage'] + '\n'

#     question += 'Options: A. True\nB. False\n'

#     question += 'Answer: The answer is {}\n\n'.format("A. True" if example['answer'] else "B. False")

#     trainset.append(question)

# count = len(trainset)

data_list = []

for example in boolq['validation']:
    question = "The following are yes/no questions (with answers) based on the given passage\n\n"
    # for i in range(5):
    #     question += trainset[random.randint(0, len(trainset)-1)]

    question += 'Passage: ' + example['passage'] + '\n\n'
    question += 'Question: ' + example['question'] + '\n\n'

    question += 'Options: A. True\nB. False\n\n'

    data_list.append(
        {
            "subject": '',
            "input": question,
            "output": "A" if example['answer'] else "B"
        }
    )

import json

# 指定输出的 JSON 文件路径
output_file_path = "boolq_0_shot_test.json"

# 将每个对象逐行写入文件
with open(output_file_path, 'w') as file:
    for data in data_list:
        json.dump(data, file)
        file.write('\n')  # 添加换行符，使得每个 JSON 对象占据文件的一行
    