import datasets
import random

gsm8k = datasets.load_dataset('gsm8k', 'main')

trainset = []

# for example in gsm8k['train']:
#     question = 'Question: ' + example['question'] + '\n'
    
#     tmp = example['answer'].split('#### ')
    
#     reason = tmp[0]
#     answer = tmp[1]
    
#     answer = 'Answer: The answer is {}.\nBecause: {}\n'.format(answer, reason)

#     trainset.append(question + answer)

# count = len(trainset)

data_list = []

for example in gsm8k['test']:
    question = "The following are linguistically diverse grade school math word problems (with answers).\n\n"
    # for i in range(5):
    #     question += trainset[random.randint(0, len(trainset)-1)]
        
    question += 'Question: ' + example['question'] + '\n'

    answer = example['answer'].split('#### ')[1].replace(',', '')

    data_list.append(
        {
            "subject": '',
            "input": question,
            "output": answer
        }
    )

import json

# 指定输出的 JSON 文件路径
output_file_path = "gsm8k_0_shot_test.json"

# 将每个对象逐行写入文件
with open(output_file_path, 'w') as file:
    for data in data_list:
        json.dump(data, file)
        file.write('\n')  # 添加换行符，使得每个 JSON 对象占据文件的一行
    