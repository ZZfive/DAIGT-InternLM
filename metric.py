
import re
import json

import numpy as np
from lmdeploy import turbomind as tm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# load model
model_path = "/root/ft-oasst1/merged"  # 本地的hf格式的模型路径
tm_model = tm.TurboMind.from_pretrained(model_path, model_name='internlm-chat-7b')
generator = tm_model.create_instance()

# prompt = tm_model.model.get_prompt(query)
# input_ids = tm_model.tokenizer.encode(query)


def infer(input_ids):
    for outputs in generator.stream_infer(
        session_id=0,
        input_ids=[input_ids]):
        res, tokens = outputs[0]

    response = tm_model.tokenizer.decode(res.tolist())
    return response


def eval(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    f.close()
    num = 0
    data = data[:100]
    for v in data:
        text = v['conversation'][0]['input']
        label = v['conversation'][0]['output'][-1]
        input_ids = tm_model.tokenizer.encode(text)
        response = infer(input_ids)
        if response and response[-1] == label:
            num += 1
    return round((num / len(data)) * 100, 2)


def find_last_number(input_string):
    # 定义匹配数字（包括整数和小数）的正则表达式
    pattern = r"[-+]?\d*\.\d+|\d+"
    
    # 使用正则表达式查找所有匹配项
    matches = re.findall(pattern, input_string)
    
    # 如果没有匹配到数字，返回 None
    if not matches:
        return None
    
    # 返回最后一个匹配到的数字
    return float(matches[-1])


def predict(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    f.close()
    predicts = []
    labels = []
    # data = data[:50]
    for v in data:
        text = v['conversation'][0]['input']
        label = v['conversation'][0]['output'][-1]
        input_ids = tm_model.tokenizer.encode(text)
        response = infer(input_ids)
        predict = find_last_number(response)
        labels.append(int(label))
        predicts.append(predict)
    return predicts, labels


def metric(predicts, labels):
    # 计算准确率
    accuracy = accuracy_score(labels, predicts)

    # 计算精确率
    precision = precision_score(labels, predicts)

    # 计算召回率
    recall = recall_score(labels, predicts)

    # 计算F1分数
    f1 = f1_score(labels, predicts)

    return accuracy, precision, recall, f1


if __name__=='__main__':
    # # 接口输出测试
    # answers = []
    # queries = ["你是谁", "who are you"]
    # for q in queries:
    #     input_ids = tm_model.tokenizer.encode(q)
    #     answers.append(infer(input_ids))
    # print(answers)

    # 简单DAIGT性能测试
    data_path = "/root/ft-oasst1/conversations_v1_140_test.json"
    # acc = eval(data_path)
    # print(acc)

    # 全面性能检测
    predicts, labels = predict(data_path)
    # 将预测的结果保存
    with open('./conversations_v1_140_test_predicts_small.json', 'w', encoding='utf-8') as f:
        json.dump(predicts, f, ensure_ascii=False, indent=4)
        f.close()
    real_predicts = []
    real_labels = []
    for i in range(len(predicts)):
        if predicts[i] is not None:
            real_predicts.append(predicts[i])
            real_labels.append(labels[i])
    accuracy, precision, recall, f1 = metric(real_predicts, real_labels)
    print(accuracy, precision, recall, f1)