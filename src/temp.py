#解析json文件
import json

json_path = '/home/hewaele/Desktop/coco数据集/annotations_trainval2014/annotations/instances_val2014.json'
fb = open(json_path)
data = json.load(fb)
print(data.keys())

print(len(data['images']))
print(data['images'][0])

print(len(data['annotations']))
print(data['annotations'][0].items())

print(data['categories'])