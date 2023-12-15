
import argparse
import os
import random
from collections import defaultdict
from tqdm import tqdm

import cv2
import re

import numpy as np
from PIL import Image
import torch
import html
import json

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config

from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

ans_options = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D'
}


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cudnn.benchmark = False
cudnn.deterministic = True

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

device = 'cuda:{}'.format(args.gpu_id)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
#bounding_box_size = 100

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

model = model.eval()

chat = Chat(model, vis_processor, device=device)

CONV_VISION = Conversation(
            system="",
            roles=(r"<s>[INST] ", r" [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

temperature = 0.6

print('Chat initialized')

def get_description(image_folder, mini_gpt_questions, num_images=48):
    # first for the initial prompts
    prompt_descriptions = []
    for i in tqdm(range(num_images)):
        image_answers = []

        image_url = image_folder +str(i)+'.jpeg'
        
        img_chat_state = CONV_VISION.copy()
        img_list = []
        chat.upload_img(image_url, img_chat_state, img_list)
        chat.encode_img(img_list)
        print('Image uploaded')

        for user_message in mini_gpt_questions:
            # reset chat for question
            chat_state = img_chat_state.copy()
            
            # ask the question
            chat.ask(user_message, chat_state)
            output_text, _ = chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    temperature=temperature,
                                    max_new_tokens=100,
                                    max_length=500)

            image_answers.append(output_text)

        prompt_descriptions.append(image_answers)
        
    return prompt_descriptions

def build_mini_gpt_questions(questions):
    mini_gpt_questions = ['[vqa] Describe this image.']
    for q in questions:
        mini_gpt_questions.append('[vqa] '+q)
    return mini_gpt_questions

def convert_bb(boxes):
    width = boxes['width']
    height = boxes['height']
    box_list = boxes['boxes']

    new_boxes, new_box_dicts = [], []
    for box in box_list:
        left = str(round(box[0]/width*100))
        top = str(round(box[1]/height*100))
        right = str(round(box[2]/width*100))
        bottom = str(round(box[3]/height*100))
        d = {"x1": box[0], "x2": box[2], "y1":box[1], "y2":box[3]}
        new_box_dicts.append(d)
        new_box = "{<"+ left +"><"+top+"><"+right+"><"+bottom+">}"
        new_boxes.append(new_box)
    return new_boxes, new_box_dicts


def replace_q_bboxes(q, boxes):
    box_list = []
    for i, part in enumerate(q):
        if isinstance(part, list):
            replacement_str = ''
            for idx in part:
                replacement_str += boxes[idx]+' '
                box_list.append(boxes[idx])
            q[i] = replacement_str.strip()

    return ' '.join(q), box_list



folder = "/scratch/ssd004/datasets/vcr/vcr1images/"
annots = "/scratch/ssd004/scratch/sahiravi/cache/vcr1/"
# Load the json file
import json
with open(annots+'val.jsonl') as f:
    data = [json.loads(line) for line in f]

print(data[0])

final_list = []
accuracy = 0
completed = 0
for annot in tqdm(data[:20]):
    ans_dict = {}
    ans_dict['gt_question'] =  " ".join([str(s) for s in annot['question']])

    image_url = folder+annot['img_fn']
    image_boxes_url = folder+annot['metadata_fn']

    print(image_url)
    print(image_boxes_url)
    with open(image_boxes_url) as f:
        boxes = json.load(f)
    
    # print(image_url)
    # print(' '.join([str(x) for x in annot['question']]))
    # print([' '.join([str(x) for x in answer]) for answer in annot['answer_choices']])
    # print(annot['objects'])
    # print(boxes['width'], boxes['height'])
    # print(boxes['boxes'])
    new_boxes, dict_boxes = convert_bb(boxes)
    question, box_list = replace_q_bboxes(annot['question'], new_boxes)

    options = []
    for answer in annot['answer_choices']:
        options.append(replace_q_bboxes(answer, new_boxes)[0])
    
    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_img(image_url, chat_state, img_list)
    chat.encode_img(img_list)
    # print('Image uploaded')

    # # ask the question
    for box_str in box_list:
        chat_message = "[identify] what is this "+box_str
        chat.ask(chat_message, chat_state)
        _, _ = chat.answer(conv=chat_state,
                                img_list=img_list,
                                temperature=temperature,
                                max_new_tokens=100,
                                max_length=500)
    

    question_with_options = f"""
Read the question and provided answer options carefully and choose the best among (A, B, C or D): 
Question: {question}
Options:
"""
    for iz, option in enumerate(options):
        question_with_options += f"""{ans_options[iz]}) {option}
"""
    question_with_options += """Answer: """
    print(question_with_options)
    chat.ask(question_with_options, chat_state)
    output_text, _ = chat.answer(conv=chat_state,
                            img_list=img_list,
                            temperature=temperature,
                            max_new_tokens=2,
                            max_length=2000)

    output_text = output_text.strip()


    if output_text == 'A':
        predicted_label = 0
    elif output_text == 'B':
        predicted_label = 1
    elif output_text == 'C':
        predicted_label = 2
    elif output_text == 'D':
        predicted_label = 3

    if predicted_label == annot['answer_label']:
        accuracy += 1

    completed += 1
    ans_dict['img_id'] = annot['img_id']
    ans_dict['annot_id'] = annot['annot_id']
    ans_dict['image_url'] = image_url
    ans_dict['new_boxes'] = dict_boxes
    ans_dict['annotations'] = new_boxes
    ans_dict['objects'] = annot['objects']
    ans_dict['new_question'] = question
    ans_dict['predicted_label'] = predicted_label
    ans_dict['GT_answer'] = annot['answer_orig']
    ans_dict['answer_label'] = annot['answer_label']
    ans_dict['chat_state'] = str(chat_state)
    # ans_dict['original_boxes'] = str(boxes)
    final_list.append(ans_dict)



final_acc = 0
for entry in final_list:
    if entry['predicted_label'] == entry['answer_label']:
        final_acc += 1

print('Final Accuracy: ', final_acc/len(final_list))

with open('val_new.json', 'w') as f:
    json.dump(final_list, f, indent=4)


    