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

from PIL import Image
import requests

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
import webbrowser
from datasets import load_dataset


def set_seeds():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cudnn.benchmark = False
    cudnn.deterministic = True


def attempt1(data, N=5):
    total = 0
    for annot in tqdm(data[:N]):
        sentence = annot["sentence"]
        urls = [annot["left_url"], annot["right_url"]]
        ims = []
        for url in urls:
            try:
                im = Image.open(requests.get(urls[i], stream=True).raw)
            except:
                # Image URL not loading.
                continue
        ground_truth = annot["label"]

        # print(annot["sentence"])
        print(annot["left_url"])
        print(annot["right_url"])

        mapping = {0: "left", 1: "right"}
        overall_chat_state = CONV_VISION.copy()
        all_images = []
        # # ask the question
        for i in range(2):
            chat_state = CONV_VISION.copy()
            # Each image
            img_list = []
            chat.upload_img(ims[i], chat_state, img_list)
            print("Image uploaded")
            chat.encode_img(img_list)
            all_images.extend(img_list)
            print(i, chat_state)

        overall_chat_state.messages.append(
            [
                "<s>[INST] ",
                "This is the left image: <Img><ImageHere></Img>. This is the right image: <Img><ImageHere></Img>",
            ]
        )
        print(overall_chat_state)
        question = (
            "[vqa] "
            + f""" Is the statement '{annot["sentence"]}' about the above images 'True' or 'False'?"""
        )
        print(question)
        chat.ask(question, chat_state)

        # Answer can accept more than one image, provided the # images matches image tags.
        output_text, _ = chat.answer(
            conv=overall_chat_state,
            img_list=all_images,
            temperature=temperature,
            max_new_tokens=100,
            max_length=1000,
        )
        print(output_text, annot["label"])
        total += output_text == annot["label"]

    print("Accuracy: ", total / N)


def run_inference(data, CONV_VISION, chat, temperature):
    final_list = []
    accuracy = 0
    completed = 0
    i = 0
    for annot in tqdm(data):
        i += 1
        if i >=10:
            break
        for pair in annot["question_answering_pairs"]:
            ans_dict = {}
            ans_dict['gt_question'] =  pair[0]
            chat_state = CONV_VISION.copy()
            img_list = []
            chat.upload_img(annot["image"].convert('RGB'), chat_state, img_list)
            chat.encode_img(img_list)
    
            chat.ask("[vqa]"+pair[0], chat_state)
            output_text, _ = chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    temperature=temperature,
                                    max_new_tokens=10,
                                    max_length=2000)
            print(pair[0], output_text)

            predicted_label = output_text.strip()

            if predicted_label == pair[1]:
                accuracy += 1

            completed += 1
            ans_dict['image_url'] = np.array(annot["image"]).tolist()
            ans_dict['question'] = pair[0]
            ans_dict['predicted_label'] = predicted_label
            ans_dict['GT_answer'] = pair[1]
            ans_dict['chat_state'] = str(chat_state)
        # ans_dict['original_boxes'] = str(boxes)
        final_list.append(ans_dict)



    final_acc = 0
    for entry in final_list:
        if entry['predicted_label'] == entry['GT_answer']:
            final_acc += 1

    print('Final Accuracy: ', final_acc/len(final_list))

    with open('val_whoops.json', 'w') as f:
        json.dump(final_list, f, indent=4)


        
def get_description(image_folder, mini_gpt_questions, num_images=48):
    # first for the initial prompts
    prompt_descriptions = []
    for i in tqdm(range(num_images)):
        image_answers = []

        image_url = image_folder + str(i) + ".jpeg"

        img_chat_state = CONV_VISION.copy()
        img_list = []
        chat.upload_img(image_url, img_chat_state, img_list)
        chat.encode_img(img_list)
        print("Image uploaded")

        for user_message in mini_gpt_questions:
            # reset chat for question
            chat_state = img_chat_state.copy()

            # ask the question
            chat.ask(user_message, chat_state)
            output_text, _ = chat.answer(
                conv=chat_state,
                img_list=img_list,
                temperature=temperature,
                max_new_tokens=100,
                max_length=500,
            )

            image_answers.append(output_text)

        prompt_descriptions.append(image_answers)

    return prompt_descriptions


def build_mini_gpt_questions(questions):
    mini_gpt_questions = ["[vqa] Describe this image."]
    for q in questions:
        mini_gpt_questions.append("[vqa] " + q)
    return mini_gpt_questions


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument(
        "--cfg-path",
        default="eval_configs/minigptv2_eval.yaml",
        help="path to configuration file.",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="specify the gpu to load the model."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="files",
        help="Input files",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="files",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch Size for Inference. Note: If you use multiple GPUs, this is per device.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="ta",
        help="Language to focus on",
    )

    args = parser.parse_args()
    return args


def main():
    set_seeds()

    print("Initializing Chat")
    args = parse_args()
    cfg = Config(args)

    device = "cuda:{}".format(args.gpu_id)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
        vis_processor_cfg
    )
    model = model.eval()
    chat_ = Chat(model, vis_processor, device=device)

    CONV_VISION = Conversation(
        system="",
        roles=(r"<s>[INST] ", r" [/INST]"),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="",
    )

    temperature = 0.6

    print("Chat initialized")
    lang = args.lang
    # Load the json file
    dataset = load_dataset("nlphuji/whoops", use_auth_token=True )
    print(dataset)
# DatasetDict({
#     test: Dataset({
#         features: ['image', 'designer_explanation', 'selected_caption', 'crowd_captions', 'crowd_explanations', 'crowd_underspecified_captions', 'question_answering_pairs', 'commonsense_category', 'image_id', 'image_designer'],
#         num_rows: 500
#     })
# })

 
    run_inference(dataset["test"], CONV_VISION, chat_, temperature)


if __name__ == "__main__":
    main()
