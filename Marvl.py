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


def run_inference(data, CONV_VISION, chat, temperature, lang="ta"):
    use_url = False
    total, count = 0, 0

    collected_outputs = []
    for annot in tqdm(data):
        mapping = {0: "left", 1: "right"}
        if use_url:
            urls = [annot["left_url"], annot["right_url"]]
        ims = []
        for i in range(2):
            try:
                if use_url:
                    im = Image.open(requests.get(urls[i], stream=True).raw)
                    ims.append(im)
                else:
                    path = f"""files/img/{annot['language']}/images/{annot['concept']}/{annot[mapping[i]+"_img"]}"""
                    print("Using this path", path)
                    im = Image.open(path).convert('RGB')
                    ims.append(im)
            except:
                # Image URL not loading.
                pass
        overall_chat_state = CONV_VISION.copy()
        all_images = []
        # # ask the question
        print("ims", len(ims))
        mapping = {0: "first", 1: "second"}
        for i in range(len(ims)):
            chat_state = CONV_VISION.copy()
            # Each image
            img_list = []
            chat.upload_img(ims[i], chat_state, img_list)
            chat.encode_img(img_list)
            all_images.extend(img_list)

            chat_message = f"Here is the {mapping[i]} image. Please provide a one line caption."
            chat.ask(chat_message, chat_state)
            output_text, _ = chat.answer(
                conv=chat_state,
                img_list=img_list,
                temperature=temperature,
                max_new_tokens=100,
                max_length=1000,
            )
            overall_chat_state.messages.extend(chat_state.messages)

        if len(ims) > 1:
            result = {}
            question = (
                "[vqa] "
                + f"""For these two images, answer if the statement '{annot["caption"].replace("left", "first").replace("right", "second")}' is 'True' or 'False'?"""
            )
            chat.ask(question, overall_chat_state)
            # print("Overall chat state", overall_chat_state)

            # Answer can accept more than one image, provided the # images matches image tags.
            output_text, _ = chat.answer(
                conv=overall_chat_state,
                img_list=all_images,
                temperature=temperature,
                max_new_tokens=100,
                max_length=1000,
            )
            output_text = True if output_text == "True" else False
            print(output_text, annot["label"])
            print(type(output_text), type(annot["label"]))
            result = annot
            result["chat_hist"] = str(overall_chat_state)
            result["pred"] = output_text
            result["true"] =  annot["label"]
            collected_outputs.append(result)
            total += output_text == annot["label"]
            count += 1
            # for i in range(2):
            #     ims[i].save(f"images/{count}_{i}.jpg")

        
    print("Accuracy: ", total/(count+1e-8))
    with open(f"files/results/{lang}_out.json", 'w') as fout:
        json.dump(collected_outputs, fout, indent=1)

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
    file_path = f"{args.file_path}/text/{lang}/annotations_machine-translate/marvl-{lang}_gmt.jsonl"
    with open(file_path) as f:
        data = [json.loads(line) for line in f]

    print("# No of data points", len(data))
    random.shuffle(data)
    run_inference(data, CONV_VISION, chat_, temperature, lang)


if __name__ == "__main__":
    main()
