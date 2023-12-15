import argparse
import os
import random
from collections import defaultdict
from tqdm import tqdm

import json
import re
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from matplotlib import gridspec
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np
# from draw_bbox import process_single_entry
def process_history(hist):
    symbols = {
        "<s>[INST]": """
""", 
        "[/INST]":"", 
        "<s>":"" ,
        "[":"", 
        "]":""
        }

    hist = str(hist).split("messages=")[1].replace(",", """""").split("offset")[0]
    for s,r in symbols.items():
        hist = hist.replace(s, r)
    return hist + "\n"

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def show_image(image_path1="", image_path2="", text="", title="", savefig_path="out.png", rectangles=None, rectangle_annots=None):
    # fig = plt.figure()
    fig = plt.figure(figsize=(19.20,10.80))
    fig.suptitle(title, fontsize="small")
    # plt.rcParams["figure.figsize"] = (10, 10)

    gs = gridspec.GridSpec(2, 1,height_ratios=[2,1])
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    plt.rcParams.update({'font.size': 15})
    plt.xticks([])
    plt.yticks([])
    plt.box(False)

    ax2.text(0.1, 0.1, text, wrap=True)
    if image_path2 is not None:
        img =  get_concat_h(Image.open(image_path1), Image.open(image_path2))
    else:
        img_arr = np.array(json.loads(str(image_path1)), dtype=np.uint8)
        print(img_arr.shape)
        img = Image.open(image_path1) if type(image_path1==str) else Image.fromarray(img_arr, dtype='uint8')
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
    if rectangles:
        draw = ImageDraw.Draw(img, mode='RGBA')
        for i in range(len(rectangles)):
            rectangle = rectangles[i]
            annots = rectangle_annots[i]
            # Create a Rectangle patch
            x1 = float(rectangle["x1"])
            x2 = float(rectangle["x2"])
            y1 = float(rectangle["y1"])
            y2 = float(rectangle["y2"])
            # print(annots)
            draw.rectangle([(x1, y1), (x2, y2)], width=4, outline="red")
            draw.text((x1+3, y1+3), str(annots), font=fnt, fill="white", font_size=24)
    imgplot = ax1.imshow(img)
    fig.savefig(savefig_path)

def marvl():
    # Load the json file
    file_path = f"files/results/ta_out.json"
    with open(file_path) as f:
        data = json.loads(f.read())
 
    if not os.path.exists("files/results/good"):
        os.mkdir("files/results/good")

    if not os.path.exists("files/results/bad"):
        os.mkdir("files/results/bad")

    print("# No of data points", len(data))
    for idx,entry in enumerate(data[:25]):
        text = ""
        text += process_history(str(entry["chat_hist"]))
        # text += str(entry["true"])
        left_img = f"files/img/ta/images/{entry['concept']}/{entry['left_img']}"
        right_img = f"files/img/ta/images/{entry['concept']}/{entry['right_img']}"
        save_path = "files/results/good/" if entry["pred"] == entry["true"] else "files/results/bad/"
        show_image(left_img, right_img, text, title=entry["caption"], savefig_path=save_path  +str(idx)+".png")


def vcr():
    # Load the json file
    file_path = f"val_new.json"
    with open(file_path) as f:
        data = json.loads(f.read())
 
    if not os.path.exists("files/vcr/good"):
        os.mkdir("files/vcr/good")

    if not os.path.exists("files/vcr/bad"):
        os.mkdir("files/vcr/bad")

    print("# No of data points", len(data))
    random.shuffle(data)
    for idx,entry in enumerate(data[:20]):
        text = ""
        text += process_history(str(entry["chat_state"]))
        text += str(entry["GT_answer"])
        save_path = "files/vcr/good/" if entry["predicted_label"] == entry["answer_label"] else "files/vcr/bad/"

        show_image(entry["image_url"], None, text, title=entry["gt_question"], savefig_path=save_path  +str(idx)+".png",
                   rectangles=entry["new_boxes"], rectangle_annots=entry["annotations"])


def whoops():
    # Load the json file
    file_path = f"val_whoops.json"
    with open(file_path) as f:
        data = json.loads(f.read())
 
    if not os.path.exists("files/w/good"):
        os.mkdir("files/w/good")

    if not os.path.exists("files/w/bad"):
        os.mkdir("files/w/bad")

    print("# No of data points", len(data))
    random.shuffle(data)
    for idx,entry in enumerate(data[:20]):
        text = ""
        text += process_history(str(entry["chat_state"]))
        text += str(entry["GT_answer"])
        save_path = "files/w/good/" if entry["predicted_label"] == entry["GT_answer"] else "files/w/bad/"

        show_image(entry["image_url"], None, text, title=entry["gt_question"], savefig_path=save_path  +str(idx)+".png",)



if __name__ == "__main__":
    #marvl()
    # vcr()
    whoops()