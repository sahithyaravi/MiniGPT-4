import argparse
import os
import random
from collections import defaultdict
from tqdm import tqdm

import json
import re
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import gridspec

def process_history(hist):
    hist = hist.split("messages=")[1].replace(",", "\n").split("offset")[0]
    return hist

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def show_image(image_path1="", image_path2="", text="", title="", savefig_path="out.png"):
    fig = plt.figure()
    fig.suptitle(title, fontsize="small")
    # plt.rcParams["figure.figsize"] = (25, 20)

    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    plt.rcParams.update({'font.size': 8})
    plt.xticks([])
    plt.yticks([])
    plt.box(False)

    ax2.text(0.1, 0.1, text, wrap=True)
    img =  get_concat_h(Image.open(image_path1), Image.open(image_path2))
    imgplot = ax1.imshow(img)
    # plt.show()
    fig.savefig(savefig_path)

def main():
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





if __name__ == "__main__":
    main()
