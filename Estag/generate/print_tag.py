
import numpy as np
from code16 import num2code
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import os

INCH = 25.4
RAW_CODE_LENGTH = 8
A4 = [210, 297]


def create_printable_code(num, factor):
    code = num2code(num)
    code = np.pad(code, pad_width=1, mode='constant', constant_values=1)
    code = np.pad(code, pad_width=1, mode='constant', constant_values=0)
    im = Image.fromarray(np.uint8(code) * 255)
    im = im.resize(size=(code.shape[0] * factor,) * 2, resample=Image.NEAREST)
    return im


# 生成pdf
def print_method1(m, n, width, tag_id, tag_pdf):
    tags_folder = '../robustCodeList.npy'
    tags = np.load(tags_folder)
    number = len(tags)
    print(number)
    plt.gcf().set_size_inches(A4[0] / INCH, A4[1] / INCH)
    plt.gcf().set_dpi(1200)
    # create tag：1.5 = 3mm, 1 =4mm, 2 = 2.5mm,1.7 for bumblebee
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=width, hspace=width)

    for i in range(m):
        for j in range(n):
            k = i * n + j
            tag = tags[k % number]
            img = create_printable_code(tag, 20)
            plt.subplot(m, n, k + 1)
            # plt.title(str(tag), size=3, x=0.5,y=-1.05) for bumblebees
            plt.title(str(tag), size=4, x=0.5,y= tag_id) # y value determine the position of title
            # plt.ylabel(str(tag), size=4)
            plt.imshow(img, interpolation='nearest', cmap='gray')
            plt.axis('off')
    plt.savefig(tag_pdf, dpi=1200)


if __name__ == '__main__':

    print_method1(30, 20, 1.7, -0.9, 'test_0122.pdf')

