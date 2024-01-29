# 1inch = 2.54cm
# A4 = 210mm x 297mm

import numpy as np
from EStag.code16 import num2code
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


def save_printable_tag(num, dpi=1200, print_length=3, out_folder='beeTag'):
    factor = round(dpi / INCH * print_length / RAW_CODE_LENGTH)
    out_folder = '_'.join([out_folder, str(print_length), str(dpi)])
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    im = create_printable_code(num, factor)
    im.save(os.path.join(out_folder, str(num) + '.png'), dpi=(dpi, dpi))


def print_method1(tags_folder='../robustCodeList.npy'):
    tags = np.load(tags_folder)
    number = len(tags)
    m, n = 30, 20
    print(number)
    plt.gcf().set_size_inches(A4[0] / INCH, A4[1] / INCH)
    plt.gcf().set_dpi(1200)
    plt.subplots_adjust(left=0.10, bottom=0.10, right=0.9, top=0.9, wspace=3, hspace=3)
    for i in range(m):
        for j in range(n):
            k = i * n + j
            tag = tags[k % number]
            img = create_printable_code(tag, 20)
            plt.subplot(m, n, k + 1)
            plt.title(str(tag), size=4)
            plt.imshow(img, interpolation='nearest', cmap='gray')
            plt.axis('off')
    save_filename = 'BeeTags.pdf'
    if os.path.exists(save_filename):
        os.remove(save_filename)
    plt.savefig('BeeTags.pdf', dpi=1200)


def print_method2(tags_folder='../robustCodeList.npy', dpi=1200, print_length=3, m=10, n=15):
    tags = np.load(tags_folder)
    w, h = int(A4[0]/INCH*dpi), int(A4[1]/INCH*dpi)
    factor = round(dpi / INCH * print_length / RAW_CODE_LENGTH)
    im = np.ones((h, w), dtype=np.int32)
    code_length = factor * RAW_CODE_LENGTH
    space = 2*code_length
    partition_x_number = (w-space)//((m*2-1)*code_length+space)
    partition_y_number = (h-space)//((n*2-1)*code_length+space)
    space_w = (w - (m*2-1)*code_length*partition_x_number) // (partition_x_number + 1)
    space_h = (h - (n*2-1)*code_length*partition_y_number) // (partition_y_number + 1)
    for partition_x in range(partition_x_number):
        for partition_y in range(partition_y_number):
            for i in range(m):
                for j in range(n):
                    tag = tags[i*m+j]
                    code = num2code(tag)
                    code = np.pad(code, pad_width=1, mode='constant', constant_values=1)
                    code = np.pad(code, pad_width=1, mode='constant', constant_values=0)
                    code = np.kron(code, np.ones((factor, factor), dtype=np.int32))
                    x = i*2*code_length + (partition_x + 1) * space_w + partition_x * (m*2-1)*code_length
                    y = j*2*code_length + (partition_y + 1) * space_h + partition_y * (n*2-1)*code_length
                    im[y:y+code_length, x:x+code_length] = code
                    cv2.putText(im, str(tag), (x+int(0.2*code_length), y-int(0.2*code_length)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
    im = Image.fromarray(np.uint8(im) * 255)
    im.save('BeeTags.png', dpi=(dpi, dpi))


if __name__ == '__main__':
    f = '../robustCodeList.npy'
    tags = np.load(f)
    print(tags)
    # for i in range(20):
    #     tag = create_printable_code(tags[i], 20)
    #     tag.save(f'{tags[i]}.jpg')
