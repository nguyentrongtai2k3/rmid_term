from unittest import result
from xlwt import Workbook
import imutils
import numpy as np
import cv2
from math import ceil
from model import CNN_Model
from collections import defaultdict


def get_x(s):
    return s[1][0]


da = {1: ['2'], 2: ['1'], 3: ['1'], 5: ['0'], 6: ['2'], 7: ['3'], 8: ['1'], 9: ['0', '3'], 10: ['2'],
      11: ['2'], 12: ['0'], 13: ['3'], 14: ['1'], 16: ['0'], 17: ['3'], 18: ['1'], 19: ['2'], 20: ['0'],
      22: ['1'], 25: ['2'], 26: ['0'], 27: ['2'], 29: ['3'], 30: ['1'], 32: ['1'], 33: ['0'], 34: ['1'],
      35: ['2'], 36: ['3'], 37: ['0'], 39: ['1'], 40: ['2'], 42: ['0'], 43: ['1'], 44: ['2'], 45: ['3'],
      46: ['2'], 47: ['0'], 49: ['1'], 51: ['1'], 52: ['3'], 53: ['0'], 54: ['1'], 55: ['2'], 56: ['1'],
      57: ['0'], 58: ['2'], 59: ['0'], 60: ['1'], 61: ['1'], 62: ['0'], 63: ['1'], 64: ['0'], 65: ['2'],
      66: ['0'], 68: ['1', '3'], 69: ['2'], 70: ['0'], 71: ['1'], 72: ['0'],
      73: ['3'], 74: ['2'], 75: ['1'], 76: ['1'], 77: ['2'], 78: ['0'], 79: ['3'], 80: ['1', '2'], 81: ['0', '3'],
      82: ['1'], 83: ['0'], 84: ['2'], 85: ['1', '3'], 86: ['3'], 87: ['2'], 88: ['0'], 89: ['2'], 90: ['1'],91: ['1'],
      92: ['2'], 94: ['0'], 95: ['3'], 96: ['2'], 97: ['0'], 98: ['1'], 99: ['3'], 100: ['0'],
      101: ['1'], 102: ['0'], 103: ['3'], 104: ['2'], 105: ['1'], 106: ['2'], 107: ['0'], 108: ['3'], 109: ['1'], 110: ['0']}

def get_y(s):
    return s[1][1]


def get_h(s):
    return s[1][3]


def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]


def crop_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

    img_canny = cv2.Canny(blurred, 100, 200)

    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ans_blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0

    if len(cnts) > 0:
        cnts = sorted(cnts, key=get_x_ver1)
        for i, c in enumerate(cnts):
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)

            if w_curr * h_curr > 100000:
                check_xy_min = x_curr * y_curr - x_old * y_old
                check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - (x_old + w_old) * (y_old + h_old)
                if len(ans_blocks) == 0:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr
                elif check_xy_min > 20000 and check_xy_max > 20000:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr
        sorted_ans_blocks = sorted(ans_blocks, key=get_x)
        return sorted_ans_blocks

#       co 4 cot
def process_ans_blocks(ans_blocks):
    list_answers = []
    for ans_block in ans_blocks:
        ans_block_img = np.array(ans_block[0])

        offset1 = ceil(ans_block_img.shape[0] / 6)
        for i in range(6):
            box_img = np.array(ans_block_img[i * offset1:(i + 1) * offset1, :])
            height_box = box_img.shape[0]

            box_img = box_img[14:height_box - 14, :]
            offset2 = ceil(box_img.shape[0] / 5)
            for j in range(5):
                list_answers.append(box_img[j * offset2:(j + 1) * offset2, :])

    return list_answers


def process_list_ans(list_answers):
    list_choices = []
    offset = 44
    start = 32
    for answer_img in list_answers:
        for i in range(4):
            bubble_choice = answer_img[:, start + i * offset:start + (i + 1) * offset]
            bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
            bubble_choice = bubble_choice.reshape((28, 28, 1))
            list_choices.append(bubble_choice)

    if len(list_choices) != 480:
        raise ValueError("Warning warning")
    return list_choices


def map_answer(idx):
    if idx % 4 == 0:
        answer_circle = '0'
    elif idx % 4 == 1:
        answer_circle = '1'
    elif idx % 4 == 2:
        answer_circle = '2'
    elif idx % 4 == 3:
        answer_circle = '3'
    else:
        answer_circle = ' '
    return answer_circle


def get_answers(list_answers):
    results = defaultdict(list)
    model = CNN_Model('weight.h5').build_model(rt=True)
    list_answers = np.array(list_answers)
    scores = model.predict_on_batch(list_answers / 255.0)
    for idx, score in enumerate(scores):
        question = idx // 4

        if score[1] > 0.9:
            chosed_answer = map_answer(idx)
            results[question + 1].append(chosed_answer)
    return results


def scores(results, x=0):
    for k, v in results.items():
        for i, j in da.items():
            if k == i and v == j:
                x += 1
    return x


if __name__ == '__main__':
    img = cv2.imread('1.jpg')
    # img = cv2.imread('2.jpg')
    list_ans_boxes = crop_image(img)
    list_ans = process_ans_blocks(list_ans_boxes)
    list_ans = process_list_ans(list_ans)
    answers = get_answers(list_ans)
    score = scores(answers)
    diem = score * 0.25
    print(da)
    print(answers)
    print(diem)