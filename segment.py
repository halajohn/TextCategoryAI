import logging
from enum import Enum
import jieba
from opencc import OpenCC

class State(Enum):
    SEE_TITLE = 1
    SEE_DESCRIPTION = 2
    SEE_CATEGORY = 3

def read_input(filename):
    current_train_x = ""
    current_train_y = ""
    train_x = []
    train_y = []

    with open(filename, "r", encoding="utf-8") as raw_data:
        open_cc = OpenCC("s2t")
        for line in raw_data:
            line = line.strip()
            line = open_cc.convert(line)
            if line == "==###=title=###==":
                state = State.SEE_TITLE
                if current_train_x.strip() and current_train_y.strip():
                    train_x.append(current_train_x)
                    train_y.append(current_train_y)
                    current_train_x = ""
                    current_train_y = ""
            elif line == "==###=description=###==":
                state = State.SEE_DESCRIPTION
            elif line == "==###=category=###==":
                state = State.SEE_CATEGORY
            else:
                if state == State.SEE_TITLE:
                    current_train_x = line
                elif state == State.SEE_DESCRIPTION:
                    current_train_x += " " + line
                elif state == State.SEE_CATEGORY:
                    current_train_y = line
        if current_train_x.strip() and current_train_y.strip():
            train_x.append(current_train_x)
            train_y.append(current_train_y)
            current_train_x = ""
            current_train_y = ""
    return train_x, train_y

def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def is_number(value):
    if is_int(value) == True:
        return True
    elif is_float(value) == True:
        return True
    else:
        return False

def pre_processing(words):
    if is_number(words):
        return "#"
    else:
        return words

def segmentation(input_filename, output_filename):
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    jieba.load_userdict("data/dict.txt.big")

    stopwordset = set()
    with open("data/stop_words.txt", "r", encoding="utf-8") as stopwords:
        for line in stopwords:
            stopwordset.add(line.strip("\n"))

    train_x, train_y = read_input(input_filename)
    with open(output_filename, "w", encoding="utf-8") as output_file:
        processed_item = 0
        for train_x_item, train_y_item in zip(train_x, train_y):
            words = jieba.cut(train_x_item)
            output_file.write("==###=X=###==\n")
            for word in words:
                word = pre_processing(word)
                if word not in stopwordset:
                    output_file.write(word + " ")
            output_file.write("\n")
            output_file.write("==###=Y=###==\n")
            output_file.write(train_y_item)
            output_file.write("\n")

            processed_item += 1
            if processed_item % 100 == 0:
                logging.info("Complete %d items", processed_item)

if __name__ == "__main__":
    segmentation("data/data.txt", "data/segement_result.txt")
