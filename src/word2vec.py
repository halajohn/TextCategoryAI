from enum import Enum
import logging
import json
import gensim
import config

class State(Enum):
    SEE_X = 1
    SEE_Y = 2

def read_input(input_filename):
    train_x_raw_string = []
    train_y_raw_string = []
    with open(input_filename, encoding="utf-8") as raw_data:
        for line in raw_data:
            line = line.strip()
            if line == "==###=X=###==":
                state = State.SEE_X
            elif line == "==###=Y=###==":
                state = State.SEE_Y
            else:
                if state == State.SEE_X:
                    wordlist = line.split(" ")
                    train_x_raw_string.append(wordlist)
                elif state == State.SEE_Y:
                    train_y_raw_string.append(line)
    return train_x_raw_string, train_y_raw_string

def vectorize_train_x(model, train_x_raw_string, train_y_raw_string):
    train_x = []
    for index, wordlist in enumerate(train_x_raw_string):
        current_train_x = 0
        word_count = 0
        for word in wordlist:
            if word in model:
                current_train_x += model[word]
                word_count += 1
        if word_count != 0:
            current_train_x /= word_count
            train_x.append(current_train_x)
        else:
            logging.info("index: %d has no meagingful raw string", index)
            del train_y_raw_string[index]
    return train_x

def vectorize_train_y(use_old_model, train_y_raw_string):
    train_y_set = set(train_y_raw_string)
    train_y_list = list(train_y_set)

    if use_old_model is False:
        with open("data/category_mapping.txt", "w", encoding="utf-8") as train_y_raw_and_train_y_mapping_file:
            for train_y_list_index, train_y_list_item in enumerate(train_y_list):
                train_y_raw_and_train_y_mapping_file.write(str(train_y_list_index))
                train_y_raw_and_train_y_mapping_file.write("\t")
                train_y_raw_and_train_y_mapping_file.write(train_y_list_item + "\b")
    
    train_y = []
    count = len(train_y_list)
    for train_y_raw_item in train_y_raw_string:
        for index, item in enumerate(train_y_list):
            if train_y_raw_item == item:
                break
        current_train_y = [0] * count
        current_train_y[index] = 1
        train_y.append(current_train_y)

    return train_y

def cal_word2vec(use_old_model, model_filename, input_filename, output_filename):
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    train_x_raw_string, train_y_raw_string = read_input(input_filename)
    if use_old_model is False:
        model = gensim.models.word2vec.Word2Vec(train_x_raw_string, size=config.Config.DNN_INPUT_SIZE.value)
        model.save(model_filename)
    else:
        model = gensim.models.Word2Vec.load(model_filename)
    
    train_x = vectorize_train_x(model, train_x_raw_string, train_y_raw_string)
    train_y = vectorize_train_y(use_old_model, train_y_raw_string)

    with open(output_filename, "w", encoding="utf-8") as output:
        processed_item = 0
        for train_x_item, train_y_item in zip(train_x, train_y):
            output.write("==###=X=###==\n")
            s = json.dumps(train_x_item.tolist())
            output.write(s)
            output.write("\n")
            output.write("==###=Y=###==\n")
            s = json.dumps(train_y_item)
            output.write(s)
            output.write("\n")

            processed_item += 1
            if processed_item % 100 == 0:
                logging.info("Complete %d items", processed_item)

if __name__ == "__main__":
    cal_word2vec(False, "data/word2vec.model", "data/segment_result.txt", "data/data_vector.txt")
