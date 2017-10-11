from enum import Enum
import numpy
import json
import keras
from keras.models import Sequential
from keras.layers import Dense
import config

class State(Enum):
    SEE_X = 1
    SEE_Y = 2

def read_input(input_filename):
    train_x = []
    train_y = []
    with open(input_filename, encoding="utf-8") as raw_data:
        for line in raw_data:
            line = line.strip()
            if line == "==###=X=###==":
                state = State.SEE_X
            elif line == "==###=Y=###==":
                state = State.SEE_Y
            else:
                if state == State.SEE_X:
                    train_x.append(numpy.array(json.loads(line)))
                elif state == State.SEE_Y:
                    train_y.append(numpy.array(json.loads(line)))
                    output_neuron_count = len(train_y[0])
    return train_x, train_y, output_neuron_count

def baseline_model(output_neuron_count):
    model = Sequential()

    model.add(Dense(12, input_dim=config.Config.DNN_INPUT_SIZE.value, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(output_neuron_count, activation="sigmoid"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def dnn(use_old_model, model_filename, input_filename):
    train_x, train_y, output_neuron_count = read_input(input_filename)
    train_x = numpy.array(train_x)
    train_y = numpy.array(train_y)

    if use_old_model is False:
        model = baseline_model(output_neuron_count)
        model.fit(train_x, train_y, epochs=100)
        model.save(model_filename)
    else:
        model = keras.models.load_model(model_filename)
        prediction = model.predict(train_x)
        highest = sorted(range(len(prediction[0])), key=lambda i: prediction[0][i], reverse=True)[:3]

        category_mapping = {}
        with open("data/category_mapping.txt", encoding="utf-8") as category_mapping_file:
            for line in category_mapping_file:
                line = line.strip()
                a = line.split("\t")
                category_mapping[a[0]] = a[1]
        
        for c in highest:
            print("category = %s\n" % category_mapping[str(c)])

if __name__ == "__main__":
    dnn(False, "data/dnn.model", "data/data_vector.txt")
