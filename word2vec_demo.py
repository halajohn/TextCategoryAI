import logging
from gensim import models

def main():
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    model = models.Word2Vec.load("data/word2vec.model")

    print("input 1 word, get 10 similar words")
    print("input 2 words, calculate their similarity")
    print("input 3 words, get its corresponding word")

    while True:
        try:
            query = input()
            q_list = query.split()

            if len(q_list) == 1:
                res = model.most_similar(q_list[0], topn=10)
                for item in res:
                    print(item[0] + "," + str(item[1]))
            elif len(q_list) == 2:
                res = model.similarity(q_list[0], q_list[1])
                print(res)
            else:
                res = model.most_similar([q_list[0], q_list[1]], [q_list[2]], topn=10)
                for item in res:
                    print(item[0] + "," + str(item[1]))
            print("--------------------------------")
        except Exception as exception:
            print(repr(exception))

if __name__ == "__main__":
    main()
