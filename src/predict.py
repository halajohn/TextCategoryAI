import dnn
import segment
import word2vec

def main():
    segment.segmentation("data/test.txt", "data/segment_result_test.txt")
    word2vec.cal_word2vec(True, "data/word2vec.model", "data/segment_result_test.txt", "data/data_vector_test.txt")
    dnn.dnn(True, "data/dnn.model", "data/data_vector_test.txt")

if __name__ == "__main__":
    main()
