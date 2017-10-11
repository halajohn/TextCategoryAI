# TextCategoryAI
A tool to automatically determine the category of a text data (through machine learning)
# Internal
This AI system includes the following 4 stages:
1. Pre-processing : such as converting numbers to #, converting simplified Chinese to traditional 
Chinese, to reduce the complexity of the later stages.
2. Segmentation : Using “Jieba” segmentation system to cut each sentence into words.
3. Vectorization : Using Google’s word2vec to convert each word to a 300-demensional vector, and
sum all vectors of an artical to a single 300-demensional vector to represent that article.
4. Neural network : Feed the resulting 300-demensional vector of the 3rd stage to a neural network,
which contains a hidden layer, and output the probability distribution of all the possible categories.
(Using sigmoid activation method)
