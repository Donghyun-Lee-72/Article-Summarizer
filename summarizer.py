import numpy as np
import argparse
import nltk
from nltk.corpus import stopwords
import re
import heapq

class Summarizer:
    """
    Class to perform summary
    """

    def __init__(self, args: argparse):
        """
        Initializer for Summarizer

        Parameters
        ----------
        args : argparse.parse_args()
            - input_file: path to input file
            - output_file: path to output file
        """
        self.args = args
        self.original_text = self.read_textfile()

        nltk.download('punkt')
        nltk.download("stopwords")
        self.stop_words = stopwords.words('english')                    # get English stopwords
        if self.stop_words is None:
            self.stop_words = []
    

    def read_textfile(self) -> str:
        """
        Read text file to summarize

        Return
        ------
        # article_text : list[str]
        #     List of input text in sentences
        data : str
            Full text from inputfile
        """
        with open(self.args.input_file, "r", encoding='UTF-8') as f:
            data = f.readlines()
        return data[0]


    def write_textfile(self, summary_list : list[str]) -> None:
        """
        Write summarized article in text file

        Parameters
        ----------
        summary_list : list[str]
            List of summarized text in sentences
        """
        with open(self.args.output_file, "w+", encoding='UTF-8') as f:
            f.write("".join(summary_list))


    def evaluate(self, word_frequencies: dict[str, int]) -> dict[str, int]:
        """
        Evaluate the Summary compared to original text

        Parameters
        ----------
        word_frequencies : dict[str, int]
            Dictionary paring word and weighted word frequency(count)

        Return
        ------
        sentence_scores : dict[str, int]
            Dictionary paring sentence and weighted frequencies of appearing word
        """
        sentence_scores = {}
        article_text = re.sub(r'\s+', ' ', self.original_text)
        sentence_list = nltk.sent_tokenize(article_text)
        for sent in sentence_list:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies:
                    if len(sent.split(' ')) < 20:
                        sentence_scores[sent] = word_frequencies.get(word, 0) + 1
        return sentence_scores


    def generate_summary(self) -> None:
        """
        Generate summarized article
        """
        formatted_text = re.sub('[^a-zA-Z]', ' ', self.original_text)           # get only English text
        formatted_text = re.sub(r'\s+', ' ', formatted_text)

        word_frequencies = {}
        for word in nltk.word_tokenize(formatted_text):
            if word not in self.stop_words:
                word_frequencies[word] = word_frequencies.get(word, 0) + 1
        maximum_frequncy = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

        sentence_scores = self.evaluate(word_frequencies)
        summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

        summary = ' '.join(summary_sentences)
        self.write_textfile(summary)
        

def main() -> None:
    parser = argparse.ArgumentParser(description='Article Summarizer')

    parser.add_argument('--i', dest="input_file", type=str, default="./sample_article.txt",
                        help='article text file to summarize - default sample_article.txt')

    parser.add_argument('--o', dest="output_file", type=str, default="./summary.txt",
                        help='summarized article text file  - default summary.txt')

    args = parser.parse_args()
    summarizer = Summarizer(args)
    summarizer.generate_summary()

if __name__ == "__main__":
    main()