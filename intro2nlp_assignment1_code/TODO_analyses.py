# Imports
import spacy
from collections import Counter
import pandas as pd
pd.set_option('display.max_columns', None)

# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt


class LinguisticAnalysis:

    def __init__(self, filename: str):
        self._nlp = spacy.load('en_core_web_sm')
        self._documents = self._nlp(open(filename, "r").read())

        # To Determine
        self.n_tokens = 0
        self.types = 0
        self.n_words = 0
        self.avg_word_sentence = 0
        self.avg_word_len = 0
        self.word_class_table: pd.DataFrame()
        self.named_entities = 0

        # Part A of The Assignment
        self._part_a()

    def _part_a(self):

        # TODO: Maybe we need to remove symbols from the documents
        # TODO: Define what we are going to classify as a 'word', for now I went with everything except punctuations
        word_frequencies = self._get_word_frequencies()

        # Tokenization (1 Point)
        self.n_tokens = self._documents.__len__()
        self.types = len(word_frequencies.keys())
        # TODO: Define what we are going to classify as a 'word', for now I went with everything except punctuations
        self.n_words = sum(word_frequencies.values())
        self.avg_word_sentence = None
        self.avg_word_len = None

        # Word Classes (1.5 Points)
        n10_pos_tags = self.get_n_pos_tags(n=10)
        self.word_class_table = self.create_word_class_table(most_frequent_tags=n10_pos_tags)

        # N-Grams (1.5 points)
        # TODO: Is This The Correct Way? Currently using the lemma's of the tokens
        bi_gram_frequencies, tri_gram_frequencies = Counter(), Counter()
        for sentence in self._documents.sents:
            bi_gram_frequencies.update(self.n_grams(sentence=sentence, n=2))
            tri_gram_frequencies.update(self.n_grams(sentence=sentence, n=3))

        # Lemmatization (1 point)
        # TODO
        for sentence in self._documents.sents:
            print(sentence)
            print("---------------------------------------------")
            for token in sentence:
                print(token.text, token.lemma_)

        # Russian Presidential spokesperson Dmitry Peskov echoed Putin , saying the troops are \" a threat .
        # Lemma: say
        # Inflection: saying

        # That said , I plan to investigate this question ( among others ) further [ in ] the next couple of years .
        # Lemma: say
        # Inflection: said




        # Named Entity Recognition (1 point)
        # TODO: How to find wrongly recognized entities
        self.n_named_entities = len(self._documents.ents)
        # for ent in self._documents.ents:
        #     print(ent.text, ent.label_)

    def n_grams(self, sentence, n):
        return [sentence[i:i+n].lemma_ for i in range(len(sentence)-n+1)]

    def _get_word_frequencies(self):
        word_frequencies = Counter()

        for sentence in self._documents.sents:
            words = []
            for token in sentence:
                # Filter out Punctuation
                if not token.is_punct:
                    words.append(token.text)
            word_frequencies.update(words)

        return word_frequencies

    def create_word_class_table(self, most_frequent_tags: list):
        data = []
        for (tag, pos), occurrence in most_frequent_tags:
            frequency = round(occurrence/self.n_tokens, 2)
            word_class = [tag, pos, occurrence, frequency]
            result = self.get_n_most_frequent_tokens(tag=tag, pos=pos, n=3)
            word_class.append([x[0][2] for x in result])
            result2 = self.get_n_least_frequent_tokens(tag=tag, pos=pos, n=1)
            word_class.append([x[0][2] for x in result2])
            data.append(word_class)

        df = pd.DataFrame(data, columns=["Finegrained POS-tag",
                                         "Universal POS-Tag",
                                         "Occurrences",
                                         "Relative Tag Frequency (%)",
                                         "3 Most Frequent Tokens",
                                         "Infrequent Token"])
        return df

    def get_n_pos_tags(self, n: int):
        return Counter(((token.tag_, token.pos_) for token in self._documents)).most_common(n)

    def get_n_most_frequent_tokens(self, tag: str, pos: str, n: int):
        return Counter((token.tag_, token.pos_, token.text) for token in self._documents if (token.tag_, token.pos_) == (tag, pos)).most_common(n)

    def get_n_least_frequent_tokens(self, tag: str, pos: str, n: int):
        return Counter((token.tag_, token.pos_, token.text) for token in self._documents if (token.tag_, token.pos_) == (tag, pos)).most_common()[:-(n+1):-1]


doc_name = "data/preprocessed/train/sentences.txt"
part_A = LinguisticAnalysis(filename=doc_name)




