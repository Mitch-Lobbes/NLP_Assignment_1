# Imports
import spacy
from spacy import displacy
from collections import Counter
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt


class LinguisticAnalysis:

    def __init__(self, filename: str):
        self._nlp = spacy.load('en_core_web_sm')
        self._documents = self._nlp(open(filename, "r").read())

        # Tokenization Variables
        self._n_tokens = 0
        self._n_types = 0
        self._n_words = 0
        self._avg_sent_len = 0
        self._avg_word_len = 0

        # Word Classes Variables
        self.word_class_table: pd.DataFrame()

        # N-Grams Variables
        self._token_bigrams: list
        self._token_trigrams: list
        self._pos_bigrams: list
        self._pos_trigrams: list

        # Lemmatization Variables
        self._lemma_dict = {}
        self._sentence_dict = {}

        # Named Entity Recognition Variables
        self._n_named_entities = 0
        self._n_entity_labels = 0

        # Tokenization (1 Point)
        # TODO: Define what we are going to classify as a 'word', for now I went with everything except punctuations
        # TODO: Maybe we need to remove symbols from the documents
        self._tokenization()

        # Word Classes (1.5 Points)
        self._word_classes()

        # N-Grams (1.5 Points)
        self._n_grams()

        # Lemmatization (1 Point)
        self._lemmatization()

        # Named Entity Recognition (1 Point)
        self._named_entity_recognition()

        self._print_results()

    def _tokenization(self):
        word_frequencies = self._get_word_frequencies()

        self._n_tokens = len(self._documents)
        self._n_types = len(word_frequencies.keys())
        self._n_words = sum(word_frequencies.values())

        sentences_lengths = []
        word_lengths = []
        for sentence in self._documents.sents:
            sentence_words = 0
            for token in sentence:
                if token.text in word_frequencies:
                    sentence_words += 1
                    word_lengths.append(len(token.text))
            sentences_lengths.append(sentence_words)

        self._avg_sent_len = round(sum(sentences_lengths) / len(sentences_lengths), 2)
        self._avg_word_len = round(sum(word_lengths) / len(word_lengths), 2)

    def _word_classes(self):
        n10_pos_tags = self._get_n_most_freq_pos_tags(n=10)
        self._word_class_table = self.create_word_class_table(most_frequent_tags=n10_pos_tags)

    def _n_grams(self):

        self._token_bigrams = Counter((chunk.text for chunk in self._documents.noun_chunks
                                       if len(chunk) == 2)).most_common(3)
        self._token_trigrams = Counter((chunk.text for chunk in self._documents.noun_chunks
                                        if len(chunk) == 3)).most_common(3)
        self._pos_bigrams = Counter((chunk.root.dep_ for chunk in self._documents.noun_chunks
                                     if len(chunk) == 2)).most_common(3)
        self._pos_trigrams = Counter((chunk.root.dep_ for chunk in self._documents.noun_chunks
                                      if len(chunk) == 3)).most_common(3)

        # bi_gram_token_frequencies, tri_gram_token_frequencies = Counter(), Counter()
        # for sentence in self._documents.sents:
        #     bi_gram_token_frequencies.update(self._token_n_grams(sentence=sentence, n=2))
        #     tri_gram_token_frequencies.update(self._token_n_grams(sentence=sentence, n=3))
        #
        # self._token_bigrams = bi_gram_token_frequencies.most_common(3)
        # self._token_trigrams = tri_gram_token_frequencies.most_common(3)

    def _lemmatization(self):

        for sentence in self._documents.sents:
            for token in sentence:
                if token.lemma_ not in self._lemma_dict:
                    self._lemma_dict[token.lemma_] = [token.text]
                    self._sentence_dict[token.text] = sentence
                elif token.text not in self._lemma_dict[token.lemma_]:
                    self._lemma_dict[token.lemma_].append(token.text)
                    self._sentence_dict[token.text] = sentence

    def _named_entity_recognition(self):
        self._n_named_entities = len(self._documents.ents)
        named_entity_frequencies = Counter([ent.label_ for ent in self._documents.ents])
        self._n_entity_labels = len(named_entity_frequencies)

    def _token_n_grams(self, sentence, n):
        return [sentence[i:i+n].text for i in range(len(sentence)-n+1)]

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
        rows = []

        for (tag, pos), occurrence in most_frequent_tags:
            frequency = round(occurrence/self._n_tokens, 2)
            values = [tag, pos, occurrence, frequency]
            result = self.get_n_most_frequent_tokens(tag=tag, pos=pos, n=3)
            values.append([x[0][2] for x in result])
            result2 = self.get_n_least_frequent_tokens(tag=tag, pos=pos, n=1)
            values.append([x[0][2] for x in result2])
            rows.append(values)

        df = pd.DataFrame(rows, columns=["Finegrained POS-tag",
                                         "Universal POS-Tag",
                                         "Occurrences",
                                         "Relative Tag Frequency (%)",
                                         "3 Most Frequent Tokens",
                                         "Infrequent Token"])
        return df

    def _get_n_most_freq_pos_tags(self, n: int):
        return Counter(((token.tag_, token.pos_) for token in self._documents)).most_common(n)

    def get_n_most_frequent_tokens(self, tag: str, pos: str, n: int):
        return Counter((token.tag_, token.pos_, token.text) for token in self._documents if (token.tag_, token.pos_) == (tag, pos)).most_common(n)

    def get_n_least_frequent_tokens(self, tag: str, pos: str, n: int):
        return Counter((token.tag_, token.pos_, token.text) for token in self._documents if (token.tag_, token.pos_) == (tag, pos)).most_common()[:-(n+1):-1]

    def _print_results(self):
        print("Tokenization (1 point)")
        print(f"Number of tokens: \t {self._n_tokens}")
        print(f"Number of types: \t {self._n_types}")
        print(f"Number of words: \t {self._n_words}")
        print(f"Avg sentence length: {self._avg_sent_len}")
        print(f"Avg word length: \t {self._avg_word_len} \n")

        print("Word Classes (1.5 points)")
        print(self._word_class_table)
        print("\n")

        print("N-Grams (1.5 points)")
        print(f"Token bigrams: \t{self._token_bigrams}")
        print(f"Token trigrams: {self._token_trigrams}")
        print(f"POS bigrams: \t{self._pos_bigrams}")
        print(f"POS trigrams: \t{self._pos_trigrams}\n")

        print("Lemmatization (1 point)")
        for lemma, inflections in self._lemma_dict.items():
            if len(inflections) == 3:
                print(f"Lemma: {lemma}")
                print(f"Inflected Forms: {inflections}")
                print("Example sentences for each form: \n")
                print(f"{inflections[0]}: {self._sentence_dict[inflections[0]]}")
                print(f"{inflections[1]}: {self._sentence_dict[inflections[1]]}")
                print(f"{inflections[2]}: {self._sentence_dict[inflections[2]]}\n")
                break

        print("Named Entity Recognition (1 point)")
        print(f"Number of named entities: {self._n_named_entities}")
        print(f"Number of different entity labels: {self._n_entity_labels}")
        print("Analyze the named entities in the first five sentences:\n")
        for idx, sentence in enumerate(self._documents.sents):
            print(sentence)
            for token in sentence:
                if len(token.ent_type_) != 0:
                        print(f"Token=[{token.text}], Label={token.ent_type_}")
            print("-------------------------------")
            if idx == 4:
                break


doc_name = "data/preprocessed/train/sentences.txt"
part_A = LinguisticAnalysis(filename=doc_name)




