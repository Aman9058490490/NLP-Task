import re
import json
from collections import defaultdict

class WordPieceTokenizer:
    def __init__(self, vocab_size=70): #default vocabulary size is 70 
        self.vocab_size = vocab_size
        self.vocab = ["[PAD]", "[UNK]"]
        self.splits = {}

    # Function for preprocessing 
    def preprocess_data(self, text):
         return re.findall(r"\w+|[^\w\s]", text)

    # function for calculating frequency of unique words 
    def word_frequenciess(self, corpus):
        word_freqs = defaultdict(int)
        for sentence in corpus:
            for word in sentence:
                word_freqs[word] += 1
        return word_freqs
    
    def build_alphabet(self, word_freqs):
        alphabet = set()
        for word in word_freqs.keys():
            for i, char in enumerate(word):
                if i == 0:
                    alphabet.add(char)
                else:
                    alphabet.add(f"##{char}")
        return sorted(alphabet)
    #  function for adding ## to non initial words 
    def construct_splits(self, word_frequencies):
       
        return {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in word_frequencies.keys()
        }
        
   #function for computing pair scores 
    def compute_pair_scores(self):
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        
        for word, freq in self.word_frequencies.items():
            split = self.splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        return {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
    # function for merging the most frequent token in a single token
    def merge_pair(self, a, b):
        
        for word in self.word_frequencies:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            self.splits[word] = split
    # function for creating vocabulary
    def construct_vocabulary(self, corpus_file, group_no):
        with open(corpus_file, "r", encoding="utf-8") as f:
            corpus = [self.preprocess_data(line.strip()) for line in f.readlines()]
        
        self.word_frequencies = self.word_frequenciess(corpus)
        alphabet = self.build_alphabet(self.word_frequencies)
        self.vocab = ["[UNK]", "[PAD]"] + alphabet
        self.splits = self.construct_splits(self.word_frequencies)

        while len(self.vocab) < self.vocab_size:
            pair_scores = self.compute_pair_scores()
            if not pair_scores:
                break
            best_pair = max(pair_scores, key=pair_scores.get)
            self.merge_pair(*best_pair)
            new_token = best_pair[0] + best_pair[1][2:] if best_pair[1].startswith("##") else best_pair[0] + best_pair[1]
            self.vocab.append(new_token)

        vocab_file = f"vocabulary_{group_no}.txt"
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.writelines(f"{token}\n" for token in self.vocab)
        print(f"Vocabulary saved to {vocab_file}")
        # function for encode a word into sub word with the help of vocabulary
    def encode_word(self, word):

        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens
      # function for tokenizs senteces into subwords using wordpiece tokenizer
    def tokenize(self, text):
        words = self.preprocess_data(text)
        return sum([self.encode_word(word) for word in words], [])
     # function to read sentence of txt files and tokenize 
    def tokenize_from_file(self, input_txt, output_json, max_length=50):
        with open(input_txt, "r", encoding="utf-8") as f:
            data = f.readlines()

        # Tokenize sentences
        tokenized_data = {idx: self.tokenize(line.strip()) for idx, line in enumerate(data)}
        
        # maximum length (or use provided max_length)
        max_len = max(max(map(len, tokenized_data.values())), max_length)
        
        for key in tokenized_data:
            tokenized_data[key] += ["[PAD]"] * (max_len - len(tokenized_data[key]))
        
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(tokenized_data, f, indent=4)

        print(f"Tokenized sentences with padding saved to {output_json}")

if __name__ == "__main__":
    group_no = 51 
    tokenizer = WordPieceTokenizer(vocab_size=20000)
    corpus = r"C:\Users\91798\Downloads\corpus.txt"
    sample = r"C:\Users\91798\Downloads\sample_test.json" 
    #  for creating Vocabulary 
    tokenizer.construct_vocabulary(corpus, group_no)
    
    # for Tokenization and store output in JSON 
    tokenizer.tokenize_from_file(sample, f"tokenized_{group_no}.json")