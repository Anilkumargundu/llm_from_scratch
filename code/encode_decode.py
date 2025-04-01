import re
with open(r"/home/anilk/LLM_Models/data_set/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    pre_processed = re.split(r'([,.:;?!()\-_"\']|\s)', raw_text)     ### this tokenizes and makes and considers each word and the special characters shown as a token
    pre_processed = [item for item in pre_processed if item.strip()]    ## preproceed list - this removes the spaces in the result which is the previous line (it is not always advisable to remove the whitespaces)
    all_words = sorted(set(pre_processed)) ### this removes the repeated words and then sorts : first set() removes the repeatative words. sorted() will sort them in ascending order
    vocab_size = len(all_words) #### size of the all words available
    vocabulary = {token:integer for integer, token in enumerate(all_words)} ## this creates a vocabulary means for each word in all_words, an integer is assisgned for each word in the all_words in the form of (word, index)


class SimpleTokenizerV1:
    def __init__(self, vocabulary):
        self.str_to_int = vocabulary
        self.int_to_str = {i:s for s,i in vocabulary.items()} ## this is used for decoding purpose
    
    def encode (self, text):
        pre_processed = re.split(r'([,.:;?!()\-_"\']|\s)', raw_text)     ### this tokenizes and makes and considers each word and the special characters shown as a token
        pre_processed = [item for item in pre_processed if item.strip()]
        ids = [self.str_to_int[s] for s in pre_processed]
        return ids
    

    def decode (self, ids):
        text = " ".join([self.int_to_str[i] for i in ids]) ## this replaces space befor the specified pnctuations
        text = re.sub(r'[,.:;?!()\-_"\']', r'\1', text)
        return text

# Example usage of the tokenizer
tokenizer = SimpleTokenizerV1(vocabulary)

# Encode a sample text
sample_text = "This is a test."
encoded = tokenizer.encode(sample_text)
#print("Encoded:", encoded)

# Decode the encoded list of indices back to text
#decoded = tokenizer.decode(encoded)
#print("Decoded:", decoded)
print(encoded)