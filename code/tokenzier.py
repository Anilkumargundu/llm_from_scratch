import re
import importlib
import tiktoken
with open(r"/home/anilk/GIT/llm_from_scratch/data_set/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    pre_processed = re.split(r'([,.:;?!()\-_"\']|\s)', raw_text)     ### this tokenizes and makes and considers each word and the special characters shown as a token
    pre_processed = [item for item in pre_processed if item.strip()]    ## preproceed list - this removes the spaces in the result which is the previous line (it is not always advisable to remove the whitespaces)
    all_words = sorted(set(pre_processed)) ### this removes the repeated words and then sorts : first set() removes the repeatative words. sorted() will sort them in ascending order
    vocab_size = len(all_words) #### size of the all words available
    vocabulary = {token:integer for integer, token in enumerate(all_words)} ## this creates a vocabulary means for each word in all_words, an integer is assisgned for each word in the all_words in the form of (word, index)


#### adding two  special tokens (unknown and end of the list token) to the pre_processed data to make all_tokens
all_tokens = sorted(set(pre_processed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocabulary = {token:integer for integer, token in enumerate(all_tokens)}

# sample text that is to be encoded from the dataset the_verdict.txt
text1 = "tea"
text2 = "In"
text = " <|endoftext|> ".join((text1, text2)) ##space before and after <|endoftext|> otherwise this mixes with the last word a file1 and first word of file2
print("text :", text)

#print(vocabulary)

##### initiating the tiktoken for byte pair encoding
tokenizer = tiktoken.get_encoding ("gpt2")
encoded_text = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
decoded_text = tokenizer.decode(encoded_text)
print("encode_text :", encoded_text)
print("decoded_text :",decoded_text)