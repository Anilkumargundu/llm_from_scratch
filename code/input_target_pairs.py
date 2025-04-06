import re
import importlib
import tiktoken  ## tiktoken is the function or class that encodes any given text using the byte pair encoding (BPE) technique. BPE is used in gpt, gpt2, 3, and even 4 

with open(r"/home/anilk/GIT/llm_from_scratch/data_set/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("input_text is :",raw_text[:10]) #prints the first 10 characters of the file raw_text from the-verdict.txt
tokenizer = tiktoken.get_encoding("gpt2")   ## this line is asking tiktoken to load the method of tokening using gpt2 model
enc_sample = tokenizer.encode(raw_text) ## this line is encoding the whole raw_text in the file the verdict.txt usig the BPE method or the method gpt2 uses
dec_sample = tokenizer.decode(enc_sample) ## this line decodes enc_sample or what is there in enc_sample (in this case encoded version of raw_text)
print("encoded_sample :", enc_sample[:10]) ## prints the first 10 tokens of the file the-verdict.txt
print("decoded_sample:", dec_sample[:10]) ## this is nothing but the raw_text. so it prints first character
#print("encoded_sampe :", tokenizer.encode(raw_text[:10])) #encodes 10 characters in raw_text
#print("decoded_sampe :", tokenizer.decode(enc_sample[:10])) # this decodes 10 samples of enc_sample (prints as it is in print)

context_size = 10 #context size : means the model is trained to look at a sequence of context size (in this case 4) words or tokens to predict the next word

for i in range (1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    #print("context or present input is :", context, "------> predicted output:", desired)
    #print(tokenizer.decode(context), "-------->", "------->", tokenizer.decode([desired]))

    #