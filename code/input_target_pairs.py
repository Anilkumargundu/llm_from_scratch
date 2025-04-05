import re
import importlib
import tiktoken  ## tiktoken is the function or class that encodes any given text using the byte pair encoding (BPE) technique. BPE is used in gpt, gpt2, 3, and even 4 

with open(r"/home/anilk/GIT/llm_from_scratch/data_set/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
#print("input_text is :",raw_text[:10])
tokenizer = tiktoken.get_encoding("gpt2")   ## encoded the whole raw_text
enc_sample = tokenizer.encode(raw_text)

#print("encoded_sampe :", enc_sample[:10])
#print("decoded_sampe :", tokenizer.decode(enc_sample[:10]))
context_size = 10 #context size : means the model is trained to look at a sequence of context size (in this case 4) words or tokens to predict the next word

for i in range (1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    #print("context or present input is :", context, "------> predicted output:", desired)
    print(tokenizer.decode(context), "-------->", "------->", tokenizer.decode([desired]))
    