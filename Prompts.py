path='/mnt/data/'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer

import gc
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from langchain.text_splitter import CharacterTextSplitter

# Document
count=0

print("started")
try:
    del outputs
except:
    pass
try:
    del text
except:
    pass
a=gc.collect()
torch.cuda.empty_cache()

write=open(path+"/try/"+"test.txt","w",encoding='utf-8')
data= Document
message=""
header=""
linenumber=0
for line in data:
    message=message+line

max_tokens = 400
# Optionally can also have the splitter not trim whitespace for you
tokenizer22 = Tokenizer.from_pretrained("bert-base-uncased")
splitter = HuggingFaceTextSplitter(tokenizer22, trim_chunks=True)
chunks = splitter.chunks(message, max_tokens)
texts = chunks
del tokenizer22
del splitter
gc.collect()
torch.cuda.empty_cache()            
del chunks
gc.collect()
torch.cuda.empty_cache()
QA=""
for each in texts:
    each= header+ " \n \n "+each
    each=each.replace("   "," ")
    each=each.lstrip().rstrip().strip()

    prompt = "I give you a text. If it is not in English, translate it into English. You must write twenty 'Q_A' for me. Each 'Q_A' must have a question with an answer to that question. You must always mention the entire name of the company for each question. The text for generating Q_A is as follows: \n\n"+each+" \n ****++++**** \n"#.replace("\n","")
    prompt=prompt.replace("  "," ")
    prompt=prompt.lstrip().rstrip().strip()
    inputs = tokenizer(
        prompt,
        return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs, max_new_tokens=2300, use_cache=True, do_sample=True,
        temperature=0.2, top_p=0.95)

    bbb=tokenizer.batch_decode(outputs)[0].replace(each,"").lstrip().rstrip().strip().split("****++++****")[-1]

    QA=QA+ " \n \n "+bbb+" \n \n "
    del outputs
    del inputs
    gc.collect()
    torch.cuda.empty_cache()

text =message+ " \n $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n \n "+ QA +"\n"
del QA
gc.collect()
torch.cuda.empty_cache()
print("+++++++++++++++++++++ done +++++++++++++++++++++++++")
write.write(text)
write.flush()
write.close()
count=count+1
try:
    del text
except:
    pass
try:
    del outputs
except:
    pass
gc.collect()
torch.cuda.empty_cache()
gc.collect()
del texts
torch.cuda.empty_cache()
