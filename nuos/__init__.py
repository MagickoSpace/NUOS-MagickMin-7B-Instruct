import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel, PeftConfig
from langchain_community.llms import HuggingFacePipeline
from . import auth

adapter_model = "MagickoSpace/NUOS-MagickMin-7B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = "google/gemma-7b-it"
tokenizer = AutoTokenizer.from_pretrained(adapter_model)
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config,device_map="auto")
model = PeftModel.from_pretrained(model, adapter_model)

pipe = pipeline("text-generation", 
               model=model, 
               tokenizer=tokenizer, 
               max_new_tokens=512
               )