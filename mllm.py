import requests
import json
import os
from transformers import AutoTokenizer
import transformers
import torch
import re
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


def GPT4(prompt,key):
    url = "https://api.openai.com/v1/chat/completions"
    api_key = key
    with open('template/template.txt', 'r') as f:
        template=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    
    textprompt= f"{' '.join(template)} \n {user_textprompt}"
    
    payload = json.dumps({
    "model": "gpt-4o", # we suggest to use the latest version of GPT, you can also use gpt-4-vision-preivew, see https://platform.openai.com/docs/models/ for details. 
    "messages": [
        {
            "role": "user",
            "content": textprompt
        }
    ]
    })
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }
    print('waiting for GPT-4 response')
    response = requests.request("POST", url, headers=headers, data=payload)
    obj=response.json()
    text=obj['choices'][0]['message']['content']
    print(text)
    # Extract the split ratio and regional prompt

    return get_params_dict(text)

def local_llm(prompt,version,model_path=None):
    if model_path==None:
        model_id = "Llama-2-13b-chat-hf" 
    else:
        model_id=model_path
    print('Using model:',model_id)
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=False, device_map='auto', torch_dtype=torch.float16)
    with open('template/template.txt', 'r') as f:
        template=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    textprompt= f"{' '.join(template)} \n {user_textprompt}"
    model_input = tokenizer(textprompt, return_tensors="pt").to("cuda")
    model.eval()
    with torch.no_grad():
        print('waiting for LLM response')
        res = model.generate(**model_input, max_new_tokens=1024)[0]
        output=tokenizer.decode(res, skip_special_tokens=True)
        output = output.replace(textprompt,'')
    return get_params_dict(output)

def get_params_dict(output_text):
    response = output_text
    # Find Final split ratio
    split_ratio_match = re.search(r"Final split ratio: ([\d.,;]+)", response)
    if split_ratio_match:
        final_split_ratio = split_ratio_match.group(1)
        print("Final split ratio:", final_split_ratio)
    else:
        print("Final split ratio not found.")
    # Find Regioanl Prompt
    prompt_match = re.search(r"Regional Prompt: (.*?)(?=\n\n|\Z)", response, re.DOTALL)
    if prompt_match:
        regional_prompt = prompt_match.group(1).strip()
        print("Regional Prompt:", regional_prompt)
    else:
        print("Regional Prompt not found.")

    image_region_dict = {'Final split ratio': final_split_ratio, 'Regional Prompt': regional_prompt}    
    return image_region_dict


def get_params_dict_hard_code(output_text=None):
    # 하드코딩된 split ratio와 regional prompt
    final_split_ratio = "1,2,1;1,1,1"  # 고정된 split ratio
    regional_prompt = (
        "Captures the dog with an engaging head tilt, eyes bright with curiosity, "
        "and a playful grin. BREAK Shows the dog lying down, paws folded, with a calm expression, "
        "its fur slightly ruffled. BREAK Presents the dog mid-play, leaping with ears perked up, "
        "full of energetic joy. BREAK Features the dog sitting serenely, a gentle breeze rustling its fur, "
        "with a calm and serene gaze."
    )

    print("Final split ratio:", final_split_ratio)
    print("Regional Prompt:", regional_prompt)

    image_region_dict = {'Final split ratio': final_split_ratio, 'Regional Prompt': regional_prompt}
    return image_region_dict
def Hard_Code(prompt, key=None):
    print("Using hardcoded values instead of GPT API.")
    return get_params_dict_hard_code()

