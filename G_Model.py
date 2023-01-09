# 출처: https://gist.github.com/mf1024/430d7fd6ff527350d3e4b5bda0d8614e
# https://discuss.huggingface.co/t/fine-tuning-gpt2-for-movie-script-generation-in-pytorch/23906
# https://gist.github.com/mf1024/3df214d2f17f3dcc56450ddf0d5a4cd7
import logging
logging.getLogger().setLevel(logging.CRITICAL)

import torch
import numpy as np

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, set_seed  

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'  

model  = AutoModelForCausalLM.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
tokenizer = AutoTokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")

model = model.to(device)

# Function to first select topN tokens from the probability list and then based on the selected N word distribution
# get random token ID
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)



def generate_some_text(input_text, text_len = 10): 
    cur_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0).long().to(device)
    model.eval()

    with torch.no_grad():
        for i in range(text_len):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0,-1], dim=0)
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=10) 
            cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word
            
        output_list = list(cur_ids.squeeze().to('cpu').numpy())
        output_text = tokenizer.decode(output_list)
        print(output_text)


if __name__ == "__main__":
    for _ in range(3): 
        generate_some_text("a boy")
