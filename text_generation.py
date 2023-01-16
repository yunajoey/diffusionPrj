
import argparse 
import logging

import numpy as np
import torch

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,  
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,  
    CTRLLMHeadModel,
    CTRLTokenizer,
    XLNetLMHeadModel,
    XLNetTokenizer,  
    BloomForCausalLM,
    BloomTokenizerFast,  
    AutoModelForCausalLM,
    AutoTokenizer,  

 
    
) 

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)



MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),  
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "gustava" : (AutoModelForCausalLM, AutoTokenizer),
    "daspar":  (AutoModelForCausalLM, AutoTokenizer),

}  

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  


#  각기 다른 모델들은 각기 다른  Different models need different input formatting and/or extra arguments
#  이거 어디서 찾나?!?!!
def prepare_gpt2_input(args, model, tokenizer, prompt_text):
    return prompt_text
    

def prepare_ctrl_input(args, model, tokenizer, prompt_text):     
    return prompt_text  


def prepare_xlnet_input(args, model, tokenizer, prompt_text): 
    return prompt_text   


def prepare_bloom_input(args, model, tokenizer, prompt_text):
    return prompt_text

def prepare_gusta_input(args, model, tokenizer, prompt_text):
    return prompt_text

def prepare_daspa_input(args, model, tokenizer, prompt_text):
    return prompt_text

# 모델들은 각기 다른 arguments를 갖고 있다. 그래서 아래처럼 해줌 
PREPROCESSING_FUNCTIONS = {
    "gpt2" : prepare_gpt2_input,   
    "ctrl" : prepare_ctrl_input, 
    "xlnet" : prepare_xlnet_input, 
    "bloom" : prepare_bloom_input, 
    "gustava" : prepare_gusta_input,
    "daspar": prepare_gusta_input,
}  


def main():
    parser = argparse.ArgumentParser()

    # model type 선언 
    parser.add_argument(
        "--model_type", 
        default=None, 
        type=str, 
        required=True, 
        help= "Model type selected in the list: " + "".join(MODEL_CLASSES.keys()), 
    )

    parser.add_argument(
        "--model_name",
        default=None, 
        type=str, 
        required=True, 
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=15)

    parser.add_argument("--repetition_penalty", type=float, default=1.5)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--num_return_sequences", type=int, default=2, help="The number of samples to generate.")
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")


    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
    )  
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)   

    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]  # (모델명, 토크나이저명)이 반환된다     
    except KeyError:
        raise KeyError("선택하신 모델은 list에 없습니다 ")  
    
    '''
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        model = GPT2Model.from_pretrained('gpt2-medium')
    '''
    model =  model_class.from_pretrained(args.model_name)
    tokenizer = tokenizer_class.from_pretrained(args.model_name)

    # prompt_message가 optional이기 때문에 만약에 없을경우 prompt 를 써달라는 input을 띄운다
    prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")

    # typing한 model 타입이 있다면은?!
    requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
    if requires_preprocessing:
        prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)  # 해당 모델에 맞는 input param과 arguements를 가져온다
        preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)
        
        # add_special_tokens (True) : encoding을 할 때 special token을 추가할지 여부 
        # '[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]' 등이 special token인데, tokenizer가 이걸 자동으로 생성한다. 
        # 이걸 막고 싶다면은 add_special_tokens=False이라고 하면은 된당 
        
        encoded_prompt = tokenizer.encode(
             preprocessed_prompt_text, add_special_tokens=False,return_tensors="pt"
        )

    encoded_prompt = encoded_prompt.to(args.device)  
    # "a boy" 라고 prompt를 했을 경우 tensor([[  64, 2933]])   torch.Size([1, 2]), encoded_prompt.size()[-1]는 2이다. 즉 input_id의 갯수  
    
    if encoded_prompt.size()[-1] == 0: # 즉  prompt에 해당되는 input_id가 없을 경우는?
        input_ids = None
    else:
        input_ids = encoded_prompt # 인코드된 번호가 있으면 그대로 input 

    # https://huggingface.co/docs/transformers/main_classes/text_generation 
    # generate에는 45개의 params이 있다 
    # input_id는 required param!! 

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length = args.length + len(encoded_prompt[0]),  # 문장을 generation할때 만든 최대 길이 
        top_k = args.k,  #  The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p = args.p,   #    only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        repetition_penalty=args.repetition_penalty,  # 문장 generation 할때 반복하는 정도? > 1이 되면은 반복의 정도가 낮아진다 
        do_sample=True,
        num_return_sequences=args.num_return_sequences,  # 문장을 새롭게 반환하는 갯수   
    )
    
   # print(len(output_sequences.shape))  # "a boy"라고 했을 때 , torch.Size([1, 17])  len()은 2개이다 
   # print(output_sequences.squeeze_().shape)  # squeeze를 하면은 torch.Size([17])
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()  
            
    generated_sequences = list()
    for idx, sequence in enumerate(output_sequences):
        generated_sequence = sequence.tolist()  # torch, tensor ==> list로 만들어줌 
        
        # decoding the text 
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        generated_sequences.append(text)   
    print(generated_sequences)

   
    
if __name__ == "__main__":
    main()

