import os
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, AutoPeftModelForCausalLM, LoraConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict, get_peft_config, get_peft_model
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from safetensors.torch import load, save_file
from huggingface_hub import HfApi, login


def load_model(model_path, peft_path=None):
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
        )
    device = model.device
    # Lora를 기본 모델에 적용
    # lora_targets=[
    # 'q_proj',
    # 'k_proj', 
    # 'v_proj',
    # 'o_proj',
    # 'gate_proj',
    # 'down_proj',
    # 'up_proj',
    # "embed_tokens",
    # 'lm_head'
    # ]
    # peft_config = LoraConfig( # Lora 설정 정의
    #     use_mora=True,  # Mora Config
    #     mora_type=6,
    #     inference_mode=False,
    #     target_modules=lora_targets,
    #     r=256,
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     bias="none",
    #     # init_lora_weights="gaussian",
    #     task_type="CAUSAL_LM")

    # model = get_peft_model(
    #     model=model,
    #     mixed=False,
    #     peft_config=peft_config)
    # model = load_state_dict_from_zero_checkpoint(
    #     model=model,
    #     checkpoint_dir=peft_path,
    #     tag=f"global_step{peft_path.split("-")[-1]}",
    #     )
    # state_dict = get_peft_model_state_dict(model=model)
    # model.to(device)
    # model.save_pretrained("outputs/")
    # model = model.merge_and_unload()
    # # print(state_dict.keys())
    # print(model.num_parameters())
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer.save_pretrained("outputs/")
    
    # Load PEFT adapter if provided
    if peft_path:
        
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=peft_path,
            is_trainable=False,
            use_safetensors=True,)
        # del test
        # model.load_adapter("outputs",)
        # # model = model.merge_and_unload()
        # print(model.num_parameters())
        # peft_model = PeftModel.from_pretrained(
        #     model=model, 
        #     model_id=peft_path,
        #     is_trainable=False,
        #     use_safetensors=True,)
        
        peft_model.to(device)
        del model

    return peft_model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    # Example usage
    base_model_path = "Gunulhona/Gemma-System-9B"  # e.g. "mistralai/Mistral-7B-v0.1"
    # base_model_path = "/home/conan/workspace/llama3_finetuning/dpo/cpo_output/checkpoint-24000/"
    # peft_path = "/home/conan/workspace/llama3_finetuning/dpo/cpo_output/checkpoint-23500"  # Path to your DPO finetuned adapter
    peft_path = "/home/conan/workspace/llama3_finetuning/outputs"
    # Load model
    try:
        model, tokenizer = load_model(base_model_path, peft_path)
        
        # Generate text
        prompt = "<start_of_turn>system\nyou are a good assistant<end_of_turn>\n<start_of_turn>user\nMarkdown 타입으로 문서 정리 예시를 만들어<end_of_turn>\n<start_of_turn>model\n"
        print(f"Prompt: {prompt}\n")
        response = generate_text(prompt, model, tokenizer)
        print(f"Generated Response: {response}")

        repo_id = "Gunulhona/Gemma-System-9B-MoRA-SimPO"
        api = HfApi()
        api.create_repo(
            repo_id=repo_id,
            repo_type="model"
        )
        api.upload_folder(
            folder_path="outputs",
            repo_id=repo_id,
            repo_type="model"
        )

    except Exception as e:
        print(traceback.format_exc())
        print("Faild to load model weight")
        print("Cannot upload to HF hub")
        