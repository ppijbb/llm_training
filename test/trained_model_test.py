from peft import PeftModel
from transformers.image_utils import load_image
import sys
import os
# sft 디렉토리가 상위 경로에 있다면 import 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import torch
import copy
from transformers.generation.configuration_utils import GenerationConfig
from models import G3MoEModel, G3MoETextModel, G3MoEConfig, G3MoEForCausalLM, G3MoEForConditionalGeneration, G3MoETextConfig
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor
from transformers.modeling_utils import VLMS
AutoConfig.register("g3moe", G3MoEConfig)
AutoConfig.register("g3moe", G3MoEConfig)
AutoConfig.register("g3moe_text", G3MoETextConfig)
AutoModel.register(G3MoEConfig, G3MoEModel)
AutoModel.register(G3MoETextConfig, G3MoETextModel)
AutoModelForCausalLM.register(G3MoETextConfig, G3MoEForCausalLM)
VLMS.append("g3moe")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



# 학습된 체크포인트 경로 (예시)
checkpoint_path = "/mls/conan/training_logs/outputs/checkpoint-150"



# 모델과 토크나이저 로딩
base_model_name = "Gunulhona/Gemma-3-4B"
model_architecture = G3MoEForConditionalGeneration

base_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
base_config = base_config.to_dict()
moe_config = {
      "n_shared_experts": 1,
      "n_routed_experts": 8,
      "n_group": 2,
      "topk_group": 2,
      "num_experts_per_tok": 2,
      "first_k_dense_replace": 8,
      "router_aux_loss_coef": 9e-1,
      "router_jitter_noise": 1e-05,
      "input_jitter_noise": 1e-05,
      "router_z_loss_coef": 1e-2,
      "ema_alpha": 0.99,
      "balancing_strength": 5e-2,
      "no_rope_layer_interval": 4,
      "use_sliding_window": True,
      "rope_scaling": {
        "rope_type": "yarn",
        "factor": 8.0
      },
        "use_bfloat16": True
    }
if "text_config" not in base_config:
    base_config['text_config'] = copy.deepcopy(base_config)

base_config['text_config'].update(moe_config)
base_config.update(base_config['text_config'])
model_config = G3MoEConfig(**base_config)
model_config.model_type = "gemma3"
model_config.text_config.model_type = "gemma3_text"
# BitsAndBytesConfig int-4 config
model_config.architectures = [
    "G3MoEForConditionalGeneration", 
    # "G3MoEModel", 
    # "G3MoEForCausalLM"
    ]

model = PeftModel.from_pretrained(
    model=model_architecture.from_pretrained(
            pretrained_model_name_or_path=base_model_name,
            config=model_config,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            attn_implementation="flash_attention_3",
        ).to("cuda"),
    model_id=checkpoint_path,
    )
model.merge_and_unload()
# 모델을 eval 모드로 설정
model.eval()

print("✅ 체크포인트에서 모델과 토크나이저를 성공적으로 로드했습니다.")
tokenizer = AutoProcessor.from_pretrained(base_model_name, use_fast=True)
with open("/home/conan/workspace/llm_training/sft/config/chat_template.txt", "r") as f:
    tokenizer.chat_template = f.read()
    # logging.set_verbosity_warning()
test_text = f"""
You are a helpful assistant named Sparkle.
Always answer in shortest possible sentence.<end_of_turn>
""" * 1

test_input = tokenizer.apply_chat_template(
    [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": test_text.strip()}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this image?"},
                # {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
                # {"type": "image", "url": "https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg"},
                {"type": "image"}
            ]
        }
    ],
    # tokenize=True,
    add_generation_prompt=True,
    # return_tensors="pt",
    # return_dict=True,
)
sample_image_urls = [
    "https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg",
    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
    "https://ocr.space/Content/Images/table-ocr-original.webp",
]
ran_image = random.choice(sample_image_urls)
image = load_image(ran_image)

inputs = tokenizer(
    text=test_input.replace("<bos>", "")[:-1],
    images=image,
    return_tensors="pt").to(model.device)

if "token_type_ids" in inputs:
    del inputs["token_type_ids"]

# 실제 forward pass를 위해 sample inputs 사용
with torch.inference_mode():
    # torch._dynamo.config.capture_dynamic_output_shape_ops = True
    fast_inputs =tokenizer(text="Jack Hallow's What's poppin?", return_tensors="pt")
    del fast_inputs["token_type_ids"]
    response =tokenizer.batch_decode(
        model.generate(
            **fast_inputs.to(model.device),
            generation_config=GenerationConfig(
                device=model.device,
            ),
            tokenizer=tokenizer
        )
    )[0]
    print(response)
    response = tokenizer.batch_decode(
        sequences=model.generate(
            **inputs,
            generation_config=GenerationConfig(
                device=model.device,
                # max_new_tokens=10,
                # do_sample=True,
                # top_p=0.9,
                # top_k=1,
                # temperature=0.7,
                # repetition_penalty=1.2,
                # length_penalty=1.0,
                # num_beams=1,
                # num_beam_groups=1,
                # num_beam_hyps=1
                ),
            tokenizer=tokenizer,
            ),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]
    print("The Sample Image was:", sample_image_urls.index(ran_image))
    print(test_text)
    print("--- Model Response ---")
    print(response)