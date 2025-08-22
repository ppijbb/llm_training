import sys
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers import Gemma3ForCausalLM, Gemma3Config
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers.image_utils import load_image
from peft.peft_model import PeftModel
import tensorrt
import pprint
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (#G2MoEConfig, G2MoEForCausalLM, G2MoETextConfig,
                    G3MoEConfig, G3MoEForCausalLM, G3MoETextConfig)

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch.compiler.disable()

# Additional safety: reset any existing dynamo state
torch._dynamo.reset()
print("version of tensorrt: " ,tensorrt.__version__)

def format_parameters(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f} M"
    else:
        return str(number)

base_model_name = "google/gemma-3-4b-it"
model_architecture = G3MoEForCausalLM ## G3MoEForCausalLM
base_config = Gemma3Config.from_pretrained(base_model_name)
base_config = base_config.to_dict()
moe_config = {
        "n_shared_experts": 1,
        "n_routed_experts": 5, # 256, 15, 6
        "n_group": 4,
        "topk_group": 8,
        # "num_key_value_heads": base_config['text_config']['num_attention_heads'],
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 18,
        "router_aux_loss_coef": 0.001,
        "router_jitter_noise": 0.01,
        "input_jitter_noise": 0.01,
        "model_type": "g3moe_text",
        "no_rope_layer_interval": 4,
        "rope_scaling":{
            "rope_type": "yarn",
            "factor": 8.0
        },
        # "intermediate_size": base_config['text_config']['hidden_size'],
        "use_bfloat16": True,
    }
base_config['text_config'].update(moe_config)
base_config.update(base_config['text_config'])
model_config = G3MoEConfig(**base_config)
pprint.pprint(model_config)
# BitsAndBytesConfig int-4 config
test_model = model_architecture.from_pretrained(
    pretrained_model_name_or_path=base_model_name,
    config=model_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    # quantization_config=BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_quant_storage=torch.bfloat16)
    ).to("cuda")
# test_model = PeftModel.from_pretrained(test_model, "/mnt/disks/local-ssd/training_logs/outputs/")
tokenizer = AutoProcessor.from_pretrained(base_model_name)
with open("/home/conan_jung/workspace/llm_training/sft/config/chat_template.txt", "r") as f:
    tokenizer.chat_template = f.read()

test_text = """
안녕하세요.<end_of_turn>
<start_of_turn>system
You are a helpful assistant named Sparkle.
Always answer in shortest possible sentence.
But you should remember... Try to answer with Korean.😉<end_of_turn>
<start_of_turn>user
this is the test text message. now you must instruct the model to generate a response to this message.<end_of_turn>
"""

test_input = tokenizer.apply_chat_template(
    [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": test_text}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in Korean."},
                # {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"}                
                {"type": "image", "url": "https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg"}
            ]
        }
    ],
    # tokenize=True,
    add_generation_prompt=True,
    # return_tensors="pt",
    # return_dict=True,
)

# Load images
image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = load_image("https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg")

inputs = tokenizer(
    text=test_input,
    images=[image2],
    return_tensors="pt")
inputs = inputs.to(test_model.device)

print(test_model)
# print(test_model.config)
print(format_parameters(test_model.num_parameters()))


with torch.inference_mode():
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    response = tokenizer.batch_decode(
        test_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=GenerationConfig(
                device=test_model.device,
                max_new_tokens=10,
                do_sample=True,
                top_p=0.9,
                top_k=1,
                temperature=0.7,
                repetition_penalty=1.2,
                length_penalty=1.0,
                # num_beams=1,
                # num_beam_groups=1,
                # num_beam_hyps=1
                ),
            tokenizer=tokenizer
            )
        )[0]
    print(test_text)
    print(response[len(test_text):].split("<start_of_turn>model\n")[-1])
