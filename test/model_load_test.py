import sys
import os
import torch
from tqdm.auto import tqdm
import logging
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers import Gemma3ForCausalLM, Gemma3ForConditionalGeneration, Gemma3Config, Gemma3Model
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers.image_utils import load_image

from peft.peft_model import PeftModel
import tensorrt
import pprint
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (#G2MoEConfig, G2MoEForCausalLM, G2MoETextConfig,
                    G3MoEModel, G3MoETextModel,
                    G3MoEConfig, G3MoEForCausalLM, G3MoEForConditionalGeneration, G3MoETextConfig)
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.modeling_utils import VLMS
AutoConfig.register("g3moe", G3MoEConfig)
AutoConfig.register("g3moe", G3MoEConfig)
AutoConfig.register("g3moe_text", G3MoETextConfig)
AutoModel.register(G3MoEConfig, G3MoEModel)
AutoModel.register(G3MoETextConfig, G3MoETextModel)
AutoModelForCausalLM.register(G3MoETextConfig, G3MoEForCausalLM)
VLMS.append("g3moe")

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch.compiler.disable()

# Additional safety: reset any existing dynamo state
torch._dynamo.reset()
print("version of tensorrt: " ,tensorrt.__version__)

def format_parameters(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.4f} B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.4f} M"
    else:
        return str(number)

base_model_name = "google/gemma-3-4b-it"
model_architecture = G3MoEForConditionalGeneration
base_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
base_config = base_config.to_dict()
moe_config = {
        "n_shared_experts": 1,
        "n_routed_experts": 6, # 256, 15, 6
        "n_group": 4,
        "topk_group": 8,
        # "num_key_value_heads": base_config['text_config']['num_attention_heads'],
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 18,
        "router_aux_loss_coef": 0.001,
        "router_jitter_noise": 0.01,
        "input_jitter_noise": 0.01,
        "model_type": "g3moe_text",
        "no_rope_layer_interval": 0,
        "rope_scaling":{
            "rope_type": "yarn",
            "factor": 8.0
        },
        # "intermediate_size": base_config['text_config']['hidden_size'],
        "use_bfloat16": True
    }
base_config['text_config'].update(moe_config)
base_config.update(base_config['text_config'])
model_config = G3MoEConfig (**base_config)
model_config.model_type = "gemma3"
model_config.text_config.model_type = "gemma3_text"
# BitsAndBytesConfig int-4 config
model_config.architectures = [
    "G3MoEForConditionalGeneration", 
    # "G3MoEModel", 
    # "G3MoEForCausalLM"
    ]


def main():
    logging.getLogger("transformers.processing_utils").setLevel(logging.WARN)
    test_model = model_architecture.from_pretrained(
        pretrained_model_name_or_path=base_model_name,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # trust_remote_code=True,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_quant_storage=torch.bfloat16)
        ).to("cuda").eval()
    # test_model = PeftModel.from_pretrained(test_model, "/mnt/disks/local-ssd/training_logs/outputs/")
    tokenizer = AutoProcessor.from_pretrained(base_model_name, use_fast=True)
    with open("/home/conan/workspace/llm_training/sft/config/chat_template.txt", "r") as f:
        tokenizer.chat_template = f.read()
    # logging.set_verbosity_warning()
    test_text = f"""
안녕하세요.<end_of_turn>
<start_of_turn>system
You are a helpful assistant named Sparkle.
Always answer in shortest possible sentence.
But you should remember... Try to answer with Korean.😉<end_of_turn>
<start_of_turn>user
this is the test text message. now you must instruct the model to generate a response to this message.<end_of_turn>
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
        text=test_input.replace("<bos>", "")[:-1],
        images=image2,
        return_tensors="pt").to(test_model.device)
    
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    logging.getLogger("transformers.processing_utils").setLevel(logging.INFO)
    print(test_model)
    # print(test_model.config)
    print(format_parameters(test_model.num_parameters()))
    print("Test Sequence Length:", inputs.input_ids.shape[1])

    with torch.inference_mode():
        # torch._dynamo.config.capture_dynamic_output_shape_ops = True
        fast_inputs =tokenizer(text="What's poppin?", return_tensors="pt")
        del fast_inputs["token_type_ids"]
        response =tokenizer.batch_decode(
            test_model.generate(
                **fast_inputs.to(test_model.device),
                generation_config=GenerationConfig(
                    device=test_model.device,
                ),
                tokenizer=tokenizer
            )
        )[0]
        print(response)
        response = tokenizer.batch_decode(
            test_model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    device=test_model.device,
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
                tokenizer=tokenizer
                )
            )[0]
        print(test_text)
        print("--- Model Response ---")
        print(response[len(test_text):].split("<start_of_turn>model\n")[-1])


def check_params_diff():
    
    model_1 = G3MoEForCausalLM.from_pretrained(base_model_name, config=model_config, dtype=torch.bfloat16) # G3MoEForCausalLM.from_pretrained(base_model_name, config=model_config, torch_dtype=torch.bfloat16)
    model_2 = G3MoEForConditionalGeneration.from_pretrained(base_model_name, config=model_config, dtype=torch.bfloat16)
    vision_model = Gemma3Model.from_pretrained(base_model_name, dtype=torch.bfloat16)

    def architecture_diff_check(item):
        return "moe" not in item and "mlp" not in item and "router" not in item

    def compare_model_weights(model_a, model_b, prefix_a="model_1", prefix_b="model_2"):
        """
        Compare the weights of two torch.nn.Module models and print the names of layers with different weights.
        """
        print("Start comparing weights of the two models...")
        state_dict_a = model_a.state_dict()
        state_dict_b = model_b.state_dict()

        # Find common keys
        common_keys = set(state_dict_a.keys()) & set(state_dict_b.keys())
        different_layers = []
        progress_bar =tqdm(sorted(common_keys), total=len(common_keys), desc="Comparing weights")
        for key in progress_bar:
            progress_bar.set_description(f"Comparing weights: {key}")
            tensor_a = state_dict_a[key]
            tensor_b = state_dict_b[key]
            if not torch.allclose(tensor_a, tensor_b, atol=1e-6, rtol=1e-5):
                different_layers.append(key)

        if different_layers:
            print("Layers with different weights:")
            for layer in different_layers:
                print(f" - {layer}")
        else:
            print("All weights match between the two models.")

        # Optionally, print keys only in one model
        only_in_a = set(item for item in state_dict_a.keys() - state_dict_b.keys() if architecture_diff_check(item))
        only_in_b = set(item for item in state_dict_b.keys() - state_dict_a.keys() if architecture_diff_check(item))
        if only_in_a:
            print(f"\nKeys only in {prefix_a}({len(only_in_a)}):")
            # for k in sorted(only_in_a):
            #     print(f" - {k}")
        if only_in_b:
            print(f"\nKeys only in {prefix_b}({len(only_in_b)}):")
            # for k in sorted(only_in_b):
            #     print(f" - {k}")

    # Compare the weights of the two models
    compare_model_weights(
        model_a=vision_model, 
        model_b=model_2.model, 
        prefix_a="BaseModel", 
        prefix_b="ConditionalGeneration")
    compare_model_weights(
        model_a=vision_model.vision_tower, 
        model_b=model_2.vision_tower, 
        prefix_a="VisionModel", 
        prefix_b="VisionModel_loaded")
    compare_model_weights(
        model_a=vision_model.language_model, 
        model_b=model_1.model, 
        prefix_a="VisionModel", 
        prefix_b="VisionModel_loaded")


if __name__ == "__main__":
    # check_params_diff()
    main()
    # model_1 = G3MoEForConditionalGeneration.from_pretrained(base_model_name, config=model_config, dtype=torch.bfloat16)
    # model_2 = Gemma3ForConditionalGeneration.from_pretrained(base_model_name, dtype=torch.bfloat16)
