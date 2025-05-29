import torch
from g2moe_config import G2MoEConfig
from g2moe_model import G2MoEForCausalLM
from g3moe_config import G3MoEConfig
from g3moe_model import G3MoEForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from transformers import Gemma3ForCausalLM, Gemma3Config
import tensorrt
import pprint

print("version of tensorrt: " ,tensorrt.__version__)

def format_parameters(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f} M"
    else:
        return str(number)

base_model_name = "google/gemma-3-1b-it"
model_architecture = Gemma3ForCausalLM
base_config = Gemma3Config.from_pretrained(base_model_name)
base_config = base_config.to_dict()
base_config['text_config'].update(
    {
        "n_shared_experts": 1,
        "n_routed_experts": 16, # 256, 15, 6
        "n_group": 4,
        "topk_group": 8,
        "num_key_value_heads": base_config['text_config']['num_attention_heads'],
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 8,
        "router_aux_loss_coef": 0.001,
        "router_jitter_noise": 0.01,
        "input_jitter_noise": 0.01,
        "model_type": "g3moe_text",
        "rope_scaling":{
            "rope_type": "linear",
            "factor": 8.0
        },
        # "intermediate_size": base_config['text_config']['hidden_size'],
        "use_bfloat16": True,
    }
)
model_config = G3MoEConfig(**base_config)
pprint.pprint(model_config)

test_model = model_architecture.from_pretrained(
    pretrained_model_name_or_path=base_model_name,
    config=model_config,
    # attention_implementation="flash_attention_2"
    )#.to("cuda:1")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

test_text = """
안녕하세요.<end_of_turn><eos>
<bos><start_of_turn>system
You are a helpful assistant named Sparkle.
Always answer in shortest possible sentence.
But you should remember... Try to answer in Korean.😉
<end_of_turn><eos>
<start_of_turn>user
this is the test text message. now you must instruct the model to generate a response to this message.
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
                {"type": "text", "text": "What animal is on the candy? Name this animal in Korean."},
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"}                
            ]
        }
    ],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)

print(test_model)
# print(test_model.config)
print(format_parameters(test_model.num_parameters()))

with torch.inference_mode():
    response = tokenizer.batch_decode(
            test_model.generate(
                input_ids=test_input.to(test_model.device),
                generation_config=GenerationConfig(
                    device=test_model.device,
                    max_new_tokens=10,
                    do_sample=True,
                    top_p=0.9,
                    top_k=1,
                    temperature=0.8,
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
