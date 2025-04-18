import torch
from g2moe_config import G2MoEConfig
from g2moe_model import G2MoEForCausalLM
from transformers import AutoTokenizer
from transformers.models.gemma3 import Gemma3ForCausalLM
import tensorrt
print("version of tensorrt: " ,tensorrt.__version__)

def format_parameters(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f} M"
    else:
        return str(number)


test_model = G2MoEForCausalLM.from_pretrained(
    pretrained_model_name_or_path="google/gemma-2-2b-it",
    ).to("cuda:1")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

test_input = """
<bos><start_of_turn>system
You are a helpful assistant named G2MoE.<end_of_turn><eos>
<start_of_turn>user
this is the test text message. now you must instruct the model to generate a response to this message.<end_of_turn><eos>
<bos><start_of_turn>model
""".lstrip()
test_input = tokenizer(
        text=test_input,
        return_tensors="pt",
    )["input_ids"]

print(test_model)
print(test_model.config)
print(format_parameters(test_model.num_parameters()))

with torch.inference_mode():
    print(
        tokenizer.batch_decode(
            test_model.generate(
                input_ids=test_input.to(test_model.device),
                # inputs_embeds=None,
                )
            )
        )
