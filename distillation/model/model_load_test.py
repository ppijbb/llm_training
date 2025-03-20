import torch
from g2moe_config import G2MoEConfig
from g2moe_model import G2MoEForCausalLM
from transformers import AutoTokenizer
from transformers.models.gemma3 import Gemma3ForCausalLM

def format_parameters(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f} M"
    else:
        return str(number)


test_model = G2MoEForCausalLM.from_pretrained(
    pretrained_model_name_or_path="google/gemma-2-2b-it",
    ).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
print(test_model)
    
print(format_parameters(test_model.num_parameters()))
test_input = tokenizer(
        text="this is the test text message. i am a",
        return_tensors="pt",
    )["input_ids"]

print(test_model.config)

with torch.inference_mode():
    print(
        tokenizer.batch_decode(
            test_model.generate(
                input_ids=test_input.to(test_model.device),
                # inputs_embeds=None,
                top_k=1
                )
            )
        )
