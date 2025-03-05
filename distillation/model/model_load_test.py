from g2moe_config import G2MoEConfig
from g2moe_model import G2MoEForCausalLM
from transformers import AutoTokenizer

def format_parameters(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f} M"
    else:
        return str(number)


test_model = G2MoEForCausalLM.from_pretrained(
    pretrained_model_name_or_path="google/gemma-2-2b-it",
    )
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
print(test_model)
    
print(format_parameters(test_model.num_parameters()))
test_input = tokenizer(
        text="this is the test text message.",
        return_tensors="pt",
    )["input_ids"]
print(test_input)
print(test_model(
    input_ids=test_input,
    inputs_embeds=None,
))