from g2moe_config import G2MoEConfig
from g2moe_model import G2MoEForCausalLM

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
print(test_model)
    
print(format_parameters(test_model.num_parameters()))
