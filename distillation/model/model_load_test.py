from g2moe_config import G2MoEConfig
from g2moe_model import G2MoEForCausalLM

test_model = G2MoEForCausalLM.from_pretrained(
    pretrained_model_name_or_path="google/gemma-2-2b-it",
    )
print(test_model.config)
