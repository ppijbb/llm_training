import torch
import logging
from transformers.image_utils import load_image
from transformers import AutoProcessor, AutoModel
from transformers.models.siglip import SiglipVisionConfig
from transformers import Gemma3ForConditionalGeneration
from transformers.generation.configuration_utils import GenerationConfig


def format_parameters(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.4f} B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.4f} M"
    else:
        return str(number)

model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")

processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

print(model)


 
with open("/home/conan_jung/workspace/llm_training/sft/config/chat_template.txt", "r") as f:
    processor.chat_template = f.read()
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

test_input = processor.apply_chat_template(
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
inputs = processor(
    text=test_input,
    images=image2,
    return_tensors="pt").to(model.device)

if "token_type_ids" in inputs:
    del inputs["token_type_ids"]
logging.getLogger("transformers.processing_utils").setLevel(logging.INFO)
# print(test_model)
# print(test_model.config)
print(format_parameters(model.num_parameters()))
print("Test Sequence Length:", inputs.input_ids.shape[1])
with torch.inference_mode():
    # torch._dynamo.config.capture_dynamic_output_shape_ops = True
    response = processor.batch_decode(
        model.generate(
            **inputs,
            generation_config=GenerationConfig(
                device=model.device,
                # max_new_tokens=10,
                do_sample=True,
                # top_p=0.9,
                # top_k=1,
                # temperature=0.7,
                # repetition_penalty=1.2,
                # length_penalty=1.0,
                # num_beams=1,
                # num_beam_groups=1,
                # num_beam_hyps=1
                ),
            tokenizer=processor
            )
        )[0]
    print(test_text)
    print("--- Model Response ---")
    print(response[len(test_text):].split("<start_of_turn>model\n")[-1])
