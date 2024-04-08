from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-2-zephyr-1_6b', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    'stabilityai/stablelm-2-zephyr-1_6b',
    trust_remote_code=True,
    device_map="auto"
)

prompt = [{'role': 'user', 'content': 'Which famous math number begins with 1.6 ...?'}]
inputs = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    return_tensors='pt'
)

tokens = model.generate(
    inputs.to(model.device),
    max_new_tokens=1024,
    temperature=0.5,
    do_sample=True
)

print(tokenizer.decode(tokens[0], skip_special_tokens=False))
