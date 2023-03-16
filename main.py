from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') 

# config unchange
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

# instantiate the model
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

# starting value
name = input("User:Hi my name is ")
starting_prompt = f"""User:Hi my name is {name}, What about you?
Answer:My name is GPT-2, I'm a model develop by OpenAI and am currently being serve as a chatbot"""
print("Answer:My name is GPT-2, I'm a model develop by OpenAI and am currently being serve as a chatbot")
start = True
num_answer = 2

while(True):
  inp = input("User:")
  if inp == "break":
    break
  if start:
    input_prompt = starting_prompt + "\nUser:" + inp + "\nAnswer:"
    start = False
  else:
    input_prompt = output_text + "\nUser:" + inp + "\nAnswer:"
    
  tokenized_text_input = torch.tensor(tokenizer.encode(input_prompt)).unsqueeze(0)

  model_outputs = model.generate(tokenized_text_input, 
                                  #bos_token_id=random.randint(1,30000),
                                  do_sample=True,   
                                  top_k=50, 
                                  max_length = len(tokenized_text_input[0]) + 25,
                                  top_p=0.95, 
                                  num_return_sequences=1, 
                                  pad_token_id=tokenizer.eos_token_id
                                  )
  output_text = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
  output_text = "\n".join(output_text.split("\n")[:(num_answer*2)])
  num_answer += 1

  output_answer = output_text.split("\n")[-1]
  
  print(output_answer)