from flask import Flask, request
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config
import torch
import json

tokenizer = GPT2Tokenizer.from_pretrained("gpt2") 
model_config = GPT2Config.from_pretrained("gpt2", output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained("gpt2", config=model_config)
generate_config = json.load(open("config.json"))
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def chat():
    # Get the input text
    req_body = request.get_json()
    text_input = req_body.get('text_input')

    # Tokenize the input
    tokenized_text_input = torch.tensor(tokenizer.encode(text_input)).unsqueeze(0)
    # Generate text based on the input using the pre-trained model
    model_outputs = model.generate(tokenized_text_input, 
                                do_sample=True,   
                                top_k=generate_config.get("top_k"), 
                                max_length=len(tokenized_text_input[0]) + generate_config.get("max_generation_length"),
                                top_p=generate_config.get("top_p"),
                                num_return_sequences=1, 
                                pad_token_id=tokenizer.eos_token_id
                                )
    # Decode the generated output
    output_text = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
    output_text = "\n".join(output_text.split("\n")[:(len(text_input.split("\n")))])

    # Return model response in JSON format
    return {'output': output_text}

if __name__ == '__main__':
    app.run(debug=True)