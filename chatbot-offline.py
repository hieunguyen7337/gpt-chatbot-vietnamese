from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config
import torch
import json

class GPT2_Chatbot():
    def __init__(self):
        # Initialize the tokenizer, model configuration and model using the GPT2 pre-trained model from Hugging Face Transformers
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2") 
        self.model_config = GPT2Config.from_pretrained("gpt2", output_hidden_states=False)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", config=self.model_config)
        self.generate_config = json.load(open("config.json"))

    def get_starting_prompt(self, name):
        # Define the starting prompt for the conversation
        starting_prompt = f"""User: Hi my name is {name}, What about you?
Chatbot: My name is GPT-2, I'm a model developed by OpenAI and am currently being used as a chatbot"""
        # Print the initial prompt to the user
        print("Chatbot: My name is GPT-2, I'm a model developed by OpenAI and am currently being used as a chatbot")
        return starting_prompt

    def model_generate(self, text_input):
        # Tokenize the input
        tokenized_text_input = torch.tensor(self.tokenizer.encode(text_input)).unsqueeze(0)
        # Generate text based on the input using the pre-trained model
        model_outputs = self.model.generate(tokenized_text_input, 
                                    do_sample=True,   
                                    top_k=self.generate_config.get("top_k"), 
                                    max_length=len(tokenized_text_input[0]) + self.generate_config.get("max_generation_length"),
                                    top_p=self.generate_config.get("top_p"),
                                    num_return_sequences=1, 
                                    pad_token_id=self.tokenizer.eos_token_id
                                    )
        # Decode the generated output
        output_text = self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)
        return output_text

    def generate_output(self, inp, num_answer, context):
        # Create the prompt for generating the model output
        input_prompt = context + f"\nUser: {inp}\nChatbot:"
        # Generate the model output based on the prompt
        output_text = self.model_generate(input_prompt)
        # Remove unnecessary text
        output_text = "\n".join(output_text.split("\n")[:(num_answer*2)])
        return output_text

    def start_chat(self):
        print("""This is a chatbot create using the Huggingface opensource GPT2 model, you can type in
break to stop the chatbot or restart to restart the chatbot conversation, please input you name:""")
        # Start the chatbot
        start = True
        while(True):
            if start:
                # Get the user's name and initialize the chatbot with a starting prompt
                name = input("User: Hi my name is ")
                starting_prompt = self.get_starting_prompt(name)
                num_answer = 2
                context = starting_prompt 
                start = False
            else:
                # Increase the number of requested answers for each round of conversation
                num_answer += 1
                context = output
            
            inp = input("User: ")
            # Allow the user to break out of the conversation or restart it
            if inp == "break":
                print("Goodbye")
                break
            if inp == "restart":
                print("----------------------------")
                start = True
                continue

            # Generate the chatbot's response to the user input
            output = self.generate_output(inp, num_answer, context)
            # Print the chatbot's response to the user
            print(output.split("\n")[-1]) 

if __name__ == '__main__':
    # Create an instance of the Chatbot class and start the chat
    chatbot = GPT2_Chatbot()
    chatbot.start_chat()