from transformers import GPT2LMHeadModel,  GPT2Tokenizer
import torch
import json

class GPT2_Chatbot():
    def __init__(self):
        # Initialize the tokenizer, model configuration and model using the GPT2 pre-trained model from Hugging Face Transformers
        self.tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_xl_model") 
        self.model = GPT2LMHeadModel.from_pretrained("./gpt2_xl_model")
        self.generate_config = json.load(open("generation_config.json"))

    def tokenize_text(self, text_input):
        # inputs = self.tokenizer(text, return_tensors="pt")
        # input_ids = inputs["input_ids"]
        tokenized_text_input = torch.tensor(self.tokenizer.encode(text_input)).unsqueeze(0)
        return tokenized_text_input

    def model_generate_token(self, tokenized_text_input):
        generation_output = self.model.generate(tokenized_text_input,
                                                do_sample=True,   
                                                top_k=self.generate_config.get("top_k"), 
                                                top_p=self.generate_config.get("top_p"),
                                                num_return_sequences=1, 
                                                pad_token_id=self.tokenizer.eos_token_id,
                                                max_length=len(tokenized_text_input[0]) + 1)
        return generation_output

    def model_generate(self, input_text):
        print("Chatbot:",end="")
        input_ids = self.tokenize_text(input_text)
        # while(True):
        for i in range(15):
            model_output = self.model_generate_token(input_ids)
            if model_output[0][-1:] == 198:
                break
            decoded_text = self.tokenizer.decode(model_output[0][-1:])
            print(decoded_text,end="")
            input_text += decoded_text
            input_ids = model_output
        print("")
        return input_text

    def start_chat(self):
        print("""This is a chatbot create using the Huggingface opensource GPT2-XL model, you can type in
break to stop the chatbot or restart to restart the chatbot conversation""")
        # Start the chatbot
        start = True
        while(True):
            if start:
                # initialize the chatbot with a starting prompt
                starting_prompt = """You are a chatbot, response to the user question as below."""
                context = starting_prompt 
                start = False
            else:
                context = output_text
            
            inp = input("User: ")
            # Allow the user to break out of the conversation or restart it
            if inp == "break":
                print("Goodbye")
                break
            if inp == "restart":
                print("----------------------------")
                start = True
                continue

            input_text = context + f"\nUser: {inp}\nChatbot:"
            output_text = self.model_generate(input_text)

if __name__ == '__main__':
    # Create an instance of the Chatbot class and start the chat
    chatbot = GPT2_Chatbot()
    chatbot.start_chat()