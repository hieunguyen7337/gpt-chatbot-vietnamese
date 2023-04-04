from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

class GPT_Chatbot():
    def __init__(self):
        # Initialize the tokenizer, model configuration and model using the GPTJ6B-VI model from Hugging Face Transformers
        self.tokenizer = AutoTokenizer.from_pretrained("./gptj6B_VI_model")
        self.model = AutoModelForCausalLM.from_pretrained("./gptj6B_VI_model", device_map="auto", load_in_8bit=True)
        self.generate_config = json.load(open("generation_config.json"))

    def tokenize_text(self, text_input):
        tokenized_text_input = torch.tensor(self.tokenizer.encode(text_input)).unsqueeze(0).to("cuda")
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
        for i in range(self.generate_config.get("max_generation_length")):
            model_output = self.model_generate_token(input_ids)
            if model_output[0][-1:] == 172:
                break
            decoded_text = self.tokenizer.decode(model_output[0][-1:])
            print(decoded_text,end="")
            input_text += decoded_text
            input_ids = model_output
        print("")
        return input_text

    def start_chat(self):
        print("""Đây là một con chatbot được tạo nhờ model GPTJ, bạn có thể gõ vào 'break' để dừng chatbot hay gõ 'restart' để bắt đầu lại cuộc nói chuyện""")
        # Start the chatbot
        start = True
        while(True):
            if start:
                # initialize the chatbot with a starting prompt
                starting_prompt = """Bạn là một con chatbot tên là GPTJ, hãy trả lời câu hỏi theo như mẫu dưới đây"""
                context = starting_prompt 
                start = False
            else:
                context = output_text
            
            inp = input("Người dùng: ")
            # Allow the user to break out of the conversation or restart it
            if inp == "break":
                print("tạm biệt")
                break
            if inp == "restart":
                print("----------------------------")
                start = True
                continue

            input_text = context + f"\nNgười dùng: {inp}\nChatbot:"
            output_text = self.model_generate(input_text)

if __name__ == '__main__':
    # Create an instance of the Chatbot class and start the chat
    chatbot = GPT_Chatbot()
    chatbot.start_chat()