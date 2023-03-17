import requests
import json

class GPT2_Chatbot():
    def __init__(self):
        self.url = "http://127.0.0.1:5000/generate"

    def get_starting_prompt(self, name):
        # Define the starting prompt for the conversation
        starting_prompt = f"""User: Hi my name is {name}, What about you?
Chatbot: My name is GPT-2, I'm a model developed by OpenAI and am currently being used as a chatbot"""
        # Print the initial prompt to the user
        print("Chatbot: My name is GPT-2, I'm a model developed by OpenAI and am currently being used as a chatbot")
        return starting_prompt

    def generate_output(self, inp, context):
        # Create the prompt for generating the model output
        input_prompt = context + f"\nUser: {inp}\nChatbot:"
        # Call GPT2 API for output 
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({"text_input": input_prompt})
        response = requests.request("POST", self.url, headers=headers, data=payload)
        output_text = response.json().get("output")
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
            output = self.generate_output(inp, context)
            # Print the chatbot's response to the user
            print(output.split("\n")[-1]) 

if __name__ == '__main__':
    # Create an instance of the Chatbot class and start the chat
    chatbot = GPT2_Chatbot()
    chatbot.start_chat()