# GPT2 Chatbot
## Description 
This is a Python program that creates a chatbot using the Huggingface GPT2 model. The chatbot is capable of responding to user input and continuing the conversation based on the context of the previous conversation.

## Requirements 
- Python 3.6 or higher

## Installation 
Installation will depend on your hardware, here I'm using Python 3.9.16 and cuda 11.6.

`pip install -r requirements.txt`

to download the model file for gpt2-small model run the following command

`wget -P ./gpt2_model https://huggingface.co/gpt2/resolve/main/pytorch_model.bin`

to download the model file for gpt2-XL model run the following command

`wget -P ./gpt2_xl_model https://huggingface.co/gpt2-xl/resolve/main/pytorch_model.bin`

## Usage
You can run the chatbot offline with running either `chatbot-gpt2small-offline.py` file or `chatbot-gpt2xl-offline.py` file to run the GPT2 small model or the GPT2 XL model. 

Alternatively, you can start a GPT-2 small Flask API with running `gpt2-API.py` and using the API with running `chatbot-readAPI.py`.

Once the chatbot is running, the user will be prompted to input their name. The chatbot will then introduce itself and start the conversation.

The user can input any text to start a conversation. The chatbot will respond to the user's input based on the context of the previous conversation. The conversation can be ended by typing "break" or restarted by typing "restart".

## Configuration
-	The chatbot can use either the pretrained GPT2-small model (117M parameters, 535MB model file) or the GPT2-XL model (1.5B parameters, 6.4GB model file), you can start the chatbot in terminal or start a Flask API with the model. The model is only the default pretrained so for it to act as a chatbot we preprompt the model in the form of  “User: …  Chatbot: …”.

-	Another config file “generation_config.json” is the config file for the model generation process and contain 3 value that can be edited which are top_p, top_k and max generation length.

## Credits
This code uses the GPT2 model from Hugging Face Transformers library, which is an open-source library for NLP models. It also uses PyTorch, a machine learning library.