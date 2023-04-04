# GPT Chatbot
## Description 
This is a Python program that creates a chatbot using the Huggingface GPT2 model. The chatbot is capable of responding to user input and continuing the conversation based on the context of the previous conversation.

## Requirements 
- Python 3.9 and cuda 11.8

## Installation 
Run the 2 command below for installation.

`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

`pip install -r requirements.txt`

To download the model file run the following command

`wget -P ./gptj6B_VI_model https://huggingface.co/VietAI/gpt-j-6B-vietnamese-news/resolve/main/pytorch_model.bin`

## Usage
You can run the chatbot offline in terminal with running the `chatbot-gpt-inference.py` file to run the GPT model. You can also finetune the model by runing the `training.py` file 

The user can input any text to start a conversation. The chatbot will respond to the user's input based on the context of the previous conversation. The conversation can be ended by typing "break" or restarted by typing "restart".

## Configuration
-	The chatbot employs the GPTJ-6B-vietnamese model, which has 6.13 billion parameters. You can start the chatbot in terminal or fine-tune the model. The default pretrained model is used for the chatbot, so we preprompt the model in the form of "User: ... Chatbot: ..." to make it function as a chatbot.

-	The config file “generation_config.json” is the config file for the model generation process and contain 3 value that can be edited which are top_p, top_k and max generation length.

## Credits
This code use the GPTJ-6B-vietnamese model developed by VietAI, coupled with the Hugging Face Transformers and Datasets libraries, both of which are open-source frameworks for NLP models. In addition, the bitsandbytes library is utilized for model loading, while the peft library is utilized for training. This code also ultilize PyTorch, a prominent machine learning library.
