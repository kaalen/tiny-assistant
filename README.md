# Tiny LLM Assistant for Raspberry Pi

Experimenting with smaller LLMs that can run on commodity hardware like Raspberry Pi as tiny personal assistant if you're GPU poor like me or just want to tinker around. Because why not! Of course, you can run this on more powerful hardware too.

## Setup

1. Clone this repository 
    ```
    git clone https://github.com/kaalen/tiny-assistant.git tiny-assistant
    cd tiny-assistant
    ```
2. Set your git repo username and email:

    ```
    git config user.name "FIRST_NAME LAST_NAME"
    git config user.email "MY_NAME@example.com"
    ```
3. Create a virtual environment and activate it
    ```
    python -m venv .venv
    source .venv/bin/activate
    ```
4. Install python dependencies `pip install -r requirements.txt`
5. Make sure you have `ffmpeg` for your OS installed. If you don't have you can download from [ffmpeg.org](https://ffmpeg.org/download.html)
6. Download the LLM models you'll be using. Here's some suggested links to models on Hugging Face:
    * TheBloke/Llama-2-7b-Chat-GGUF [TheBloke/llama-2-7b-chat.Q4_K_M.gguf](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main)
    * TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF [TheBloke/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tree/main)
7. Run the code

    `python ./src/app_chat.py`

## Hardware and Dev Environment Setup

I tested this code on Raspberry Pi 5 8GB running Raspbian Bookworm distro. 

## References

This project is inspired by [A Weekend AI Project: Using Speech Recognition, PTT, and a Large Action Model on a Raspberry Pi](https://medium.com/p/ac8d839d078a) by [Dmitrii Eliuseev](https://dmitryelj.medium.com/).

