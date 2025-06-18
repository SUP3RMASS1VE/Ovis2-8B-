import subprocess
import os
import re
import logging
from typing import List, Any
from threading import Thread
import time
import argparse

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, TextIteratorStreamer, AutoImageProcessor
from moviepy.editor import VideoFileClip
from PIL import Image
import requests

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run Ovis2 model with Gradio interface')
parser.add_argument('--model_path', type=str, default='AIDC-AI/Ovis2-8B', 
                    help='Path to model, either HuggingFace model ID or local path')
parser.add_argument('--port', type=int, default=7860,
                    help='Port to run the Gradio interface on')
parser.add_argument('--host', type=str, default="127.0.0.1",
                    help='Host to run the Gradio interface on')
args = parser.parse_args()

# Patch the default timeout in requests
old_request = requests.Session.request
def new_request(self, *args, **kwargs):
    if kwargs.get('timeout') is None:
        kwargs['timeout'] = 120  # 120 seconds timeout
    return old_request(self, *args, **kwargs)
requests.Session.request = new_request

# Monkey patch AutoImageProcessor.from_pretrained to use use_fast=False
original_from_pretrained = AutoImageProcessor.from_pretrained
def patched_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
    # Set use_fast=False to maintain current behavior and silence the warning
    kwargs['use_fast'] = False
    return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
AutoImageProcessor.from_pretrained = patched_from_pretrained

# Set up model paths
model_id = args.model_path
# Use the model_id directly without creating a custom cache directory
# This will use Hugging Face's built-in caching mechanism

use_thread = False

IMAGE_MAX_PARTITION = 16

VIDEO_FRAME_NUMS = 32
VIDEO_MAX_PARTITION = 1

# load model with retry logic
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    max_retries = 3
    retry_delay = 10
    
    # First try to load from local path if it's a directory path
    if os.path.isdir(model_id):
        logger.info(f"Found local model at {model_id}, attempting to load...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                multimodal_max_length=8192,
                trust_remote_code=True,
                local_files_only=True
            ).to(device='cuda')
            logger.info("Local model loaded successfully!")
            return model
        except Exception as e:
            logger.warning(f"Failed to load local model: {str(e)}")
            logger.info("Will try to download from Hugging Face...")
    else:
        logger.info(f"Will try to load model from Hugging Face: {model_id}")
    
    # If local loading failed or model_id is a HF model ID, load from HF with caching
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading model from HuggingFace: {model_id}")
            logger.info(f"Attempt {attempt+1}/{max_retries}")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                multimodal_max_length=8192,
                trust_remote_code=True
            ).to(device='cuda')
            
            logger.info("Model loaded successfully!")
            return model
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed with error: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("All attempts to load the model failed.")
                raise

# Load the model
try:
    model = load_model()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    streamer = TextIteratorStreamer(text_tokenizer, skip_prompt=True, skip_special_tokens=True)
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

image_placeholder = '<image>'
cur_dir = os.path.dirname(os.path.abspath(__file__))

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def initialize_gen_kwargs():
    return {
        "max_new_tokens": 1536,
        "do_sample": False,
        "top_p": None,
        "top_k": None,
        "temperature": None,
        "repetition_penalty": 1.05,
        "eos_token_id": model.generation_config.eos_token_id,
        "pad_token_id": text_tokenizer.pad_token_id,
        "use_cache": True
    }

def submit_chat(chatbot, text_input):
    response = ''
    chatbot.append({"role": "user", "content": text_input})
    chatbot.append({"role": "assistant", "content": response})
    return chatbot, ''

def ovis_chat(chatbot: List[dict], image_input: Any, video_input: Any):
    conversations, model_inputs = prepare_inputs(chatbot, image_input, video_input)
    gen_kwargs = initialize_gen_kwargs()

    with torch.inference_mode():
        generate_func = lambda: model.generate(**model_inputs, **gen_kwargs, streamer=streamer)
        
        if use_thread:
            thread = Thread(target=generate_func)
            thread.start()
        else:
            generate_func()

        response = ""
        for new_text in streamer:
            response += new_text
            chatbot[-1]["content"] = response
            yield chatbot

        if use_thread:
            thread.join()

    log_conversation(chatbot)

    
def prepare_inputs(chatbot: List[dict], image_input: Any, video_input: Any):
    # conversations = [{
    #     "from": "system",
    #     "value": "You are a helpful assistant, and your task is to provide reliable and structured responses to users."
    # }]
    conversations= []

    # Process all messages except the last assistant message (which is empty)
    for i in range(0, len(chatbot) - 1):
        msg = chatbot[i]
        if msg["role"] == "user":
            conversations.append({"from": "human", "value": msg["content"]})
        elif msg["role"] == "assistant":
            conversations.append({"from": "gpt", "value": msg["content"]})
    
    # Process the last user message
    last_query = chatbot[-2]["content"].replace(image_placeholder, '')
    conversations.append({"from": "human", "value": last_query})

    max_partition = IMAGE_MAX_PARTITION
    
    if image_input is not None:
        for conv in conversations:
            if conv["from"] == "human":
                conv["value"] = f'{image_placeholder}\n{conv["value"]}'
                break
        max_partition = IMAGE_MAX_PARTITION
        image_input = [image_input]
    
    if video_input is not None:
        for conv in conversations:
            if conv["from"] == "human":
                conv["value"] = f'{image_placeholder}\n' * VIDEO_FRAME_NUMS + f'{conv["value"]}'
                break
        # extract video frames here
        with VideoFileClip(video_input) as clip:
            total_frames = int(clip.fps * clip.duration)
            if total_frames <= VIDEO_FRAME_NUMS:
                sampled_indices = range(total_frames)
            else:
                stride = total_frames / VIDEO_FRAME_NUMS
                sampled_indices = [min(total_frames - 1, int((stride * i + stride * (i + 1)) / 2)) for i in range(VIDEO_FRAME_NUMS)]
            frames = [clip.get_frame(index / clip.fps) for index in sampled_indices]
            frames = [Image.fromarray(frame, mode='RGB') for frame in frames]
        image_input = frames
        max_partition = VIDEO_MAX_PARTITION

    logger.info(conversations)
    
    prompt, input_ids, pixel_values = model.preprocess_inputs(conversations, image_input, max_partition=max_partition)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    
    model_inputs = {
        "inputs": input_ids.unsqueeze(0).to(device=model.device),
        "attention_mask": attention_mask.unsqueeze(0).to(device=model.device),
        "pixel_values": [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)] if image_input is not None else [None]
    }
    
    return conversations, model_inputs

def log_conversation(chatbot):
    logger.info("[OVIS_CONV_START]")
    for i, msg in enumerate(chatbot, 1):
        if msg["role"] == "user":
            print(f'Q{i}:\n {msg["content"]}')
        elif msg["role"] == "assistant":
            print(f'A{i}:\n {msg["content"]}')
    logger.info("[OVIS_CONV_END]")

def clear_chat():
    return [], None, "", None

with open(f"{cur_dir}/resource/logo.svg", "r", encoding="utf-8") as svg_file:
    svg_content = svg_file.read()
font_size = "2.5em"
svg_content = re.sub(r'(<svg[^>]*)(>)', rf'\1 height="{font_size}" style="vertical-align: middle; display: inline-block;"\2', svg_content)
html = f"""
<div style="background: linear-gradient(90deg, #ff00ff 0%, #ffff00 50%, #00ffff 100%); padding: 15px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
    <p align="center" style="font-size: {font_size}; line-height: 1; margin: 0; text-shadow: 2px 2px 4px #000;">
        <span style="display: inline-block; vertical-align: middle; filter: drop-shadow(0 0 5px #ff00ff);">{svg_content}</span>
        <span style="display: inline-block; vertical-align: middle; color: #fff; font-weight: bold; font-family: 'Comic Sans MS', cursive;">{model_id.split('/')[-1] if '/' in model_id else model_id}</span>
    </p>
    <center style="margin-top: 10px;">
        <font size=3 color="#fff" style="text-shadow: 1px 1px 2px #000;">
            <b style="background: -webkit-linear-gradient(#ff00ff, #ffff00); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Ovis</b> 
            has been open-sourced on 
            <a href='https://huggingface.co/{model_id}' style="color: #ff88ff; text-decoration: none; font-weight: bold;">😊 Huggingface</a> 
            and 
            <a href='https://github.com/AIDC-AI/Ovis' style="color: #88ffff; text-decoration: none; font-weight: bold;">🌟 GitHub</a>. 
            If you find Ovis useful, a like❤️ or a star🌟 would be appreciated!
        </font>
    </center>
</div>
"""

latex_delimiters_set = [{
        "left": "\\(",
        "right": "\\)",
        "display": False 
    }, {
        "left": "\\begin{equation}",
        "right": "\\end{equation}",
        "display": True 
    }, {
        "left": "\\begin{align}",
        "right": "\\end{align}",
        "display": True
    }, {
        "left": "\\begin{alignat}",
        "right": "\\end{alignat}",
        "display": True
    }, {
        "left": "\\begin{gather}",
        "right": "\\end{gather}",
        "display": True
    }, {
        "left": "\\begin{CD}",
        "right": "\\end{CD}",
        "display": True
    }, {
        "left": "\\[",
        "right": "\\]",
        "display": True
    }]

# Create a funky custom theme
funky_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.purple,
    secondary_hue=gr.themes.colors.lime,
    neutral_hue=gr.themes.colors.gray,
    font=("Comic Sans MS", "sans-serif"),
).set(
    button_primary_background_fill="linear-gradient(90deg, rgba(255,0,255,1) 0%, rgba(255,255,0,1) 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, rgba(255,255,0,1) 0%, rgba(255,0,255,1) 100%)",
    button_primary_text_color="black",
    button_secondary_background_fill="linear-gradient(90deg, rgba(0,255,255,1) 0%, rgba(255,165,0,1) 100%)",
    button_secondary_background_fill_hover="linear-gradient(90deg, rgba(255,165,0,1) 0%, rgba(0,255,255,1) 100%)",
    button_secondary_text_color="black",
    background_fill_primary="#111122",
    background_fill_secondary="#222233",
    body_text_color="#EEEEFF",
    block_title_text_color="#FF88FF",
    block_label_text_color="#88FFFF",
    input_background_fill="#333344",
)

text_input = gr.Textbox(label="prompt", placeholder="Enter your text here...", lines=1, container=False)
with gr.Blocks(title=model_id.split('/')[-1] if '/' in model_id else model_id, theme=funky_theme) as demo:
    # Add custom CSS for funky animations and styling
    gr.HTML("""
    <style>
        /* Funky animations for buttons */
        button {
            transition: all 0.3s ease !important;
            animation: rainbow-border 4s linear infinite !important;
            position: relative !important;
            z-index: 1 !important;
        }
        
        button:hover {
            transform: scale(1.05) rotate(2deg) !important;
            animation: rainbow-border 1s linear infinite !important;
        }
        
        @keyframes rainbow-border {
            0% { box-shadow: 0 0 5px #ff0000; }
            14% { box-shadow: 0 0 5px #ff7f00; }
            28% { box-shadow: 0 0 5px #ffff00; }
            42% { box-shadow: 0 0 5px #00ff00; }
            57% { box-shadow: 0 0 5px #0000ff; }
            71% { box-shadow: 0 0 5px #4b0082; }
            85% { box-shadow: 0 0 5px #9400d3; }
            100% { box-shadow: 0 0 5px #ff0000; }
        }
        
        /* Pulsating effect for the header */
        .gradio-html:first-child > div {
            animation: pulse 3s infinite alternate;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 10px #ff00ff; transform: scale(0.98); }
            50% { box-shadow: 0 0 20px #ffff00; transform: scale(1); }
            100% { box-shadow: 0 0 10px #00ffff; transform: scale(0.98); }
        }
        
        /* Funky styling for chatbot messages */
        .message.user {
            border-radius: 20px 5px 20px 5px !important;
            border-left: 4px solid hotpink !important;
            background: linear-gradient(135deg, #222233, #111122) !important;
            transform: rotate(-0.5deg) !important;
        }
        
        .message.bot {
            border-radius: 5px 20px 5px 20px !important;
            border-right: 4px solid lime !important;
            background: linear-gradient(135deg, #111122, #222233) !important;
            transform: rotate(0.5deg) !important;
        }
        
        /* Funky input field */
        input[type="text"] {
            border-radius: 15px !important;
            border: 2px solid transparent !important;
            background-image: linear-gradient(#333344, #333344), 
                              linear-gradient(90deg, hotpink, lime) !important;
            background-origin: border-box !important;
            background-clip: padding-box, border-box !important;
            transition: all 0.3s ease !important;
        }
        
        input[type="text"]:focus {
            transform: scale(1.02) !important;
            background-image: linear-gradient(#333344, #333344), 
                              linear-gradient(90deg, lime, hotpink) !important;
        }
        
        /* Radio buttons styling */
        .my_radio label {
            transition: all 0.3s ease !important;
        }
        
        .my_radio label:hover {
            transform: scale(1.05) !important;
            color: hotpink !important;
        }
        
        /* Funky title animation */
        .block.gradio-chatbot .label-wrap span {
            background: linear-gradient(to right, #ff00ff, #ffff00, #00ffff, #ff00ff);
            background-size: 200% auto;
            color: transparent !important;
            -webkit-background-clip: text !important;
            background-clip: text !important;
            animation: shine 3s linear infinite;
        }
        
        @keyframes shine {
            to {
                background-position: 200% center;
            }
        }
        
        /* Funky scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #111122;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(45deg, hotpink, lime);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(45deg, lime, hotpink);
        }
    </style>
    """)
    gr.HTML(html)
    with gr.Row():
        with gr.Column(scale=3):
            input_type = gr.Radio(choices=["image + prompt", "video + prompt"], label="Select input type:", value="image + prompt", elem_classes="my_radio")

            image_input = gr.Image(label="image", height=350, type="pil", visible=True)
            video_input = gr.Video(label="video", height=350, format='mp4', visible=False)
            
            # Examples sections removed

        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="Ovis", layout="panel", height=600, show_copy_button=True, latex_delimiters=latex_delimiters_set, type='messages')
            text_input.render()
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")
                
        def update_input_and_clear(selected):
            if selected == "image + prompt":
                visibility_updates = (gr.update(visible=True), gr.update(visible=False))
            else:
                visibility_updates = (gr.update(visible=False), gr.update(visible=True))
            clear_chat_outputs = clear_chat()
            return visibility_updates + clear_chat_outputs

        input_type.change(fn=update_input_and_clear, inputs=input_type, 
                        outputs=[image_input, video_input, chatbot, image_input, text_input, video_input])

    send_click_event = send_btn.click(submit_chat, [chatbot, text_input], [chatbot, text_input]).then(ovis_chat,[chatbot, image_input, video_input],chatbot)
    submit_event = text_input.submit(submit_chat, [chatbot, text_input], [chatbot, text_input]).then(ovis_chat,[chatbot, image_input, video_input],chatbot)
    clear_btn.click(clear_chat, outputs=[chatbot, image_input, text_input, video_input])

# Launch with settings from command line arguments
demo.launch(
    server_name=args.host,     # Host from command line
    server_port=args.port,     # Port from command line
    share=False,               # don't use Gradio's sharing feature
    inbrowser=False,           # don't open in browser
    debug=True                 # show debug information
)
