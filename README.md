# Ovis2 Model with Gradio Interface
![Screenshot 2025-03-12 143744](https://github.com/user-attachments/assets/968ff905-e70b-451f-909a-41aa67f91603)

## Overview
This project provides a Gradio-based interface for interacting with the Ovis2-8B model. The script allows users to load the model, process image and video inputs, and generate text-based responses using a conversational chatbot.

## Features
- Load Ovis2-8B model from Hugging Face or a local path
- Gradio-powered web interface for interactive usage
- Supports both image and video-based inputs
- Implements retry logic for model loading
- Custom theme with interactive UI elements

## Installation
## Option one
Install via pinokio

## Option 2
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- CUDA-compatible GPU for optimal performance
- pip

### Install Required Packages
Run the following command to install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Usage

### Running the Interface
To start the Gradio interface, run the following command:
```bash
python app.py
```

### Command-Line Arguments
| Argument       | Default Value            | Description                                       |
|---------------|------------------------|---------------------------------------------------|
| `--model_path` | `AIDC-AI/Ovis2-8B`       | Path to model (Hugging Face model ID or local path) |
| `--port`       | `7860`                   | Port to run the Gradio interface                 |
| `--host`       | `127.0.0.1`              | Host for the Gradio interface                     |

## Model Loading
The script attempts to load the model with retry logic:
1. Tries to load from a local path if available.
2. If unsuccessful, attempts to download from Hugging Face.
3. Retries up to 3 times with exponential backoff.

## Features
- Uses `torch.bfloat16` for improved performance.
- Implements an enhanced timeout system for requests (120 seconds default).
- Supports image and video input preprocessing.
- Uses a `TextIteratorStreamer` for efficient text generation.

## UI Elements
- **Text Input:** Enter text prompts for the chatbot.
- **Image Input:** Upload an image as part of the conversation.
- **Video Input:** Upload a video to extract frames and use in conversation.
- **Chatbot:** Displays conversation history in a structured format.

## Customizations
The script includes a funky Gradio theme with animations and styling. If needed, modify the CSS in the `gr.HTML` section to change the appearance.

## License
This project follows the license terms as specified by the Ovis2-8B model.

## Acknowledgments
- [Hugging Face](https://huggingface.co/) for model hosting.
- [Gradio](https://gradio.app/) for easy-to-use web UI.
- [MoviePy](https://zulko.github.io/moviepy/) for video processing.

## Contributing
Pull requests and issues are welcome! Please follow standard Python coding guidelines.

## Contact
For any inquiries, reach out via GitHub or the Hugging Face model page.

