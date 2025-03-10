# GPT-2 Text Generation Application

This is a simple text generation application that uses GPT-2, a state-of-the-art language model developed by OpenAI, to generate human-like text. The application provides a graphical user interface (GUI) built with PyQt5, where users can input a prompt and generate text based on that input.

## Features:
- Text generation with GPT-2
- Simple user interface with PyQt5
- Input a prompt and get text generated by GPT-2
- The generated text can be easily copied and used for various purposes

## Installation:

To run this application, you will need to have Python installed on your machine. Follow the steps below to set it up:

1. Clone this repository:
    ```bash
    git clone https://github.com/genzo61/gpt-2-text-generation.git
    ```
   

2. Run the application:
    ```bash
    python main.py
    ```

## Requirements:
- Python 3.x
- PyQt5
- transformers
- torch
- numpy

Make sure all the dependencies are installed before running the application.

## How It Works:
- The user types a text prompt in the input field (`lineEdit`).
- The `Generate Text` button triggers the GPT-2 model to generate text based on the input.
- The generated text is then displayed in the output field (`lineEdit_2`).

## Contributing:
Feel free to fork this project, make improvements, and create pull requests. Contributions are always welcome!

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements:
- OpenAI for GPT-2
- Hugging Face for the Transformers library
- PyQt5 for creating the graphical user interface
