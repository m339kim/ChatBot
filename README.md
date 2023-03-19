# Chatbot using Python and Pytorch

This is a command line interface chatbot built using Python and Pytorch. The chatbot uses an initial input dictionary in a JSON file which is read during the training process. The user can train the model by running `python train.py` and then initialize the chatbot by running `python chat.py`.

## Requirements

To run this chatbot project, you need to have the following installed on your system:
* Python 3.6 or later
* Pytorch
* Numpy

## Project Structure

The project contains the following files and directories:
* `train.py` - This file contains the code for training the chatbot model.
* `chat.py` - This file contains the code for initializing the chatbot and responding to user inputs.
* `intents.json` - This file contains the initial input dictionary for the chatbot.
* `model.pth` - This file contains the trained Pytorch model.
* `utils.py` - This file contains utility functions used by the chatbot.

## Usage
<img width="443" alt="chatbot" src="https://user-images.githubusercontent.com/55883282/226205819-02d2635e-47a9-4a12-9f26-28ab3a859429.png">
Follow these steps to use the chatbot:
1. Clone this repository to your local system.
2. Open a terminal or command prompt and navigate to the project directory.
3. Run `python train.py` to train the model. This step may take some time depending on the size of the input dictionary.
4. Once the model has been trained, run `python chat.py` to initialize the chatbot.
5. You can now start chatting with the chatbot by entering text in the terminal!

