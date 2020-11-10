# Basic-Bot

After seeing the post about minGPT I decided to start a little project I've been meaning to do. I decided to train a GPT model using the multi-year chat history of one of my friends on Discord. I had a lot to go on but I decided to also add some conversations from the reddit corpus just make the results more interesting. I trained the model on Google Colab since they have freely assessable GPU's and I created a Discord bot that uses the model to make for some entertaining reply's. I hosted the bot on a tiny AWS linux server so that I can keep it running 24/7. 

## Installation

Clone the public git repository and create a virtual environment with:

```bash
python3 -m venv --copies Env/
```

Install all the requirements using pip:

```bash
pip install -r requirements.txt
```

Depending on your operating system you will need to manually install the latest version of Pytorch. Make sure to select the CPU only version if you don't have a GPU or your going to have a lot of trouble (https://pytorch.org/get-started/locally/).

## Training

Upload your own conversation data into ```Data/```. Make sure each separate message is separated by the \n special character. Then go through the ```make_dataset.ipynb``` notebook to parse the Discord conversation text and inject some random Reddit dialog. Next this text needs to be used as input to train the minGPT model. If you don't have your own GPU, it is really easy to get it trained using Google Colab. First mount your Google Drive and again clone this repository into a folder on your Google Drive. Upload your data onto the cloned repository and then open the  ```train_model.ipynb``` notebook using Google Colab. 

## Deploy the Discord Bot

First create Bot in discord and save the key in a file called ```config.yaml``` in the root of the repository. The file should look something like this:

```bash
token : "${token}"
```

Next, run the  ```Discord_connection.py``` script to activate your bot on your Discord server. If you type ">>" followed by a message, the bot will use the message as context for the trained minGPT. For some fun I trained the model to replace any links with a place holder <HTTPS> which I substitute with some very "Basic" current Buzzfeed articles. If you want to have this bot running 24/7 the easiest and most free way is to create an account on AWS (first year is free) and create a tiny Linux server. Here you can clone the repository, upload the trained models into the ```/model``` folder. Next install PM2 (https://pm2.keymetrics.io/docs/usage/quick-start/) and start the bot script with:
  
 ```bash
 pm2 start Discord_connection.p
 ```



