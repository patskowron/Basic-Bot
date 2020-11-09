#I originally had an error connecting the discord bot in wondows 10 due to this error:
# [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired 
# Follow the instuctions on this github ticket https://github.com/Rapptz/discord.py/issues/4159 and add the discord.com certificate

#Some open questions
# Do I have to have my training data and pytorch dataset class in order to just load my model?? (really messy to include all of this)

#-------------------------------
# Load Packages
#-------------------------------
import helpers
import discord
import sys
import pickle
import torch
import sys
sys.path.insert(1, '../') #Add project root directory to the path
import mingpt
import yaml

#Load the Discord Token config file
with open("../config.yaml", "r") as fh:
  parsed_yaml_file = yaml.load(fh, Loader=yaml.SafeLoader)

#-------------------------------
# Load the Pytorch Model
#-------------------------------
import math
from torch.utils.data import Dataset

# Pytorch dataset class setup for training data parsing
class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

#Load the training data 
path = "../data/Training_data_input.pickle"
with open(path, 'rb') as f:
    train_dataset_raw = pickle.load(f)
    
#Creat the pytorch model input object    
block_size = 128 
train_dataset = CharDataset(train_dataset_raw, block_size)    

#Set up the model
from mingpt.model import GPT, GPTConfig
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)

# initialize a trainer instance but don't train!
from mingpt.trainer import Trainer, TrainerConfig
tconf = TrainerConfig(max_epochs=1, batch_size=128, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*train_dataset.block_size,
                      num_workers=4)
trainer = Trainer(model, train_dataset, None, tconf)

#Load the GPU trained model into a cpu type pytorch version
path = "../models/mingpt_trained.pickle" 
device = torch.device('cpu')
model.load_state_dict(torch.load(path, map_location=device))

#-------------------------------
# Setup the Discord Bot
#-------------------------------

from mingpt.utils import sample

client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('>>'):

        #Run the Model with the user input message
        context = message.content
        x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
        y = sample(model, x, 200, temperature=1.5, sample=True, top_k=10)[0]
        completion = ''.join([train_dataset.itos[int(i)] for i in y])
        
        #Parse the raw model output
        # 1) Allow the model to finish the current line and repor the next line
        # 2) Replace any intance of <HTTP> with a link to a current headline buzzfeed article
        parsed_model_output=helpers.parse_model_output(completion, 2)
        
        await message.channel.send(parsed_model_output)

client.run(parsed_yaml_file["token"])


