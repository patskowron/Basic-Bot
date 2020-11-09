import re
import feedparser
from random import randint

def get_RSS_link(rss_feed):
    '''Given a rss-feed this function outputs a random link'''
    
    # Access the buzzfeed headlines RSS feed and parse with the feedparser library
    parsed_feed = feedparser.parse(rss_feed)
    
    #Find number of headlines and pick a random one
    rand_num=randint(0,len(parsed_feed.entries))
    
    #Return the buzzfeed link
    return parsed_feed.entries[rand_num].link

def parse_model_output(input_str, num_line):
    '''Parse the output of the multiline text generating model'''
    
    #Save each line to a seperate list element
    input_str=input_str.split("\n")
    
    #Throw out the last truncated message (message not ending in \n)
    input_str=input_str[:-1]
    
    #Replace any link 
    input_str = [re.sub("<HTTPS>", get_RSS_link('https://www.buzzfeed.com/index.xml'), x) for x in input_str]
    
    #Return the nth element
    if(len(input_str) > (num_line -1)):
        return input_str[num_line - 1]
    else:
        return input_str[len(input_str)-1]
    