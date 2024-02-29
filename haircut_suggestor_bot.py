from __future__ import annotations


# Bot Imports
import os
import requests
import math
import telebot
from telebot import types
from dotenv import load_dotenv

# Model Imports
from google.cloud import aiplatform
import vertexai.preview
from tenacity import retry, stop_after_attempt, wait_random_exponential
from google.api_core.exceptions import ResourceExhausted
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import AlreadyExists
import numpy as np
import glob
import os
from typing import Dict, List
import pandas as pd
from logging import error
import re
import textwrap
from typing import Tuple, List
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel
from vertexai.preview.generative_models import GenerativeModel, Image
from PyPDF2 import PdfReader
import PIL
import json
from vertexai.preview.vision_models import ImageGenerationModel

# Load environment variables from .env file. Your BOT_TOKEN should be there.
load_dotenv()

# Model preparation
#Set PROJECT_ID and REGION variables
region = "us-central1"
project_id = "vertex-test-415606"

#Vertex AI Init
vertexai.init(project=project_id, location=region)

# creating a pdf reader object
reader = PdfReader('FaceShapeAndSuggestions.pdf')

# printing number of pages in pdf file
print(len(reader.pages))

# getting a specific page from the pdf file
page = reader.pages[0]

# extracting text from page
text = page.extract_text()
extracted_string = text

generation_model = GenerativeModel("gemini-pro-vision")

image_female_round1 = Image.load_from_file("Faces/round_female1.JPG")
image_female_heart1 = Image.load_from_file("Faces/heart_female1.JPG")
image_female_square1 = Image.load_from_file("Faces/square_female1.JPG")
image_female_oval1 = Image.load_from_file("Faces/oval_female1.JPG")
image_female_oblong1 = Image.load_from_file("Faces/oblong_female1.JPG")
image_female_triangle1 = Image.load_from_file("Faces/triangle_female1.JPG")
image_female_rectangle1 = Image.load_from_file("Faces/rectangle_female1.JPG")
image_female_diamond1 = Image.load_from_file("Faces/diamond_female1.JPG")
image_male_round1 = Image.load_from_file("Faces/round_male1.JPG")
image_male_heart1 = Image.load_from_file("Faces/heart_male1.JPG")
image_male_square1 = Image.load_from_file("Faces/square_male1.JPG")
image_male_oval1 = Image.load_from_file("Faces/oval_male1.JPG")
image_male_oblong1 = Image.load_from_file("Faces/oblong_male1.JPG")
image_male_triangle1 = Image.load_from_file("Faces/triangle_male1.JPG")
image_male_rectangle1 = Image.load_from_file("Faces/rectangle_male1.JPG")
image_male_diamond1 = Image.load_from_file("Faces/diamond_male1.JPG")

def generate_prompt(image, gender):
    context = extracted_string
    if gender=='female':
        question = "From the context and the sample images above, categorize the closest face shape of the face in the image below into one of the following categories: round, diamond, heart, pear, oblong, square, rectangle, triangle. Even if it is not exactly matching entirely, choose the shape only from the categories listed that is close enough. For this identified shape, from the context, suggest top 3 most suited haircuts in 2 categories for females from the context provided. formulate a text prompt that would help generate 3 images for the 3 most suited styles identified in the response. Write haircut names and amount of photos needed explicitly, not just describe the person"
    elif gender=='male':
        question = "From the context and the sample images above, categorize the closest face shape of the face in the image below into one of the following categories: round, diamond, heart, pear, oblong, square, rectangle, triangle. Even if it is not exactly matching entirely, choose the shape only from the categories listed that is close enough. For this identified shape, from the context, suggest top 3 most suited haircuts in 2 categories for males from the context provided. formulate a text prompt that would help generate 3 images for the 3 most suited styles identified in the response. Write haircut names and amount of photos needed explicitly, not just describe the person"
    else:
        question = "From the context and the sample images above, categorize the closest face shape of the face in the image below into one of the following categories: round, diamond, heart, pear, oblong, square, rectangle, triangle. Even if it is not exactly matching entirely, choose the shape only from the categories listed that is close enough. For this identified shape, from the context, suggest top 3 most suited haircuts in 2 categories from the context provided. formulate a text prompt that would help generate 3 images for the 3 most suited styles identified in the response. Write haircut names and amount of photos needed explicitly, not just describe the person"

    prompt = [
    "Here is the context: " + context,
    "Here are the sample images for the context: ",
    "Round shape image of a female identifying individual: ", image_female_round1,
    "Heart shape image of a female identifying individual: ", image_female_heart1,
    "Square shape image of a female identifying individual: ", image_female_square1,
    "Oval shape image of a male or female identifying individual: ", image_female_oval1,
    "Oblong shape image of a female identifying individual: ", image_female_oblong1 ,
    "Triangle shape image of a female identifying individual: ", image_female_triangle1 ,
    "Rectangle shape image of a female identifying individual: ", image_female_rectangle1 ,
    "Diamond shape image of a female identifying individual: ", image_female_diamond1 ,
    "Round shape image of a male identifying individual: ", image_male_round1,
    "Heart shape image of a male identifying individual: ", image_male_heart1,
    "Square shape image of a male identifying individual: ", image_male_square1 ,
    "Oblong shape image of a male identifying individual: ", image_male_oblong1 ,
    "Triangle shape image of a male identifying individual: ", image_male_triangle1 ,
    "Rectangle shape image of a female identifying individual: ", image_male_rectangle1 ,
    "Diamond shape image of a male identifying individual: ", image_male_diamond1 ,

    question,
    image,
    "Return the response in JSON format, you must have 'text_prompt' key there"
    ]
    return prompt

def generate_image(prompt, image_generation_model, amount_of_images=1, save_to='tmp/'):
    regenerate_photo=True
    while regenerate_photo:
        response = image_generation_model.generate_images(
            prompt = prompt,
            number_of_images=amount_of_images,
        )
        for i in range(amount_of_images):
            try:
                response.images[i].save(f'{save_to}{i}.jpg', False)
                regenerate_photo=False
            except:
                regenerate_photo=True
                print('Error occured, regenerating photo')


# Bot preparation
BOT_TOKEN = os.environ.get('BOT_TOKEN')

bot = telebot.TeleBot(BOT_TOKEN)

welcome_prompt = '''Welcome to Haircut Suggestor Bot!'''

male_button = types.InlineKeyboardButton('Male', callback_data='male')
female_button = types.InlineKeyboardButton('Female', callback_data='female')
other_button = types.InlineKeyboardButton('Other', callback_data='other')

keyboard = types.InlineKeyboardMarkup()
keyboard.add(female_button)
keyboard.add(male_button)
keyboard.add(other_button)

gender_statuses = {}

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    """Give a welcome prompt."""
    bot.reply_to(message, welcome_prompt)
    global gender_statuses
    gender_statuses[message.chat.id] = 'female'
    bot.send_message(message.chat.id, text="Choose photo models' appearance", reply_markup=keyboard)

@bot.message_handler(commands=['mode'])
def send_welcome(message):
    """Gender menu"""
    bot.send_message(message.chat.id, text="Choose photo models' appearance", reply_markup=keyboard)

@bot.message_handler(commands=['current_status'])
def send_welcome(message):
    """View current gender status"""
    global gender_statuses
    bot.send_message(message.chat.id, text=f'Currently in {gender_statuses[message.chat.id]} mode')

@bot.message_handler(func=lambda message: message.chat.type=='private', content_types=['photo']) 
def photo_worker(message): 
    file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = 'photos/tmp.png'
    with open(src, "wb") as new_file:
        new_file.write(downloaded_file)
    bot.send_message(message.chat.id, 'Received photo, creating suggestions...')
    image = Image.load_from_file('photos/tmp.png')
    continue_generating_prompts = True
    while continue_generating_prompts:
        prompt = generate_prompt(image, gender_statuses[message.chat.id])
        responses = generation_model.generate_content(prompt
        ,
        generation_config={
            "max_output_tokens": 2048,
            "temperature": 0.15,
            "top_p": 1,
            "top_k": 32
        },
        stream=False,
        )
        response_str = responses.text.replace("```json", "")
        response_str = response_str.replace("```", "")
        json_object = json.loads(response_str)
        if 'text_prompt' in json_object.keys():
            continue_generating_prompts = False
        else:
            print('Error occured, regenerating prompt')
    suggestion = json_object['text_prompt']
    print(suggestion)
    image_generation_model = ImageGenerationModel.from_pretrained("imagegeneration@005")
    generate_image(suggestion, image_generation_model, 3)
    bot.send_photo(message.chat.id, photo=open('tmp/0.jpg', 'rb'))
    bot.send_photo(message.chat.id, photo=open('tmp/1.jpg', 'rb'))
    bot.send_photo(message.chat.id, photo=open('tmp/2.jpg', 'rb'))
    bot.send_message(message.chat.id, 'Send another photo to get more suggestions or use /mode to change model\'s appearance')


@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    global gender_statuses
    if call.data == "female":
        gender_statuses[call.from_user.id] = 'female'
        bot.answer_callback_query(call.id, "Showing female suggestions")
        bot.send_message(call.from_user.id, 'Send photo to get suggestions or use /mode to change model\'s appearance')
    elif call.data == "male":
        gender_statuses[call.from_user.id] = 'male'
        bot.answer_callback_query(call.id, "Showing male suggestions")
        bot.send_message(call.from_user.id, 'Send photo to get suggestions or use /mode to change model\'s appearance')
    elif call.data == "other":
        gender_statuses[call.from_user.id] = 'other'
        bot.answer_callback_query(call.id, "Showing other suggestions")
        bot.send_message(call.from_user.id, 'Send photo to get suggestions or use /mode to change model\'s appearance')




bot.infinity_polling()