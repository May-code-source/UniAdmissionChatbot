# University Admission Chatbot

## Table of content

- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Future Work](#future-work)


## Project Overview

This project involved building a chatbot using Python and natural language processing to answer common university admission FAQs.

## Data Source
- Custom intent JSON file with tagged questions and responses
- Dialog corpus for additional training example

## Features
- Answers questions on courses, admission requirements, fees etc
- Built using nltk, Keras, and sklearn
- Leverages lemmatization, bag of words model, and deep learning
- Provides relevant links for more information
- Dialogue-based interface for natural conversation
- Streamlit GUI for easy interaction

## Model Architecture
- Sequential model with dense layers and dropout
- Input word vectors
- Output softmax layer with intents
- SGD optimizer and categorical cross-entropy loss

## Usage
Run the Streamlit app and chat with the bot in the GUI.

## Future Work 
- Expand intents dataset from website knowledge base
- Add more conversational features
