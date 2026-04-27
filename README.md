# Yelp Review Sumamrizer

---

In this project, we built an AI application that summarizes the Yelp reviews for various companies.

## Data

This dataset is a subset of Yelp's businesses, reviews, and user data. It was originally assembled for the Yelp Dataset Challenge, which offered students an opportunity to conduct research and/or analysis on Yelp's data. For reference, the data can be found [here](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data?select=yelp_academic_dataset_review.json)

## Our Project

As mentioned previously, the goal of this project is to develop an AI application that summarizes the reviews of various companies. A schematic can be seen below:

![alt text](images/schematic.jpg)

In this project, we built a RAG to house the Yelp reviews. From there, we used a large language model with an appropriate prompt to summarize the reviews. Furthermore, we developed a retrieval mechanism that fetches relevant reviews from our RAG, which are then passed to the language model as context. The language model will then summarize that context and output its summary to the user.

#### To accomplish this project, we:

1. Gathered the Yelp Data
2. Preprocessed the data
3. Built a RAG
4. Designed a prompt that:
   1. queries the database for relevant entries
   2. encorporates these entries into its context
   3. returns a summary of the reviews as well as some general sentiments
5. Built a web app using Gradio
6. Deployed the web app to the cloud

### Web Application

The web app was built using Gradio. It gives the user the ability enter data into several text fields. Given the user's input, the relevant information will be retrieved from the RAG, summarized by a Large Language Model and returned to the user.

here's a video demo of the app:

https://github.com/user-attachments/assets/5ec02350-b459-42f0-90fe-695790745ae0

If the video doesn't render, watch it [here](images/NLP_video_demo.mp4)

## How to Use this Repo

---

### Data Preprocessing

- Download the `yelp_academic_dataset_business.json` and `yelp_academic_dataset_review.json` files from
  [kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data?select=yelp_academic_dataset_review.json)

- Run `utils/data_prep.py` to combine the two JSON files into one DataFrame, and clean the resulting data

### Build the RAG

- Run `build_rag.py`

### Launch the Web Application

- Run `app.py`

### Deploy to Google Cloud Run

- Run `deploy.sh`
