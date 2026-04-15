"""
This script builds a dataset for RAG evaluation. This script will open `data/all_reviews_dataset.csv`, which is the cleaned and merged dataset of Yelp reviews and business information. It will then filter the dataset to include only reviews that are relevant to a specific query (e.g., "breakfast spots in Alton"). From there it will store a set of review_ids that match the query as the "ground truth" labels for evaluation. This dataset can then be used to evaluate the performance of the RAG system in retrieving relevant reviews based on the specified query.
"""

import pandas as pd
import os
import pickle

df = pd.read_csv("data/all_reviews_dataset.csv")
df.head()

ground_truth_path = "data/ground_truth_labels.pkl"
ground_truths = []

# query 1 - breakfast spots in Alton
filter_1 = (df["city"] == "Alton") & (df["categories"].str.contains("Breakfast", case = False))
alton_breakfast_reviews = df[filter_1]
ground_truth_labels_1 = set(alton_breakfast_reviews["review_id"].tolist())
# len(ground_truth_labels_1)
ground_truths.append({
    "query": "breakfast spots in Alton",
    "metadata_filter": {
        "city": "Alton",
        "categories": "Breakfast"
    },
    "review_ids": ground_truth_labels_1})

# query 2 - reviews of Kettle Restaurant
filter_2 = df["business_name"].str.contains("Kettle Restaurant", case = False)
kettle_restaurant_reviews = df[filter_2]
ground_truth_labels_2 = set(kettle_restaurant_reviews["review_id"].tolist())
# len(ground_truth_labels_2)
ground_truths.append({
    "query": "reviews of Kettle Restaurant",
    "metadata_filter": {
        "business_name": "Kettle Restaurant"
    },
    "review_ids": ground_truth_labels_2})

# query 3 - brunch spots in Clearwater
filter_3 = (df["city"] == "Clearwater") & (df["categories"].str.contains("Brunch", case = False))
clearwater_brunch_reviews = df[filter_3]
ground_truth_labels_3 = set(clearwater_brunch_reviews["review_id"].tolist())
# len(ground_truth_labels_3)
ground_truths.append({
    "query": "brunch spots in Clearwater",
    "metadata_filter": {
        "city": "Clearwater",
        "categories": "Brunch"
    },
    "review_ids": ground_truth_labels_3})

# query 4 - 1-star reviews in DE
filter_4 = (df["review_stars"] == 1) & (df["state"] == "DE")
one_star_reviews = df[filter_4]
ground_truth_labels_4 = set(one_star_reviews["review_id"].tolist())
# len(ground_truth_labels_4)
ground_truths.append({
    "query": "1-star reviews in DE",
    "metadata_filter": {
        "review_stars": 1,
        "state": "DE"
    },
    "review_ids": ground_truth_labels_4})

# query 5 - 5-start indian restaurants in AZ
filter_5 = (df["review_stars"] == 5) & (df["state"] == "AZ") & (df["categories"].str.contains("Indian", case = False))
five_star_indian_az_reviews = df[filter_5]
ground_truth_labels_5 = set(five_star_indian_az_reviews["review_id"].tolist())
# len(ground_truth_labels_5)
ground_truths.append({
    "query": "5-star Indian restaurants in AZ",
    "metadata_filter": {
        "review_stars": 5,
        "state": "AZ",
        "categories": "Indian"
    },
    "review_ids": ground_truth_labels_5})

# query 6 - 4-star Mexican restaurants in Nashville
filter_6 = (df["review_stars"] == 4) & (df["city"] == "Nashville") & (df["categories"].str.contains("Mexican", case = False))
mexican_nashville_reviews = df[filter_6]
ground_truth_labels_6 = set(mexican_nashville_reviews["review_id"].tolist())
# len(ground_truth_labels_6)
ground_truths.append({
    "query": "4-star Mexican restaurants in Nashville",
    "metadata_filter": {
        "review_stars": 4,
        "city": "Nashville",
        "categories": "Mexican"
    },
    "review_ids": ground_truth_labels_6})

# query 7 - halal restaurants in Tampa
filter_7 = (df["city"] == "Tampa") & (df["categories"].str.contains("Halal", case = False))
halal_tampa_reviews = df[filter_7]
ground_truth_labels_7 = set(halal_tampa_reviews["review_id"].tolist())
# len(ground_truth_labels_7)
ground_truths.append({
    "query": "halal restaurants in Tampa",
    "metadata_filter": {
        "city": "Tampa",
        "categories": "Halal"
    },
    "review_ids": ground_truth_labels_7})

# query 8 - Jack in the Box reviews in Goleta
filter_8 = (df["business_name"].str.contains("Jack in the Box", case = False)) & (df["city"] == "Goleta")
jack_box_goleta_reviews = df[filter_8]
ground_truth_labels_8 = set(jack_box_goleta_reviews["review_id"].tolist())
# len(ground_truth_labels_8)
ground_truths.append({
    "query": "Jack in the Box reviews in Goleta",
    "metadata_filter": {
        "business_name": "Jack in the Box",
        "city": "Goleta"
    },
    "review_ids": ground_truth_labels_8})

# query 9 - 1-star bars in DE
filter_9 = (df["review_stars"] == 1) & (df["state"] == "DE") & (df["categories"].str.contains("Bar", case = False))
one_star_bars_de_reviews = df[filter_9]
ground_truth_labels_9 = set(one_star_bars_de_reviews["review_id"].tolist())
# len(ground_truth_labels_9)
ground_truths.append({
    "query": "1-star bars in DE",
    "metadata_filter": {
        "review_stars": 1,
        "state": "DE",
        "categories": "Bar"
    },
    "review_ids": ground_truth_labels_9})

# query 10 - hardware stores in PA
filter_10 = (df["state"] == "PA") & (df["categories"].str.contains("Hardware", case = False))
hardware_pa_reviews = df[filter_10]
ground_truth_labels_10 = set(hardware_pa_reviews["review_id"].tolist())
# len(ground_truth_labels_10)
ground_truths.append({
    "query": "hardware stores in PA",
    "metadata_filter": {
        "state": "PA",
        "categories": "Hardware"
    },
    "review_ids": ground_truth_labels_10})

# query 11 - hardware stores in IN
filter_11 = (df["state"] == "IN") & (df["categories"].str.contains("Hardware", case = False))
hardware_in_reviews = df[filter_11]
ground_truth_labels_11 = set(hardware_in_reviews["review_id"].tolist())
# len(ground_truth_labels_11)
ground_truths.append({
    "query": "hardware stores in IN",
    "metadata_filter": {
        "state": "IN",
        "categories": "Hardware"
    },
    "review_ids": ground_truth_labels_11})

# query 12 - planet fitness reviews
filter_12 = df["business_name"].str.contains("Planet Fitness", case = False)
planet_fitness_reviews = df[filter_12]
ground_truth_labels_12 = set(planet_fitness_reviews["review_id"].tolist())
# len(ground_truth_labels_12)
ground_truths.append({
    "query": "Planet Fitness reviews",
    "metadata_filter": {
        "business_name": "Planet Fitness"
    },
    "review_ids": ground_truth_labels_12})

# query 13 - 5-start gyms in FL
filter_13 = (df["review_stars"] == 5) & (df["state"] == "FL") & (df["categories"].str.contains("Gym", case = False))
five_star_gyms_fl_reviews = df[filter_13]
ground_truth_labels_13 = set(five_star_gyms_fl_reviews["review_id"].tolist())
# len(ground_truth_labels_13)
ground_truths.append({
    "query": "5-star gyms in FL",
    "metadata_filter": {
        "review_stars": 5,
        "state": "FL",
        "categories": "Gym"
    },
    "review_ids": ground_truth_labels_13})

# query 14 - rock climbing spots 
filter_14 = df["categories"].str.contains("Rock Climbing", case = False)
rock_climbing_reviews = df[filter_14]
ground_truth_labels_14 = set(rock_climbing_reviews["review_id"].tolist())
# len(ground_truth_labels_14)
ground_truths.append({
    "query": "rock climbing spots",
    "metadata_filter": {
        "categories": "Rock Climbing"
    },
    "review_ids": ground_truth_labels_14})

#query 15 - pet groomers in PA
filter_15 = (df["state"] == "PA") & (df["categories"].str.contains("Pet groomer", case = False))
pet_groomers_pa_reviews = df[filter_15]
ground_truth_labels_15 = set(pet_groomers_pa_reviews["review_id"].tolist())
# len(ground_truth_labels_15)
ground_truths.append({
    "query": "pet groomers in PA",
    "metadata_filter": {
        "state": "PA",
        "categories": "Pet groomer"
    },
    "review_ids": ground_truth_labels_15})

# quey 16 - 5-star hotels in Reno
filter_16 = (df["review_stars"] == 5) & (df["city"] == "Reno") & (df["categories"].str.contains("Hotel", case = False))
five_star_hotels_reno_reviews = df[filter_16]
ground_truth_labels_16 = set(five_star_hotels_reno_reviews["review_id"].tolist())
# len(ground_truth_labels_16)
ground_truths.append({
    "query": "5-star hotels in Reno",
    "metadata_filter": {
        "review_stars": 5,
        "city": "Reno",
        "categories": "Hotel"
    },
    "review_ids": ground_truth_labels_16})

# query 17 - 1-star hotels in Reno
filter_17 = (df["review_stars"] == 1) & (df["city"] == "Reno") & (df["categories"].str.contains("Hotel", case = False))
one_star_hotels_reno_reviews = df[filter_17]
ground_truth_labels_17 = set(one_star_hotels_reno_reviews["review_id"].tolist())
# len(ground_truth_labels_17)
ground_truths.append({
    "query": "1-star hotels in Reno",
    "metadata_filter": {
        "review_stars": 1,
        "city": "Reno",
        "categories": "Hotel"
    },
    "review_ids": ground_truth_labels_17})

# query 18 - caterer services in New Orleans
filter_18 = (df["city"] == "New Orleans") & (df["categories"].str.contains("Caterer", case = False))
caterer_no_reviews = df[filter_18]
ground_truth_labels_18 = set(caterer_no_reviews["review_id"].tolist())
# len(ground_truth_labels_18)
ground_truths.append({
    "query": "caterer services in New Orleans",
    "metadata_filter": {
        "city": "New Orleans",
        "categories": "Caterer"
    },
    "review_ids": ground_truth_labels_18})

# query 19 - 3-star event planning services in DE
filter_19 = (df["review_stars"] == 3) & (df["state"] == "DE") & (df["categories"].str.contains("Event Planning", case = False))
event_planning_de_reviews = df[filter_19]
ground_truth_labels_19 = set(event_planning_de_reviews["review_id"].tolist())
# len(ground_truth_labels_19)
ground_truths.append({
    "query": "3-star event planning services in DE",
    "metadata_filter": {
        "review_stars": 3,
        "state": "DE",
        "categories": "Event Planning"
    },
    "review_ids": ground_truth_labels_19})

# query 20 - meditation centers in FL
filter_20 = (df["state"] == "FL") & (df["categories"].str.contains("Meditation", case = False))
meditation_fl_reviews = df[filter_20]
ground_truth_labels_20 = set(meditation_fl_reviews["review_id"].tolist())
# len(ground_truth_labels_20)
ground_truths.append({
    "query": "meditation centers in FL",
    "metadata_filter": {
        "state": "FL",
        "categories": "Meditation"
    },
    "review_ids": ground_truth_labels_20})

# ============ ADD MORE QUERIES AS NEEDED ============

# query 21 - mediocre breafast spots in New Orleans
filter_21 = (df["review_stars"] < 3) & (df["city"] == "New Orleans") & (df["categories"].str.contains("Breakfast", case = False))
mediocre_breakfast_no_reviews = df[filter_21]
ground_truth_labels_21 = set(mediocre_breakfast_no_reviews["review_id"].tolist())
# len(ground_truth_labels_21)
ground_truths.append({
    "query": "mediocre breakfast spots in New Orleans",
    "metadata_filter": {
        "review_stars": {"op": "lt", "value": 3},
        "city": "New Orleans",
        "categories": "Breakfast"
    },
    "review_ids": ground_truth_labels_21})

# query 22 - Amazing breakfast spots in New Orleans
filter_22 = (df["review_stars"] >= 4) & (df["city"] == "New Orleans") & (df["categories"].str.contains("Breakfast", case = False))
amazing_breakfast_no_reviews = df[filter_22]
ground_truth_labels_22 = set(amazing_breakfast_no_reviews["review_id"].tolist())
# len(ground_truth_labels_22)
ground_truths.append({
    "query": "Amazing breakfast spots in New Orleans",
    "metadata_filter": {
        "review_stars": {"op": "gte", "value": 4},
        "city": "New Orleans",
        "categories": "Breakfast"
    },
    "review_ids": ground_truth_labels_22})

# query 23 sushi places in Tampa
filter_23 = (df["city"] == "Tampa") & (df["categories"].str.contains("Sushi", case = False))
sushi_tampa_reviews = df[filter_23]
ground_truth_labels_23 = set(sushi_tampa_reviews["review_id"].tolist())
# len(ground_truth_labels_23)
ground_truths.append({
    "query": "sushi places in Tampa",
    "metadata_filter": {
        "city": "Tampa",
        "categories": "Sushi"
    },
    "review_ids": ground_truth_labels_23})

# query 24 - average sushi places in Tampa
filter_24 = (df["review_stars"] == 3) & (df["city"] == "Tampa") & (df["categories"].str.contains("Sushi", case = False))
average_sushi_tampa_reviews = df[filter_24]
ground_truth_labels_24 = set(average_sushi_tampa_reviews["review_id"].tolist())
# len(ground_truth_labels_24)
ground_truths.append({
    "query": "average sushi places in Tampa",
    "metadata_filter": {
        "review_stars": 3,
        "city": "Tampa",
        "categories": "Sushi"
    },
    "review_ids": ground_truth_labels_24})

# query 25 - great sushi places in Tampa
filter_25 = (df["review_stars"] >= 4) & (df["city"] == "Tampa") & (df["categories"].str.contains("Sushi", case = False))
great_sushi_tampa_reviews = df[filter_25]
ground_truth_labels_25 = set(great_sushi_tampa_reviews["review_id"].tolist())
# len(ground_truth_labels_25)
ground_truths.append({
    "query": "great sushi places in Tampa",
    "metadata_filter": {
        "review_stars": {"op": "gte", "value": 4},
        "city": "Tampa",
        "categories": "Sushi"
    },
    "review_ids": ground_truth_labels_25})

# =========== NO MORE QUERIES BEYOND THIS POINT ===========

# ground_truths[:3]

with open(ground_truth_path, "wb") as f:
    pickle.dump(ground_truths, f)
print(f"Saved ground truth labels for {len(ground_truths)} queries to {ground_truth_path}")