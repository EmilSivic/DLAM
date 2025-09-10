import pandas as pd
import re
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize

def clean_ingredient(ingredient):
    ingredient = ingredient.lower()
    ingredient = re.sub(r'\(.*?\)', '', ingredient)  # remove text in brackets
    ingredient = re.sub(r'\d+[\w\s/]*', '', ingredient)  # remove quantities
    ingredient = re.sub(r'[^a-zA-Z\s]', '', ingredient)  # remove punctuation
    ingredient = ingredient.strip()
    return ingredient

def preprocess_recipe_nlg(csv_path, output_path="data/processed_recipes.csv", max_samples=100000):
    df = pd.read_csv(csv_path)

    # Only keep necessary columns
    df = df[['title', 'NER']]
    df = df.dropna()

    # Filter out bad rows
    df = df[df['NER'].apply(lambda x: isinstance(x, str) and len(x) > 2)]

    input_texts = []
    target_texts = []

    for i, row in df.iterrows():
        title = row['title']
        ner = row['NER']

        # Clean input
        input_text = title.lower().strip()

        # Parse NER list: from string to list of ingredients
        try:
            ingredients = eval(ner)
            ingredients = [clean_ingredient(ing) for ing in ingredients if len(ing) > 1]
            ingredients = list(set(ingredients))  # remove duplicates
            if len(ingredients) == 0:
                continue
            input_texts.append(input_text)
            target_texts.append(ingredients)
        except:
            continue

        if len(input_texts) >= max_samples:
            break

    df_clean = pd.DataFrame({'input': input_texts, 'target': target_texts})
    df_clean.to_csv(output_path, index=False)
    print(f"Saved {len(df_clean)} cleaned samples to {output_path}")
    return df_clean