import jsonlines
import re

LIST_KEYS = ["ingredients", "categories"]
IMPORTANT_KEYS= ["recipe_title", "directions"]

text_string = ''

with jsonlines.open('recipes.jsonlines') as reader:
    for obj in reader:


        recipe_title = obj['recipe_title']
        text_string += (str(recipe_title) + " ")


        categories = obj['categories']
        separator = ' '
        text_string += (separator.join(categories) + " ")

        ingredients = obj['ingredients']
        separator = ' '
        text_string += (separator.join(ingredients) + " ")

        

        directions = obj['directions']
        text_string += (str(directions) + " ")

        text_string += "eof "

        """
        for key, value in obj.items():
            if key in IMPORTANT_KEYS:
                text_string += (str(value)+" ")
            elif key in LIST_KEYS:
                separator = ' '
                text_string += (separator.join(value)+" ")
        """




text_string = re.sub(r'[^a-zA-Z ]+', '', text_string).lower()

with open('recipe_corpus.txt', 'w') as f:
    f.write(text_string)

