import jsonlines
import json
import re

LIST_KEYS = ["ingredients", "categories", "prepSteps", "instructions"]
IMPORTANT_KEYS= ["recipe_title", "directions", "title", "dek", "hed", "description", "dietary_considerations", "course", "type_of_dish"]

text_string = ''


json_files = ['allrecipes-recipes.json', 'bbccouk-recipes.json', 'cookstr-recipes.json', 'epicurious-recipes.json']

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

for f in json_files:
    with open(f, 'r') as json_file:
        for line in json_file.readlines():
            data = json.loads(line)
            for key, value in data.items():
                if key in IMPORTANT_KEYS:
                    text_string += (str(value)+" ")
                elif key in LIST_KEYS:
                    separator = ' '
                    text_string += (separator.join(value)+" ")
            text_string += "eof "
            

text_string = re.sub(r'[^a-zA-Z ]+', '', text_string).lower()
text_string = text_string.replace('grape juice', 'grape_juice')
text_string = text_string.replace('orange juice', 'orange_juice')
text_string = text_string.replace('potato chips', 'potato_chips')
text_string = text_string.replace('cracker', 'crackers')
text_string = text_string.replace('oranges', 'orange')
text_string = text_string.replace('apples', 'apple')

with open('mega_recipe_corpus.txt', 'w') as f:
    f.write(text_string)

