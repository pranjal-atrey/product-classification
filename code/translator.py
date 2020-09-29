from google.cloud import translate_v2 as translate
import csv
import html
import pymongo

translate_client = translate.Client()

filename = 'csv_files/accessories.csv'

with open(filename) as file:
    reader = csv.reader(file)
    
    columns = {}
    for column in reader.__next__():
        columns[column] = None

    columns_keys = list(columns.keys())

    translated = []

    for line in reader:
        for i in range(len(line)):
            if i < 3 and i != 0:
                columns[columns_keys[i]] = html.unescape(translate_client.translate(line[i], target_language='en')['translatedText'])
            else:
                columns[columns_keys[i]] = line[i]
    
    for t in translated:
        print(t)