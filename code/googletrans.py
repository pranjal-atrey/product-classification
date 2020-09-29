from googletrans import Translator
import csv
import pymongo

t = Translator()

filename = 'csv_files/' + 'accessories' + '.csv'

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
                columns[columns_keys[i]] = t.translate(line[i], dest='en').text
            else:
                columns[columns_keys[i]] = line[i]
        translated.append(columns)
    
    for t in translated:
        print(t)