import googletrans
import csv
import httpx

# Replace [PATH] Strings appropriately for your machine.
# For the two temporary files, I named them after the file being translated (i.e. for 'bags' they'd be 'bags_tran1.csv' and 'bags_tran2.csv')
# This only translates one file at a time; you can obviously alter it to do all of them but it'll take forever.
# I've been running a couple/bunch simultaneously in different terminals

is_translated = False
fn1 = '[PATH TO TEMPORARY FILE 1]'
fn2 = '[PATH TO TEMPORARY FILE 2]'

timeout = httpx.Timeout(5)
translator = googletrans.Translator(timeout=timeout)

print('Beginning translation of [file] at [time]...')

# Initial translation
with open('[PATH TO INITIAL, UNTRANSLATED FILE]', encoding='utf-8') as initial:
    with open(fn1, 'w+', encoding='utf-8', newline='') as t1:
        reader = csv.reader(initial)
        writer = csv.writer(t1)
        for line in reader:
            lc = line.copy()
            if translator.detect(lc[1]).lang == 'fr': 
                lc[1] = translator.translate(lc[1], src='fr', dest='en').text
            if translator.detect(lc[2]).lang == 'fr':
                lc[2] = translator.translate(lc[2], src='fr', dest='en').text
            writer.writerow(lc)

# Forced back-and-forth translation until none of the product names are in French
while not is_translated:
    with open(fn1, encoding='utf-8') as f1:
        with open(fn2, 'w+', encoding='utf-8', newline='') as f2:
            reader = csv.reader(f1)
            writer = csv.writer(f2)
            is_translated = True

            for line in reader:
                try:
                    lc = line.copy()
                    if translator.detect(lc[2]).lang == 'fr':
                        is_translated = False
                        lc[2] = translator.translate(lc[2], src='fr', dest='en').text
                    writer.writerow(lc)
                finally: True
                
            # Switches the files
            fholder = f1
            f1 = f2; f2 = fholder

print(f1,'should be fully translated.')
