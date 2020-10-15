import googletrans, sys, csv, httpx, datetime, time

# Replace [PATH] Strings appropriately for your machine.
# For the two temporary files, I named them after the file being translated (i.e. for 'bags' they'd be 'bags_tran1.csv' and 'bags_tran2.csv')
# This only translates one file at a time; you can obviously alter it to do all of them but it'll take forever.
# I've been running a couple/bunch simultaneously in different terminals

# if there's a sysarg for a file name, use that, otherwise use hard-coded value
if len(sys.argv) <= 1:
    print("Please launch the program in the command line! (ex. \"python iterative_translator.py <csv filename to translate>\"")
    print("Press enter to exit.")
    input()
    sys.exit(0)
else:
    fileToTranslate = sys.argv[1]

# flags and hard-coded paths as well
is_translated = False
fn1 = fileToTranslate[:-4] + "_translated.csv"

# create a translator and a timeout for it
timeout = httpx.Timeout(5)
translator = googletrans.Translator(timeout=timeout)
start_time = datetime.datetime.now()

print('Initialization complete!')
print(f'Beginning translation of {fileToTranslate} at {start_time.strftime("%H:%M:%S")}...')

# now translate until none detected
while not is_translated:
    with open(fileToTranslate, 'r', encoding='utf-8') as f1:
        with open(fn1, 'w+', encoding='utf-8', newline='') as f2:
            reader = csv.reader(f1)
            writer = csv.writer(f2)
            is_translated = True

            for line in reader:
                try:
                    lc = line.copy()
                    if translator.detect(lc[1]).lang == 'fr':
                        is_translated = False
                        lc[1] = translator.translate(lc[1], src='fr', dest='en').text
                    if translator.detect(lc[2]).lang == 'fr':
                        is_translated = False
                        lc[2] = translator.translate(lc[2], src='fr', dest='en').text
                    writer.writerow(lc)
                finally: True
                
            # Switches the files
            fholder = f1
            f1 = f2
            f2 = fholder

end_time = datetime.datetime.now()
print(datetime.datetime.now())
print(f"[SUCCESS IN {datetime.datetime.now() - start_time} SECONDS]")
print(f"{fileToTranslate} has been translated and saved into filename \"{fn1}\".")
print("Press enter to exit.")
input()
sys.exit(0)