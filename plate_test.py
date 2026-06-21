import easyocr

reader = easyocr.Reader(['en'])

results = reader.readtext("data/images.png")

for result in results:
    print(result[1])