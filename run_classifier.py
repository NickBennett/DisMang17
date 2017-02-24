import disaster_classification_module as dc

test_file = open('news_telegraph.txt','r')

for line in test_file:
    try:
        disaster_type, confidence = dc.classify_disaster(line)
        print disaster_type,confidence
    except:
        continue
