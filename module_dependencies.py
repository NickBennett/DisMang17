import nltk
import random
import pickle
import datetime
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import movie_reviews
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers #Passes some classifiers through this all the time when called
        
    def classify(self, features): #Pass through features so they can be classified
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
        
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = float(choice_votes) / len(votes) #Gives a confidence score of the algorithm results
        return conf
        

start_time = datetime.datetime.now()
print start_time
print("Opening training documents.")
#short_pos = open("short_reviews/positive.txt","r").read()
#short_neg = open("short_reviews/negative.txt","r").read()
earthquake_file = open("Classification Articles/earthquake.txt","r").read()
air_file = open("Classification Articles/air.txt","r").read()
collapse_file = open("Classification Articles/collapse.txt","r").read()
drought_file = open("Classification Articles/drought.txt","r").read()
epidemic_file = open("Classification Articles/epidemic.txt","r").read()
explosion_file = open("Classification Articles/explosion.txt","r").read()
fire_file = open("Classification Articles/fire.txt","r").read()
flood_file = open("Classification Articles/flood.txt","r").read()
fog_file = open("Classification Articles/fog.txt","r").read()
impact_file = open("Classification Articles/impact.txt","r").read()
infestation_file = open("Classification Articles/infestation.txt","r").read()
insect_file = open("Classification Articles/insect.txt","r").read()
landslide_file = open("Classification Articles/landslide.txt","r").read()
poisoning_file = open("Classification Articles/poisoning.txt","r").read()
radiation_file = open("Classification Articles/radiation.txt","r").read()
rail_file = open("Classification Articles/rail.txt","r").read()
road_file = open("Classification Articles/road.txt","r").read()
storm_file = open("Classification Articles/storm.txt","r").read()
water_file = open("Classification Articles/water.txt","r").read()
wildfire_file = open("Classification Articles/wildfire.txt","r").read()

print("Documents successfully loaded.")

documents = []

all_words = []

allowed_word_types = ["J","V","R","N"]

stopword_f = open("/Users/user/anaconda/lib/python2.7/site-packages/newspaper/resources/text/stopwords-en.txt","r").read()
stopword_list = []
for line in stopword_f:
    stopword_list.append(line)

print("Creating all_words list (takes a while)...")

print("Starting earthquake words...")
for p in earthquake_file.split('\n'):
    documents.append( (p, "earthquake") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower())
print("Earthquake words done.")            
 
print("Starting air words...")
for p in air_file.split('\n'):
    documents.append( (p, "air") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower())
print("Air words done.")            
        
print("Starting collapse words...")
for p in collapse_file.split('\n'):
    documents.append( (p, "collapse") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower())  
print("Collapse words done.")            
            
print("Starting drought words...")
for p in drought_file.split('\n'):
    documents.append( (p, "drought") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower()) 
print("Drought words done.")            
            
print("Starting epidemic words...")
for p in epidemic_file.split('\n'):
    documents.append( (p, "epidemic") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower()) 
print("Epidemic words done.")            
            
print("Starting explosion words...")
for p in explosion_file.split('\n'):
    documents.append( (p, "explosion") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower())   
print("Explosion words done.")               
            
print("Starting fire words...")
for p in fire_file.split('\n'):
    documents.append( (p, "fire") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower()) 
print("Fire words done.")  
         
print("Starting flood words...")            
for p in flood_file.split('\n'):
    documents.append( (p, "flood") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower())            
print("Flood words done.")            
            
print("Starting fog words...")
for p in fog_file.split('\n'):
    documents.append( (p, "fog") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower())
print("Fog words done.")            
            
print("Starting impact words...")
for p in impact_file.split('\n'):
    documents.append( (p, "impact") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower())   
print("Impact words done.")            

print("Starting infestation words...")
for p in infestation_file.split('\n'):
    documents.append( (p, "infestation") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower())  
print("Insect words done.")            
            
print("Starting insect words...")            
for p in insect_file.split('\n'):
    documents.append( (p, "insect") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower()) 
print("Insect words done.")            
            
print("Starting landslide words...")
for p in landslide_file.split('\n'):
    documents.append( (p, "landslide") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower())  
print("Landslide words done.")            

print("Starting poison words...")
for p in poisoning_file.split('\n'):
    documents.append( (p, "poisoning") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower())
print("Poisoning words done.")            

print("Starting radiation words...")
for p in radiation_file.split('\n'):
    documents.append( (p, "radiation") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower())
print("Radiation words done.")            

print("Starting rail words...")
for p in rail_file.split('\n'):
    documents.append( (p, "rail") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower())   
print("Rail words done.")            
            
for p in road_file.split('\n'):
    documents.append( (p, "road") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower()) 
print("Road words done.")           
            
print("Starting storm words...")
for p in storm_file.split('\n'):
    documents.append( (p, "storm") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower()) 
print("Storm words done.")            
            
for p in water_file.split('\n'):
    documents.append( (p, "water") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower()) 
print("Water words done.") 
            
print("Starting wildfire words...")
for p in wildfire_file.split('\n'):
    documents.append( (p, "wildfire") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w not in stopword_list:
            all_words.append(w[0].lower())     
print("Wildfire words done.")            
            
list_time = datetime.datetime.now()
print("List created at: " + str(list_time) + ". It took: " + str(list_time - start_time))

print("Pickling documents...")
save_documents = open("pickled_algos/news_articles_train_set.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

print("Pickling 5k feature list...")
word_features = list(all_words.keys())[:5000]
save_word_features = open("pickled_algos/word_features.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        
    return features
    

print("Pickling featuresets...")
before_pickle = datetime.datetime.now()
featuresets = [(find_features(rev), category)for (rev, category) in documents]
save_classifier = open("pickled_algos/featuresets.pickle","wb")
pickle.dump(featuresets, save_classifier)
save_classifier.close() 
after_pickle = datetime.datetime.now()        
random.shuffle(featuresets)
               
training_set = featuresets[:1000]
testing_set = featuresets[1000:]
print("Featuresets pickled. It took: " + str(after_pickle - before_pickle))

print("Training classifiers...")
algo_pickle_start = datetime.datetime.now()
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes: ", (nltk.classify.accuracy(classifier, testing_set)) *100)
print("Pickling Naive Bayes...")
save_classifier = open("pickled_algos/originalnaivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier: ", (nltk.classify.accuracy(MNB_classifier, testing_set)) *100)
print("Pickling MNB_classifier...")
save_classifier = open("pickled_algos/MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) *100)
print("Pickling Bernoulli...")
save_classifier = open("pickled_algos/BernoulliNB_classifier.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

algo_pickle_end = datetime.datetime.now()
print("All algorithms pickled, it took: " + str(algo_pickle_end - algo_pickle_start))

print("Creating voting system at: " + str(datetime.datetime.now()))
#voted_classifier = VoteClassifier(classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier,SGDClassifier_classifier)
voted_classifier = VoteClassifier(classifier,MNB_classifier,BernoulliNB_classifier)
voting_end = datetime.datetime.now()
print("Voting system created at: " + str(datetime.datetime.now()) + ". It took: " + str(voting_end - algo_pickle_end))
print("Voted_classifier accuracy:", (nltk.classify.accuracy(voted_classifier, testing_set)) *100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)
print("Finished! It took: " + str(datetime.datetime.now() - start_time))
