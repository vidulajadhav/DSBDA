#PRACTICAL 7 — Text Analysis & NLP Preprocessing
#PART 1 : Tokenization, POS Tagging, Stopword Removal, Stemming, Lemmatization
#PART 2 : TF-IDF (Term Frequency - Inverse Document Frequency)

#Install (run once if needed) 
# pip install nltk scikit-learn
import nltk
import pandas as pd
import math
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

#Download required NLTK data
nltk.download("punkt",        quiet=True)
nltk.download("punkt_tab",    quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("stopwords",    quiet=True)
nltk.download("wordnet",      quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

#Sample Document 
document = """
Natural Language Processing is a fascinating field of Artificial Intelligence.
Machines are learning to understand human language through various techniques.
Text preprocessing helps improve the quality of data before applying machine learning models.
"""

print("=== Original Document ===")
print(document)

#PART 1 — Document Preprocessing

#Step 1 : Sentence Tokenization 
sent_tokens = sent_tokenize(document)
print("=== 1. Sentence Tokenization ===")
for i, s in enumerate(sent_tokens, 1):
    print(f"  Sentence {i}: {s.strip()}")

#Step 2 : Word Tokenization 
word_tokens = word_tokenize(document)
print("\n=== 2. Word Tokenization ===")
print(word_tokens)

#Step 3 : POS Tagging 
pos_tags = pos_tag(word_tokens)
print("\n=== 3. POS Tagging (Part-of-Speech) ===")
print(pos_tags)

print("\n--- POS Tag Legend ---")
print("  NN=Noun  VBG=Verb(gerund)  JJ=Adjective  RB=Adverb")
print("  DT=Determiner  IN=Preposition  NNP=Proper Noun")

#Step 4 : Stopword Removal 
stop_words    = set(stopwords.words("english"))
filtered_words = [w for w in word_tokens
                  if w.isalpha() and w.lower() not in stop_words]

print("\n=== 4. Stopword Removal ===")
print("Before:", [w for w in word_tokens if w.isalpha()])
print("After :", filtered_words)

#Step 5 : Stemming 
stemmer      = PorterStemmer()
stemmed      = [stemmer.stem(w) for w in filtered_words]

print("\n=== 5. Stemming (PorterStemmer) ===")
print(pd.DataFrame({"Original": filtered_words, "Stemmed": stemmed}).to_string(index=False))

#Step 6 : Lemmatization 
lemmatizer   = WordNetLemmatizer()
lemmatized   = [lemmatizer.lemmatize(w) for w in filtered_words]

print("\n=== 6. Lemmatization (WordNetLemmatizer) ===")
print(pd.DataFrame({"Original": filtered_words, "Lemmatized": lemmatized}).to_string(index=False))

print("\n--- Stemming vs Lemmatization ---")
print(pd.DataFrame({
    "Word"      : filtered_words,
    "Stemmed"   : stemmed,
    "Lemmatized": lemmatized
}).to_string(index=False))

# PART 2 — TF-IDF Representation
#Corpus of 4 documents
corpus = [
    "Natural Language Processing is a field of Artificial Intelligence",
    "Machine learning helps machines learn from data",
    "Text preprocessing is important for natural language processing",
    "Deep learning is a subset of machine learning and artificial intelligence"
]

print("\n\n=== PART 2 — TF-IDF ===")
print("Corpus Documents:")
for i, doc in enumerate(corpus, 1):
    print(f"  D{i}: {doc}")

#TF-IDF using sklearn
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

terms  = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray().round(4),
    columns=terms,
    index=[f"Doc{i+1}" for i in range(len(corpus))]
)

print("\n=== TF-IDF Matrix ===")
print(tfidf_df.T.to_string())   #Transposed for better readability

#Top terms per document
print("\n=== Top 3 Important Terms per Document ===")
for i, doc in enumerate(tfidf_df.index):
    top3 = tfidf_df.loc[doc].sort_values(ascending=False).head(3)
    print(f"  {doc}: {dict(top3.round(4))}")

#Manual TF-IDF explanation for one word 
print("\n=== Manual TF-IDF Calculation Example ===")
word     = "learning"
doc_idx  = 1   # Doc2: "Machine learning helps machines learn from data"
doc_text = corpus[doc_idx].lower().split()
tf       = doc_text.count(word) / len(doc_text)

df_count = sum(1 for d in corpus if word in d.lower())
idf      = math.log(len(corpus) / df_count)
tfidf_manual = tf * idf
print(f"  Word   : '{word}'  in  Doc2")
print(f"  TF     = {doc_text.count(word)} / {len(doc_text)} = {tf:.4f}")
print(f"  IDF    = log({len(corpus)} / {df_count}) = {idf:.4f}")
print(f"  TF-IDF = {tf:.4f} × {idf:.4f} = {tfidf_manual:.4f}")
