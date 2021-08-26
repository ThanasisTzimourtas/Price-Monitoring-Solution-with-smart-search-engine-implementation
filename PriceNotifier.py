import nltk
#import dash_html_components as html
import io
import pandas as pd
import numpy as np
from random import random
from bs4 import BeautifulSoup
import requests
import smtplib
import time
import json
import string
import math
import copy

from collections import Counter
from collections import OrderedDict
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import TreebankWordTokenizer

stemmer = LancasterStemmer()
nltk.download('wordnet')
nltk.download('stopwords')

stopwordseng = nltk.corpus.stopwords.words('english')
wnl = nltk.WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()


'''

The product function checks the product we want and create the target veriable to compare it with the price of the product when ih changes.

'''

def product(product_name):
    productlist = soup.find("section", {"class": "main-content"})

    product = productlist.findAll("div", {"class": "price react-component reviewable"})

    productList = dict()

    for item in product:
        name = item.find("a")['title']
        price = (item.find("a").text).replace('από', '').strip(' €').replace(',', '').replace('.','')
        price = int(price)
        link = "www.skroutz.gr" + item.find("a")['href']

    
        productList[name] = {
            'name': name,
            'price': price,
            'link': link.strip(),
        }
    target = productList[product_name]['price']
    return target
    
def search(soup):
    productlist = soup.find_all('li', class_ = 'cf card with-skus-slider')
    price_list = []
    list_title = []
    for i in range(len(productlist)):
        product = productlist[i].find("a", {"class":"js-sku-link"})
        title = product['title']
        list_title.append(title)

    for i in range(len(productlist)):
        price = productlist[i].find("a", {"class":"js-sku-link sku-link"}).text
        price = price[3:].strip(' €').replace(',', '').replace('.','')
        price = int(price)
        price_list.append(price)

    # Create dictionary 
    dict_data = {'Title': list_title, 'Price': price_list}

    # Create a DataFrame to save name and price 
    csv_data = pd.DataFrame(dict_data)
    csv_data.to_csv('data_list_products.csv', index = False)
    df = pd.read_csv('data_list_products.csv')
    key_names = df['Title']
    list_keys = []
    c = 0
    for key in key_names:
        list_keys.append(key)
        c += 1  

    #print(list_keys)
    return list_keys

    
def checker(soup, target, product_name):

    productlist = soup.find("section", {"class": "main-content"})

    product = productlist.findAll("div", {"class": "price react-component reviewable"})

    productList = dict()

    for item in product:
        name = item.find("a")['title']
        price = (item.find("a").text).replace('από', '').strip(' €').replace(',', '').replace('.','')
        price = int(price)
        link = "www.skroutz.gr" + item.find("a")['href']

    
        productList[name] = {
            'name' : name,
            'price': price,
            'link': link.strip(),
        }

    list_poduct = [productList[product_name]['name'], productList[product_name]['link'], productList[product_name]['price'], target]

    #checks the price and if the current price is lower than the original's, a notification will be sented to your email
    if(productList[product_name]['price'] < target):
            notification(list_poduct)


def notification(list_poduct):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls() # Starts the connection
    server.ehlo()

    # Login to our mail and set the email that we want to sent when the condition is true
    server.login('ENTER YOUR EMAIL','YOUR PASSWORD')
    subject = 'Price Fell Down on ' + str(list_poduct[0])
    body = 'Check the ' + str(list_poduct[0]) + '\nThe product from ' + str(list_poduct[3]) + ' went ' + str(list_poduct[2]) + '\nGo check this on the link below\n' + list_poduct[1]
    # Message variable to parse in the sentmain below
    msg = f"Subject : {subject}\n\n{body}"

    server.sendmail('ENTER YOUR EMAIL', 'ENTER YOUR EMAIL', msg)
    

    # Print a message to check if the email has been sent
    print('HEY EMAIL HAS BEEN SENT!')


    # Quit the server
    server.quit()

def tf_idf(corpus):
    doc_tokens = []
    all_doc_tokens = []

    for doc in corpus:
        doc_tokens +=[sorted(tokenizer.tokenize(doc.lower()))]
    all_doc_tokens = sum(doc_tokens, [])
    lexicon = sorted(set(all_doc_tokens))
    zero_vector = OrderedDict((token, 0) for token in lexicon)
    
    # The tf-idf process
    document_tfidf_vectors = []
    for doc in corpus:
        vec = copy.copy(zero_vector)
        tokens = tokenizer.tokenize(doc.lower())
      
        token_counts = Counter(tokens)
    
        for key, value in token_counts.items():
            docs_containing_key = 0
            for _doc in corpus:
                if key in _doc.lower():
                    docs_containing_key += 1
            
            if docs_containing_key == 0:
                continue
            
            tff = value / len(tokens)
            idf = len(corpus) / docs_containing_key
            vec[key] = tff * idf
        document_tfidf_vectors.append(vec)

    return document_tfidf_vectors


def question_to_corpus(query, documents):
    question_vector = []
    documents_tokens = []
    for doc in documents:
        documents_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
    documents_tokens[0]

    all_documents_tokens = sum(documents_tokens, [])
    lexicon = sorted(set(all_documents_tokens))
    zero_vector = OrderedDict((token, 0) for token in lexicon)
    query_vec = copy.copy(zero_vector)
    query_vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(query.lower())

    token_counts = Counter(tokens)
    for key, value in token_counts.items():
     docs_containing_key = 0
     for _doc in documents:
         if key in _doc.lower():
             docs_containing_key += 1
             
     if docs_containing_key == 0:
         continue
     tff = value / len(tokens)
     idf = len(documents) / docs_containing_key
     query_vec[key] = tff * idf
    question_vector.append(query_vec)

    return question_vector[0]

def consine_sim(vec1, vec2):
    vec1 = [val for val in vec1.values()]
    vec2 = [val for val in vec2.values()]

    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]

    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))

    return dot_prod / (mag_1 * mag_2)


def urlproduct(argument):
    switcher = {
        1: "https://www.skroutz.gr/c/40/kinhta-thlefwna.html?o=kinita&page=",
        2: "https://www.skroutz.gr/c/1705/Smartwatches.html?from=families&page=",
        3: "https://www.skroutz.gr/c/12/television.html?page=",
    }

    return switcher.get(argument, "nothing")

if __name__ == '__main__':

    # Set url and header 
    print("Choose category.")
    print("1: Mobile Phones")
    print("2: Smartwatches")
    print("3: Smart Television")

    
    pageurl = input("Which category you want: ")
    pageurlname = urlproduct(int(pageurl))
    print(pageurlname)
     
    page_found = input("Give the page tha you found it: ")
    url = pageurlname+str(page_found)
    
    print(url)
    headers = {"User-agent" : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'}
    html_content = requests.get(url, headers = headers).text
    soup = BeautifulSoup(html_content, 'html.parser')

    # Give the name that you are looking for 
    product_name =input('Give the name of the product you are looking for: ') #'Samsung Galaxy A20e Dual (32GB) Black' input('Give the name of the product you are looking for: ')
    question = product_name.lower()

    corpus = search(soup)
    tfidf = tf_idf(corpus) 
    q = question_to_corpus(question, corpus)

    similarity = []
    pos = 0
    found = 0
    for i in range(len(tfidf)):
        similarity_item = consine_sim(q, tfidf[i])*100
        similarity.append(similarity_item)
        pos += 1
       
    max_item = max(float(item) for item in similarity)
    found = [i for i, j in enumerate(similarity) if j == max_item]
    found = int(found[0])
    print('The similarity ' + str(max_item) + ' found at ' + str(found))

    final_product_name = corpus[found]
    price_targer = product(final_product_name)
    print(price_targer)
    
    while (True):
        checker(soup,price_targer, final_product_name)
        time.sleep(60 * 60)