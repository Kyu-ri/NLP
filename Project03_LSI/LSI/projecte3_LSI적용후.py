from collections import defaultdict
import math
import sys
from functools import reduce
import gensim
from gensim import models, corpora, similarities

document_filenames = {0: "C:/Users/귤★/Desktop/movie_scripts/500DaysOfSummer.txt",  # 10개의 영화 scripts
                      1: "C:/Users/귤★/Desktop/movie_scripts/BeautyAndTheBeast.txt",
                      2: "C:/Users/귤★/Desktop/movie_scripts/Frozen.txt",
                      3: "C:/Users/귤★/Desktop/movie_scripts/KungFuPanda.txt",
                      4: "C:/Users/귤★/Desktop/movie_scripts/LaLaLand.txt",
                      5: "C:/Users/귤★/Desktop/movie_scripts/Logan.txt",
                      6: "C:/Users/귤★/Desktop/movie_scripts/ToyStory.txt",
                      7: "C:/Users/귤★/Desktop/movie_scripts/Up.txt",
                      8: "C:/Users/귤★/Desktop/movie_scripts/X-Men.txt",
                      9: "C:/Users/귤★/Desktop/movie_scripts/Zootopia.txt"}

N = len(document_filenames)
dictionary = set()

postings = defaultdict(dict)

document_frequency = defaultdict(int)  # scripts id도 포함

length = defaultdict(float)  # Euclidean length == documents vector, 벡터화

characters = " .,!#$%^&*();:\n\t\\\"?!{}[]<>"  # 불필요한 문자 제거하기 위해


def main():
    initialize_terms_and_postings()
    initialize_document_frequencies()
    initialize_lengths()
    while True:
        do_search()  # 계속 query를 돌림


def initialize_terms_and_postings():  # 단어들을 나누고, 불필요한 stopwords 제거
    global dictionary, postings
    for id in document_filenames:
        f = open(document_filenames[id], 'r', encoding="utf-8")
        document = f.read()
        f.close()
        terms = tokenize(document)
        stopwords = {'a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an', 'and',
                     'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can', 'cannot', 'could',
                     'dear', 'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for', 'from', 'get', 'got',
                     'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however', 'i', 'if', 'in',
                     'into', 'is', 'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may', 'me', 'might',
                     'most', 'must', 'my', 'neither', 'no', 'nor', 'not', 'of', 'off', 'often', 'on', 'only', 'or',
                     'other', 'our', 'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since', 'so', 'some',
                     'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to',
                     'too', 'twas', 'us', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while',
                     'who', 'whom', 'why', 'will', 'with', 'would', 'yet', 'you', 'your'}  # [1]의 stopwords
        terms = [tok for tok in terms if
                 len(tok.lower()) > 1 and (tok.lower() not in stopwords)]  # [1]을 이용한 stopwords 제거

        unique_terms = set(terms)  # Term frequency 계산을 위한 unique term 추출
        dictionary = dictionary.union(unique_terms)
        for term in unique_terms:
            postings[term][id] = terms.count(term)  # Term frequency 값


def tokenize(document):  # 불필요한 character문자 제거하여 토큰화
    terms = document.lower().split()
    return [term.strip(characters) for term in terms]


def initialize_document_frequencies():  # DF 측정, documents에서 단어 빈도수 측정
    global document_frequency
    for term in dictionary:
        document_frequency[term] = len(postings[term])


def initialize_lengths():  # script들의 길이를 비교, 길이가 다른 것을 정규화하기 위한 작업
    global length
    for id in document_filenames:
        l = 0
        for term in dictionary:
            l += imp(term, id) ** 2
        length[id] = math.sqrt(l)


def imp(term, id):  # script에서 중요한 단어이며, 없을 경우 0을 리턴
    if id in postings[term]:
        return postings[term][id] * inverse_document_frequency(term)  # TF-IDF로 구함
    else:
        return 0.0


def inverse_document_frequency(term):  # IDF, 단어가 없을 경우 0 리턴
    if term in dictionary:
        return math.log(N / document_frequency[term], 2)
    else:
        return 0.0


def do_search():  # cosine similarity를 이용하여 query를 통해 연관있는 script와 rank(순서)를 보여줌
    query = tokenize(input("Input query >> "))
    if query == []:
        sys.exit()

    relevant_document_ids = intersection(
        [set(postings[term].keys()) for term in query])
    if not relevant_document_ids:
        print("No movie scripts matched all query terms.")
    else:
        scores = sorted([(id, similarity(query, id))
                         for id in relevant_document_ids],
                        key=lambda x: x[1],
                        reverse=True)
        print("Score: Movie name")
        for (id, score) in scores:
            print(str(score) + ": " + document_filenames[id][34:])


def intersection(sets):  # 하나라도 겹치는 단어가 있는지 확인
    return reduce(set.intersection, [s for s in sets[0:20]])


def similarity(query, id):  # query와 script들 사이에 코사인유사도 값을 측정하여 리턴

    similarity = 0.0

    for term in query:
        if term in dictionary:
            similarity += inverse_document_frequency(term) * imp(term, id)

    similarity = similarity / length[id]

    return similarity


if __name__ == "__main__":
    main()
