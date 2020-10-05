import nltk  # 자연어 처리를 위한 패키지
from nltk import regexp_tokenize

file = open("C:/Users/귤★/Desktop/movie_scripts/500DaysOfSummer.txt", 'r', encoding="utf-8")  # movie_script 디렉토리에서 읽어오기
file_open = file.read()
tokenizer = regexp_tokenize('\s+', gaps=True, pattern="[\w']+")
tokens = regexp_tokenize(file_open, "[\w']+")  # 공백을 기준으로 토큰 분해
en = nltk.Text(tokens)

print(len(en.tokens))   # 총 토큰의 개수 = 단어의 개수
print(en.vocab())   # unique 토큰의 개수

en.plot(17856)  # 총 토큰의 개수를 입력하면 많은 결과 순으로 그래프 보여줌
