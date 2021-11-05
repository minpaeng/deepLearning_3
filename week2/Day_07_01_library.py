import requests
import re

url = 'http://211.251.214.176:8800/index.php?room_no=2'
received = requests.get(url)
# print(received)
# print(received.text)

# 정규표현식에서 +는 한글자 이상을 나타냄
# 괄호로 묶으면 찾은 패턴을 리스트로 가져옴
result = re.findall(r'<td align="center" style="border: 1px #000000 solid;">([0-9]+) 석</td>', received.text)
print('빈 좌석 :', result[-2])

# 퀴즈
# 흥덕 도서관의 노트북 열람실의 빈 좌석 번호를 알려주세요
empty = re.findall(r'<font style="color:green;.+><b>([0-9]+)</b>', received.text)
empty = [int(n) for n in empty]
empty.sort()
print('빈 좌석 번호: ', end='')
[print(n, '', end='') for n in empty]
print()

# 퀴즈
# body 태그 안쪽의 내용을 가져오세요
# re.DOTALL: 전체 문장을 한줄로 만들어줌
body = re.findall(r'<body  style="background-repeat: no-repeat;">(.+)</body>', received.text, re.DOTALL)
# print('<body> 태그: ')
# print(body)

# 퀴즈
# body 태그 안쪽에 있는 table 태그를 찾으세요
# .+ : 탐욕적(greedy 방식), <table.+?> 의 패턴을 찾을 때  body 태그 내부의 수많은 '>' 중 가장 마지막 '>'를 찾음
# .+? : 비탐욕적(non-greedy), <table.+?> 의 패턴을 찾을 때  body 태그 내부의 수많은 '>' 중 가장 먼저 나타난 '>'를 찾음
tables = re.findall(r'<table.+?>(.+?)</table>', body[0], re.DOTALL)
print('<table> 태그: ')
print(tables)
print(len(tables))
