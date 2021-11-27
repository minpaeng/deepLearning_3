# Day_07_01_library.py
import requests
import re


url = 'http://211.251.214.176:8800/index.php?room_no=2'
received = requests.get(url)
print(received)
print(received.text)

results = re.findall(r'<td align="center" style="border: 1px #000000 solid;">([0-9]+) 석</td>', received.text)
print(results)
# ['91', '69', '22', '29', '19', '10']
print('빈 좌석 :', results[-2])

# 퀴즈
# 흥덕 도서관의 노트북 열람실의 빈 좌석 번호를 알려주세요
# empty = re.findall(r'<font style="color:green;font-size:13pt;font-family:Arial"><b>([0-9]+)</b></font>', received.text)
empty = re.findall(r'<.+:green;.+"><b>([0-9]+)</b></font>', received.text)
empty = [int(n) for n in empty]
print(empty)

# 퀴즈
# body 태그 안쪽의 내용을 가져오세요
body = re.findall(r'<body  style="background-repeat: no-repeat;">(.+)</body>',
                  received.text, re.DOTALL)
print(body)
print(len(body))

# 퀴즈
# body 태그 안쪽에 있는 table 태그를 찾으세요
# .+ : 탐욕적(greedy)
# .+? : 비탐욕적(non-greedy)
tables = re.findall(r'<table .+?>(.+?)</table>', body[0], re.DOTALL)
print(tables)
print(len(tables))

