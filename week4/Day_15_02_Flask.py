from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import random
import os
# https://tutorial.djangogirls.org/ko/    장고 튜토리얼
# 파이썬 애니웨어: 무료 호스팅 사이트
# ddns : 내 컴퓨터를 서버로 사용할 수 있음(사설 ip)
# 오픈한글 페이지 소스를 sample_2.html에 통째로 복사
# static, templates

app = Flask(__name__)         # __name__: 사용하는 파일 이름


@app.route('/')
def index():
    return 'hello, flask!!'


@app.route('/randoms')
def show_randoms():
    a = make_randoms()
    return str(a)


def make_randoms():
    return [random.randrange(100) for _ in range(10)]


@app.route('/html')
def show_html():
    a = make_randoms()
    return render_template('sample_1.html', numbers=a)


@app.route('/upload')
def upload():
    return render_template('sample_3.html')


# 퀴즈
# 브라우저로부터 이미지를 수신해서 서버에 저장하세요
@app.route('/save', methods=['POST'])
def save_image():
    if request.method == 'POST':
        f = request.files['file']
        # 저장할 경로 + 파일명
        filename = secure_filename(f.filename)
        f.save(os.path.join('static', filename))  # os.path.join: 파일 경로를 합쳐줌
    return render_template('show_img.html', img_name=filename)


if __name__ == '__main__':
    app.run(debug=True)     # debug=True 개발서버. 코드 수정 시 자동 업데이트

