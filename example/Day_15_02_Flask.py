# Day_15_02_Flask.py
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import random
import os

# static, templates

app = Flask(__name__)


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


@app.route('/copy')
def show_copy():
    return render_template('sample_2.html')


@app.route('/upload')
def upload():
    return render_template('sample_3.html')


# 퀴즈
# 브라우저로부터 이미지를 수신해서 서버에 저장하세요
@app.route('/save', methods=['GET', 'POST'])
def save_image():
    if request.method == 'POST':
        f = request.files['file']
        # 저장할 경로 + 파일명
        filename = secure_filename(f.filename)
        f.save(os.path.join('static', filename))
        # return 'uploads 디렉토리 -> 파일 업로드 성공!'
        # return render_template('sample_2.html')
        return render_template('sample_1.html', image_name=filename)


if __name__ == '__main__':
    app.run(debug=True)
