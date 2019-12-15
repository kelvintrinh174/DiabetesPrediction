from flask import Flask, escape, url_for,request,render_template

app = Flask(__name__)

@app.route('/')
def index():
 return render_template('views/home.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)