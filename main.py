from flask import Flask, jsonify, request, Response, abort
import EncoderDecoder
import AIclassfier
import tensorflow as tf

app = Flask(__name__)
A = AIclassfier.classifier()
E = EncoderDecoder.Decoder()


@app.route('/', methods=['GET'])
def index():


    return jsonify({
        'author': 'Aayush Phuyal',
        'author_url': 'http://viveksb007.wordpress.com/',
        'base_url': 'zeolearn.com/',
        'endpoints': {
            'Returns URLS of images': '/magazines/photos/{number of photos}',
        }
    })


@app.route('/api/v1.0/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        if request.json:
            print(request.json['image'])
            decoded = E.decode(request.json['image'])
            classification = A.classify(decoded)
            return jsonify({"item":classification})
        return abort(400)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
