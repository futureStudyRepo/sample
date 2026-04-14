from flask import Flask, request, jsonify
from flask_cors import CORS
from googletrans import Translator
import asyncio

app = Flask(__name__)
CORS(app, resources={r"/translate": {"origins": "*"}})

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        # Flask 환경에서 비동기 라이브러리 실행을 위해 내장 asyncio.run 사용
        translator = Translator()
        translation_result = asyncio.run(translator.translate(text, dest='en'))
        
        return jsonify({
            "text": text,
            "translation": translation_result.text
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)