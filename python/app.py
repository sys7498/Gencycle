from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_cors import cross_origin

from detector import Detector

app = Flask(__name__)
CORS(app, origins=['http://localhost:4200'])  # 4200에서의 접근을 허용

@app.route('/api/hello', methods=['GET'])
@cross_origin()
def send_message():
    return "hello"

@app.route('/api/generated_model', methods=['GET'])
@cross_origin()
def send_generated_model():
    # OBJ 파일 읽기
    with open('assets/skull.obj', 'r') as obj_file:
        obj_data = obj_file.read()

    # MTL 파일 읽기
    with open('assets/skull.mtl', 'r') as mtl_file:
        mtl_data = mtl_file.read()

    # 두 파일의 내용을 JSON으로 반환
    return jsonify({
        'obj': obj_data,
        'mtl': mtl_data
    })

if __name__ == '__main__':
    detection_model = Detector(model_path='yolo11n.pt') # 감지 클래스 생성
    app.run(port=5001, debug=True)
