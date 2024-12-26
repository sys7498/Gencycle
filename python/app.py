from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_cors import cross_origin

from detector import Detector
from gpt import Gpt
from PIL import Image
import io
app = Flask(__name__)
CORS(app, origins=['http://localhost:4200'])  # 4200에서의 접근을 허용

@app.route('/api/pc', methods=['GET'])
@cross_origin()
def send_message():
    prompt = request.args.get('prompt', default='car', type=str)
    return jsonify({
        'pc': detection_model.generate_pc(prompt)
    })

@app.route('/api/test_obj', methods=['GET'])
@cross_origin()
def send_test_obj():
    # OBJ 파일 읽기
    with open('assets/grenade_gen.obj', 'r') as obj_file:
        obj_data = obj_file.read()

    # 두 파일의 내용을 JSON으로 반환
    return jsonify({
        'obj': obj_data
    })

@app.route('/api/detect_image', methods=['GET'])
@cross_origin()
def detect_image_start():
    result = detection_model.generate_depthmap('assets/bottle.jpeg')
    return result

@app.route('/api/size', methods=['GET'])
@cross_origin()
def get_size():
    obj = request.args.get('object', default='car', type=str)
    rgb_path = obj + ".jpg"
    bbsize = detection_model.get_size(rgb_path)
    return jsonify({
        'size': {
            'x': bbsize['size'][0],
            'y': bbsize['size'][1],
            'z': bbsize['size'][2]
        }
    })

@app.route('/api/gen_image', methods=['GET'])
@cross_origin()
def generate_image():
    #prompt = request.args.get('prompt', default='car', type=str)
    obj = request.args.get('object', default='car', type=str)
    result = gpt_model.generate_image(obj)
    return result

@app.route('/api/edit_image', methods=['POST'])
@cross_origin()
def edit_image():
    if 'file' not in request.files or 'prompt' not in request.form:
        return jsonify({"error": "File or prompt missing"}), 400
    file = request.files['file']
    prompt = request.form['prompt']  # 문자열 프롬프트
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Blob 데이터를 PIL 이미지로 변환
        image = Image.open(io.BytesIO(file.read()))
        gpt_model.generate_masked_image(image)
        result = gpt_model.edit_image(prompt)
        return result
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/generate_mesh', methods=['GET'])
@cross_origin()
def generate_mesh():
    try:
        gpt_model.generate_mesh("gen")
        return send_file(
            "generated_mesh.glb",
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='generated_mesh.glb'
        )
    except Exception as e:
        return {"error": str(e)}, 500
    
@app.route('/api/generate_original_mesh', methods=['GET'])
@cross_origin()
def generate_original_mesh():
    try:
        obj = request.args.get('object', default='car', type=str)
        rgb_path = obj + "_mesh.jpg"
        gpt_model.generate_mesh(rgb_path)
        return send_file(
            "generated_original_mesh.glb",
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='generated_original_mesh.glb'
        )
    except Exception as e:
        return {"error": str(e)}, 500
    
@app.route('/api/generate_instruction', methods=['GET'])
@cross_origin()
def generate_instruction():
    try:
        return gpt_model.generate_instruction()
    except Exception as e:
        return {"error": str(e)}, 500
    
@app.route('/api/detected_object_info', methods=['GET'])
@cross_origin()
def get_detected_object_info():
    name = request.args.get('name', default='bottle', type=str)
    material = request.args.get('material', default='plastic', type=str)
    result = gpt_model.get_product_description({'name': name, 'material': material})
    return result
    


if __name__ == '__main__':
    detection_model = Detector(model_path='./checkpoints/yolo11x-seg.pt') # 감지 클래스 생성
    gpt_model = Gpt(detection_model.bgremoval_model) # GPT 클래스 생성
    app.run(port=5001, debug=True)
