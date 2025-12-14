from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import google.generativeai as genai

app = Flask(__name__)


try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except Exception:
    st.error("Chưa tìm thấy API Key. Hãy tạo file .streamlit/secrets.toml!")

# Đường dẫn đến file TFLite model
TFLITE_MODEL_PATH = "tea_disease_model.tflite"

class_names = [
    "Anthracnose", 
    "Algal leaf", 
    "Bird eye spot", 
    "brouwn blight", 
    "gray light", 
    "healthy", 
    "red leaf spot", 
    "white spot"
]


try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors() 
    
    # Lấy thông tin input/output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
   
    input_shape = input_details[0]['shape']
    IMG_HEIGHT = input_shape[1]
    IMG_WIDTH = input_shape[2]
    
    print(f"✅ Tải model TFLite thành công. Input yêu cầu: {input_shape}")

except Exception as e:
    print(f"❌ Lỗi tải model TFLite: {e}")
    interpreter = None

# --- 3. HÀM TƯ VẤN GEMINI ---
def get_gemini_advice(disease_name):
    """Gửi yêu cầu đến Gemini và lấy phản hồi dạng text"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        prompt = f"""
            Đóng vai chuyên gia nông nghiệp, tư vấn ngắn gọn (bằng tiếng Việt)
            về bệnh trên cây trà: "{disease_name}":
            1. Nguyên nhân?
            2. Dấu hiệu nhận biết?
            3. Cách trị bệnh (ưu tiên biện pháp an toàn)?
            4. Cách phòng tránh?
            Trình bày định dạng Markdown đẹp, dễ đọc. Nói thẳng vào về các vấn đề trên,
            bỏ qua các câu chào hỏi, viết ngắn gọn khoảng 500 từ.
            No Yapping
            No fluff
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Không thể kết nối với chuyên gia AI: {str(e)}"

@app.route('/predict', methods=['POST'])
def process_image():
    if interpreter is None:
        return jsonify({'error': 'Model chưa được tải'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'Vui lòng upload ảnh'}), 400
    
    try:
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        input_data = np.array(image, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        predictions = output_data[0]
        pred_idx = np.argmax(predictions)
        pred_score = float(np.max(predictions))
        
        if pred_idx < len(class_names):
            result_class = class_names[pred_idx]
        else:
            result_class = "Unknown"

        advice = ""
        advice = get_gemini_advice(result_class)

        # Trả về JSON
        return jsonify({
            'result': result_class,
            'confidence': pred_score,
            'advice': advice
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)