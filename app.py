# app.py
from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
import re
import os
import json
from flask_session import Session
from dotenv import load_dotenv

# ================== LOAD ENV ==================
load_dotenv()

# ================== CẤU HÌNH ==================
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY không tồn tại trong file .env!")

genai.configure(api_key=api_key)

GENERATION_MODEL = 'gemini-2.5-flash'
EMBEDDING_MODEL = 'models/text-embedding-004'

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret-key-change-this-in-production")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
app.config["PERMANENT_SESSION_LIFETIME"] = 1800
Session(app)

# ================== VIDEO DATABASE ==================
VIDEO_DATABASE = [
    {
        "id": 1,
        "title": "Hướng dẫn An toàn Giao thông cho trẻ em",
        "youtube_id": "EazUZzNl0JI",
        "tags": ["an toàn", "giao thông", "trẻ em", "quy tắc", "cơ bản"],
        "description": "Video hướng dẫn trẻ em tham gia giao thông an toàn"
    },
    {
        "id": 2,
        "title": "Hướng dẫn an toàn giao thông cho bé thiếu nhi | Video 2D Animation",
        "youtube_id": "S4mj5gyx8h0",
        "tags": ["an toàn", "bé", "thiếu nhi", "giao thông", "hoạt hình"],
        "description": "Video hoạt hình 2D hướng dẫn an toàn giao thông cho thiếu nhi"
    },
    {
        "id": 3,
        "title": "Biển báo giao thông cơ bản dành cho học sinh",
        "youtube_id": "oaFXEDkotyY",
        "tags": ["biển báo", "giao thông", "học sinh", "cơ bản", "hiệu lệnh"],
        "description": "Nhận biết các biển báo giao thông thường gặp dành cho học sinh"
    },
    {
        "id": 4,
        "title": "Video tuyên truyền An toàn giao thông",
        "youtube_id": "UW_1nVW492k",
        "tags": ["tuyên truyền", "an toàn", "giao thông", "ý thức", "nâng cao"],
        "description": "Video tuyên truyền nâng cao ý thức an toàn giao thông"
    },
    {
        "id": 5,
        "title": "Giữ Khoảng Cách Với Xe Trước Là Bao Nhiêu Để An Toàn Và Tránh Bị Phạt",
        "youtube_id": "1Fd7D6gW6L8",
        "tags": ["khoảng cách", "xe", "an toàn", "phạt", "luật"],
        "description": "Hướng dẫn giữ khoảng cách an toàn với xe phía trước để tránh bị phạt"
    }
]

def get_video_iframe(youtube_id, title):
    """Tạo iframe cho video YouTube"""
    return f'''
    <div class="video-card">
        <div class="video-wrapper">
            <iframe 
                src="https://www.youtube.com/embed/{youtube_id}" 
                title="{title}"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
            </iframe>
        </div>
        <div class="video-info">
            <h4>{title}</h4>
        </div>
    </div>
    '''

def get_relevant_videos_from_db(keywords, limit=2):
    """Tìm video liên quan từ database"""
    relevant = []
    
    for keyword in keywords:
        for video in VIDEO_DATABASE:
            # Kiểm tra keyword trong tags hoặc title
            if (keyword in video['tags'] or 
                keyword.lower() in video['title'].lower() or
                keyword.lower() in video['description'].lower()):
                if video not in relevant:
                    relevant.append(video)
    
    # Nếu không tìm thấy video liên quan, lấy random
    if not relevant:
        import random
        relevant = random.sample(VIDEO_DATABASE, min(limit, len(VIDEO_DATABASE)))
    
    # Giới hạn số lượng
    return relevant[:limit]

# ================== FORMAT RESPONSE ==================
def format_response(text):
    """Định dạng response từ AI sang HTML"""
    if not text:
        return ""
    
    text = str(text)
    
    # Format markdown -> HTML
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    
    # Format lists
    lines = text.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        if line.strip().startswith(('•', '-', '1.', '2.', '3.', '4.', '5.')):
            if not in_list:
                formatted_lines.append('<ul class="message-list">')
                in_list = True
            content = re.sub(r'^[•\-]\s*|\d+\.\s*', '', line)
            formatted_lines.append(f'<li>{content}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = True
            if line.strip():
                formatted_lines.append(f'<p>{line}</p>')
    
    if in_list:
        formatted_lines.append('</ul>')
    
    result = '\n'.join(formatted_lines)
    
    # Ensure proper closing
    result = result.replace('<p></p>', '')
    return result.strip()

# ================== SESSION MANAGEMENT ==================
def get_client_id():
    """Lấy ID client"""
    ip = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')[:50]
    return f"{ip}_{hash(user_agent)}"

def get_history():
    """Lấy lịch sử chat"""
    client_id = get_client_id()
    key = f"history_{client_id}"
    if key not in session:
        session[key] = []
    return session[key]

def save_history(history):
    """Lưu lịch sử chat"""
    client_id = get_client_id()
    key = f"history_{client_id}"
    session[key] = history[-20:]  # Giới hạn 20 tin nhắn
    session.modified = True

# ================== PROMPT ENGINEERING ==================
def build_prompt(user_message, history):
    """Xây dựng prompt cho AI"""
    # Context từ lịch sử
    context_lines = []
    for msg in history[-5:]:  # Lấy 5 tin nhắn gần nhất
        context_lines.append(msg)
    
    context = "\n".join(context_lines) if context_lines else "Chưa có lịch sử chat."
    
    prompt = f"""Bạn là Trợ Lý AI chuyên về An Toàn Giao Thông tại Trường THCS Long Phước, Đồng Nai.

MỤC TIÊU: Giáo dục và nâng cao nhận thức về an toàn giao thông cho học sinh.

LỊCH SỬ CHAT GẦN ĐÂY:
{context}

CÂU HỎI HIỆN TẠI: {user_message}

HƯỚNG DẪN TRẢ LỜI:
1. Trả lời trực tiếp và đầy đủ câu hỏi của học sinh trước
2. Sử dụng ngôn ngữ tiếng Việt đơn giản, dễ hiểu, thân thiện với học sinh
3. Cung cấp thông tin chính xác, cập nhật về an toàn giao thông, nhưng không trả lời luật pháp chi tiết
4. Sử dụng **in đậm** cho từ khóa quan trọng
5. Sử dụng *in nghiêng* cho lưu ý đặc biệt
6. Dùng • cho danh sách liệt kê
7. Luôn khuyến khích tìm hiểu thêm
8. Kết thúc bằng: "🎬 **Xem video bên dưới để hiểu rõ hơn nhé!**"

TRẢ LỜI:"""
    
    return prompt

# ================== ROUTES ==================
@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Xử lý chat"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'response': 'Lỗi: Không có dữ liệu!'}), 400
            
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'response': format_response('Chào bạn! Hãy hỏi tôi về an toàn giao thông nhé! 🚦')})
        
        # Lấy và cập nhật lịch sử
        history = get_history()
        history.append(f"Học sinh: {user_message}")
        
        # Tạo prompt
        prompt = build_prompt(user_message, history)
        
        # Gọi AI
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.7,
                'top_p': 0.5,
                'top_k': 10,
                'max_output_tokens': 1800,
            }
        )
        
        ai_response = response.text
        
        # Tìm video liên quan
        keywords = re.findall(r'\b[a-zà-ỹ]{3,}\b', user_message.lower())
        relevant_videos = get_relevant_videos_from_db(keywords, limit=2)
        
        # Định dạng response
        response_html = format_response(ai_response)
        
        # Thêm video nếu có
        if relevant_videos:
            video_section = '<div class="video-suggestions">'
            video_section += '<h4>📹 Video liên quan:</h4>'
            for video in relevant_videos:
                video_section += get_video_iframe(video['youtube_id'], video['title'])
            video_section += '</div>'
            response_html += video_section
        
        # Lưu lịch sử
        history.append(f"Trợ lý ATGT: {ai_response}")
        save_history(history)
        
        return jsonify({'response': response_html})
        
    except Exception as e:
        print(f"Lỗi trong /chat: {e}")
        return jsonify({
            'response': format_response(
                'Xin lỗi, hệ thống đang gặp sự cố. '
                'Vui lòng thử lại sau hoặc liên hệ quản trị viên!'
            )
        }), 500

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Xóa lịch sử chat"""
    try:
        client_id = get_client_id()
        key = f"history_{client_id}"
        session.pop(key, None)
        session.modified = True
        return jsonify({'success': True})
    except:
        return jsonify({'success': False}), 500

@app.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Lấy gợi ý câu hỏi"""
    suggestions = [
        "Biển báo cấm là gì?",
        "Luật đội mũ bảo hiểm như thế nào?",
        "Làm sao để qua đường an toàn?",
        "Xử lý khi gặp tai nạn giao thông?",
        "Quy tắc khi đi xe đạp điện?",
        "Phân biệt các loại biển báo?",
        "Kỹ năng lái xe an toàn cho học sinh?"
    ]
    return jsonify({'suggestions': suggestions})

# ================== ERROR HANDLERS ==================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Không tìm thấy trang'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Lỗi máy chủ'}), 500

# ================== RUN ==================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)