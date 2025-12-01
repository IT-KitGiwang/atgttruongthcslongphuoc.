# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
import google.generativeai as genai
import PyPDF2
import re
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from flask_session import Session
from io import BytesIO
from werkzeug.utils import secure_filename
import pandas as pd
from dotenv import load_dotenv

# ================== LOAD ENV ==================
load_dotenv()

# ================== CẤU HÌNH ==================
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY không tồn tại!")

genai.configure(api_key=api_key)

GENERATION_MODEL = 'gemini-2.5-flash-lite'
EMBEDDING_MODEL = 'text-embedding-004'

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret-key")
app.config["SESSION_TYPE"] = "filesystem"
app.config['UPLOAD_FOLDER'] = './static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
Session(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# ================== KIỂM TRA FILE ==================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ================== RAG DATA ==================
RAG_DATA = {"chunks": [], "embeddings": np.array([]), "is_ready": False}

def extract_text(file_path):
    """Đọc text từ PDF hoặc TXT"""
    text = ""
    try:
        if file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
    except Exception as e:
        print(f"Lỗi đọc file {file_path}: {e}")
    return text

def create_chunks(directory='./static', size=500):
    """Tạo các đoạn (chunk) từ tất cả PDF và TXT trong thư mục"""
    chunks = []
    if not os.path.exists(directory):
        return []
    for f in os.listdir(directory):
        if f.lower().endswith(('.pdf', '.txt')):
            path = os.path.join(directory, f)
            content = extract_text(path)
            for i in range(0, len(content), size):
                chunk = content[i:i + size].strip()
                if chunk:
                    chunks.append(f"[Nguồn: {f}] {chunk}")
    return chunks

def embed_with_retry(texts, model, retries=5):
    """Tạo embedding có retry nếu lỗi API"""
    embeddings = []
    for text in texts:
        for _ in range(retries):
            try:
                res = genai.embed_content(model=model, content=text)
                embeddings.append(res["embedding"])
                break
            except Exception as e:
                print("Lỗi embedding, thử lại sau 2s:", e)
                time.sleep(2)
        else:
            raise e
    return np.array(embeddings)

def init_rag():
    """Khởi tạo hoặc tải lại RAG"""
    global RAG_DATA
    print("🔄 Đang tải lại RAG...")
    RAG_DATA = {"chunks": [], "embeddings": np.array([]), "is_ready": False}
    chunks = create_chunks()
    if not chunks:
        print("⚠️ Không có PDF hợp lệ trong thư mục static/.")
        return
    try:
        embeddings = embed_with_retry(chunks, EMBEDDING_MODEL)
        RAG_DATA.update({"chunks": chunks, "embeddings": embeddings, "is_ready": True})
        print(f"✅ RAG tải xong: {len(chunks)} đoạn từ {len(os.listdir('./static'))} file PDF.")
    except Exception as e:
        print(f"❌ Lỗi RAG: {e}")
        RAG_DATA["is_ready"] = False

# Tải RAG khi khởi động server
init_rag()

# ================== RAG RETRIEVAL ==================
def retrieve_context(query, k=3):
    """Tìm đoạn liên quan nhất từ RAG"""
    if not RAG_DATA["is_ready"]:
        return "Không có tài liệu."
    try:
        q_vec = embed_with_retry([query], EMBEDDING_MODEL)[0].reshape(1, -1)
        sims = cosine_similarity(q_vec, RAG_DATA["embeddings"])[0]
        idxs = np.argsort(sims)[-k:][::-1]
        return "\n\n---\n\n".join(RAG_DATA["chunks"][i] for i in idxs)
    except Exception as e:
        print("Lỗi retrieve_context:", e)
        return "Lỗi tìm kiếm."

# ================== FORMAT RESPONSE ==================
def format_response(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*(?!\s)(.*?)(?<=\S)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'(?m)^\s*\*\s+(.*)', r'• \1', text)
    text = text.replace('\n', '<br>')
    return text

# ================== SESSION HISTORY ==================
def get_ip():
    return request.remote_addr

def get_history():
    key = f"hist_{get_ip()}"
    if key not in session:
        session[key] = []
    return session[key]

def save_history(h):
    key = f"hist_{get_ip()}"
    session[key] = h[-50:]
    session.modified = True

# ================== ROUTES ==================

@app.route('/')
def index():
    status = "Sẵn sàng" if RAG_DATA["is_ready"] else "Chưa có tài liệu"
    return render_template('index.html', rag_status=status)

@app.route('/daknong')
def daknong():
    return render_template('daknong.html')

@app.route('/daknong/<site>')
def site_detail(site):
    lang = request.args.get('lang', 'vi')  # Default to Vietnamese
    sites = {
        'ho-ta-dung': {
            'title': 'Hồ Tà Đùng - Viên ngọc xanh của Tây Nguyên',
            'title_en': 'Ta Dung Lake - The Blue Gem of the Central Highlands',
            'image': 'https://mia.vn/media/uploads/blog-du-lich/ho-ta-dung-10-1689609663.jpg',
            'content_vi': '''
            <p><strong>Hồ Tà Đùng - "Viêm ngọc xanh" của Tây Nguyên</strong></p>

            <p>Nằm ẩn mình giữa những cánh đồng lúa xanh mướt và đồi núi hùng vĩ của huyện Đăk Glong, tỉnh Lâm Đồng, hồ Tà Đùng như một viên ngọc xanh quý giá được thiên nhiên ban tặng. Với diện tích mặt nước lên đến 2.500 ha, hồ không chỉ là nguồn sống mà còn là biểu tượng của sự hài hòa giữa con người và thiên nhiên ở vùng đất Tây Nguyên.</p>

            <h2>Lịch sử hình thành và nguồn gốc địa chất</h2>
            <p>Hồ Tà Đùng được hình thành từ hoạt động phun trào của núi lửa cách đây khoảng 2-3 triệu năm trong kỷ Đệ Tứ. Đây là kết quả của quá trình kiến tạo địa chất phức tạp, khi dung nham nguội đi tạo thành các cấu trúc đá bazan đặc trưng. Theo các nhà địa chất học, hồ nằm trên một miệng núi lửa cổ xưa, nơi dung nham đã bị nước mưa và sông suối xói mòn qua hàng triệu năm, tạo nên lòng chảo tự nhiên rộng lớn.</p>

            <p>Điều đặc biệt thú vị là hồ Tà Đùng thuộc loại hồ "crater lake" - hồ miệng núi lửa, tương tự như hồ Taal ở Philippines hay hồ Rotorua ở New Zealand. Sự hình thành này khiến hồ có độ sâu trung bình 20-30m, với điểm sâu nhất lên đến 45m, tạo nên một hệ sinh thái nước ngọt phong phú.</p>

            <img src="https://ik.imagekit.io/tvlk/blog/2023/03/go-and-share-ho-ta-dung-1.jpg?tr=q-70,c-at_max,w-1000,h-600" alt="Hồ Tà Đùng với cảnh quan tuyệt đẹp" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Hồ Tà Đùng phản chiếu bầu trời xanh và những cánh đồng lúa xanh mướt xung quanh, tạo nên bức tranh thiên nhiên tuyệt đẹp.</p>

            <h2>Hệ sinh thái đa dạng và giá trị bảo tồn</h2>
            <p>Hồ Tà Đùng là một trong những hệ sinh thái nước ngọt quan trọng nhất của Tây Nguyên. Hồ là nơi sinh sống của hơn 200 loài thực vật thủy sinh, 50 loài cá nước ngọt, và hàng chục loài chim di trú. Đặc biệt, hồ là môi trường sống của các loài cá quý hiếm như cá lăng nha, cá mè, và cá trắm cỏ.</p>

            <p>Theo báo cáo của Sở Khoa học và Công nghệ tỉnh Lâm Đồng, hồ Tà Đùng còn là nơi cư trú của ít nhất 15 loài chim nước quý hiếm, bao gồm cò, vạc, và các loài chim di trú từ vùng Siberia. Mỗi năm, vào mùa khô (tháng 12-4), hồ trở thành điểm dừng chân quan trọng cho hàng nghìn con chim di trú từ phương Bắc.</p>

            <img src="https://dulichvietnam.com.vn/kinh-nghiem/wp-content/uploads/2019/10/kinh-nghiem-du-lich-ta-dung-1.jpg" alt="Cảnh quan xung quanh hồ Tà Đùng" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Những cánh đồng lúa xanh mướt bao quanh hồ, tạo nên khung cảnh yên bình và thơ mộng.</p>

            <h2>Ý nghĩa kinh tế - xã hội và văn hóa</h2>
            <p>Hồ Tà Đùng không chỉ là "lá phổi xanh" của vùng Tây Nguyên mà còn đóng vai trò quan trọng trong đời sống kinh tế - xã hội của người dân địa phương. Hồ cung cấp nước tưới cho hơn 3.000 ha đất nông nghiệp, chủ yếu là lúa và cà phê của các huyện Đăk Glong, Krông Nô và Đăk Mil.</p>

            <p>Theo số liệu từ UBND tỉnh Lâm Đồng, hồ Tà Đùng cung cấp nước cho hệ thống thủy lợi phục vụ sản xuất nông nghiệp với sản lượng lương thực đạt 25.000-30.000 tấn/năm. Bên cạnh đó, hồ còn là nguồn thu nhập quan trọng từ du lịch và nuôi trồng thủy sản.</p>

            <p>Về mặt văn hóa, hồ Tà Đùng gắn liền với đời sống của các dân tộc thiểu số Ê Đê, M'Nông và Gia Rai. Người dân địa phương vẫn lưu giữ nhiều truyền thuyết về hồ, trong đó có câu chuyện về "bà mẹ hồ" - một vị thần bảo vệ nguồn nước và mang lại sự sung túc cho cộng đồng.</p>

            <h2>Du lịch và phát triển bền vững</h2>
            <p>Mỗi năm, hồ Tà Đùng thu hút hàng chục nghìn du khách trong và ngoài nước. Các hoạt động du lịch chủ yếu tập trung vào tham quan cảnh quan, câu cá giải trí, và trải nghiệm văn hóa bản địa. Đặc biệt, vào dịp lễ hội Ok Om Bok (lễ hội mừng lúa mới) của người Ê Đê, hồ trở thành điểm nhấn văn hóa quan trọng.</p>

            <p>Tuy nhiên, du lịch cũng đặt ra thách thức lớn cho công tác bảo tồn. Theo báo cáo của Bộ Tài nguyên và Môi trường, hồ Tà Đùng đang đối mặt với nguy cơ ô nhiễm từ hoạt động nông nghiệp và du lịch không kiểm soát. Để giải quyết vấn đề này, tỉnh Lâm Đồng đã triển khai nhiều dự án bảo tồn như:</p>

            <ul>
                <li>Xây dựng hệ thống xử lý nước thải xung quanh hồ</li>
                <li>Phát triển du lịch sinh thái bền vững</li>
                <li>Giám sát và bảo vệ hệ sinh thái tự nhiên</li>
                <li>Tuyên truyền nâng cao ý thức cộng đồng</li>
            </ul>

            <h2>Giá trị khoa học và giáo dục</h2>
            <p>Hồ Tà Đùng là "phòng thí nghiệm sống" quý giá cho các nhà khoa học nghiên cứu về địa chất, sinh thái và biến đổi khí hậu. Nhiều nghiên cứu của Đại học Quốc gia Hà Nội và Đại học Khoa học Tự nhiên TP.HCM đã sử dụng hồ như mô hình nghiên cứu về hệ sinh thái nước ngọt nhiệt đới.</p>

            <p>Đối với học sinh và sinh viên, hồ Tà Đùng là điểm đến lý tưởng để học tập thực tế về địa lý, sinh học và bảo tồn môi trường. Nhiều trường THPT và Đại học trong khu vực đã đưa hồ vào chương trình ngoại khóa, giúp thế hệ trẻ hiểu rõ hơn về giá trị của tài nguyên thiên nhiên.</p>

            <h2>Tương lai và tầm nhìn</h2>
            <p>Với tầm quan trọng chiến lược về kinh tế, văn hóa và sinh thái, hồ Tà Đùng đang được định hướng phát triển thành khu du lịch sinh thái quốc gia. Theo quy hoạch của tỉnh Lâm Đồng đến năm 2030, hồ sẽ trở thành điểm nhấn của "Hành lang du lịch Tây Nguyên" với các sản phẩm du lịch đặc trưng như:</p>

            <ul>
                <li>Du lịch sinh thái và trải nghiệm bản địa</li>
                <li>Nghiên cứu khoa học và giáo dục môi trường</li>
                <li>Phát triển nông nghiệp hữu cơ ven hồ</li>
                <li>Bảo tồn và phát huy văn hóa các dân tộc thiểu số</li>
            </ul>

            <p>Hồ Tà Đùng không chỉ là một hồ nước mà còn là biểu tượng của sự phát triển bền vững, nơi con người và thiên nhiên cùng chung sống hài hòa. Bảo vệ hồ Tà Đùng chính là bảo vệ tương lai cho các thế hệ mai sau.</p>
            ''',
            'content_en': '''
            <p><strong>Ta Dung Lake - The "Blue Emerald" of the Central Highlands</strong></p>

            <p>Nestled amidst the lush green rice fields and majestic mountains of Dak Glong District, Dak Nong Province, Ta Dung Lake is like a precious blue gem bestowed by nature. With a water surface area of up to 2,500 hectares, the lake is not only a source of life but also a symbol of harmony between humans and nature in the Central Highlands region.</p>

            <h2>Geological History and Formation</h2>
            <p>Ta Dung Lake was formed from volcanic eruptions approximately 2-3 million years ago during the Quaternary period. This is the result of complex geological processes, when lava cooled to form characteristic basalt structures. According to geologists, the lake lies on an ancient volcanic crater, where lava was eroded by rainwater and streams over millions of years, creating a vast natural basin.</p>

            <p>What's particularly fascinating is that Ta Dung Lake is a "crater lake" type - a volcanic crater lake, similar to Taal Lake in the Philippines or Rotorua Lake in New Zealand. This formation gives the lake an average depth of 20-30 meters, with the deepest point reaching 45 meters, creating a rich freshwater ecosystem.</p>

            <img src="https://ik.imagekit.io/tvlk/blog/2023/03/go-and-share-ho-ta-dung-1.jpg?tr=q-70,c-at_max,w-1000,h-600" alt="Ta Dung Lake with stunning landscape" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Ta Dung Lake reflects the blue sky and surrounding lush green rice fields, creating a magnificent natural painting.</p>

            <h2>Diverse Ecosystem and Conservation Value</h2>
            <p>Ta Dung Lake is one of the most important freshwater ecosystems in the Central Highlands. The lake is home to more than 200 species of aquatic plants, 50 species of freshwater fish, and dozens of migratory bird species. Particularly, the lake provides habitat for precious fish species such as snakehead fish, climbing perch, and grass carp.</p>

            <p>According to reports from Dak Nong Province's Department of Science and Technology, Ta Dung Lake is also home to at least 15 rare water bird species, including herons, storks, and migratory birds from Siberia. Each year, during the dry season (December-April), the lake becomes an important stopover for thousands of migratory birds from the north.</p>

            <img src="https://dulichvietnam.com.vn/kinh-nghiem/wp-content/uploads/2019/10/kinh-nghiem-du-lich-ta-dung-1.jpg" alt="Surrounding landscape of Ta Dung Lake" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Lush green rice fields surround the lake, creating a peaceful and poetic scenery.</p>

            <h2>Socio-Economic and Cultural Significance</h2>
            <p>Ta Dung Lake is not only the "green lung" of the Central Highlands but also plays an important role in the socio-economic life of local people. The lake provides irrigation water for more than 3,000 hectares of agricultural land, mainly rice and coffee in Dak Glong, Krong No, and Dak Mil districts.</p>

            <p>According to data from Dak Nong Provincial People's Committee, Ta Dung Lake supplies water for the irrigation system serving agricultural production with food output reaching 25,000-30,000 tons per year. In addition, the lake is an important source of income from tourism and aquaculture.</p>

            <p>Culturally, Ta Dung Lake is closely associated with the lives of the Ede, Mnong, and Gia Rai ethnic minorities. Local people still preserve many legends about the lake, including the story of the "lake mother" - a deity who protects the water source and brings prosperity to the community.</p>

            <h2>Tourism and Sustainable Development</h2>
            <p>Each year, Ta Dung Lake attracts tens of thousands of domestic and international tourists. Tourism activities mainly focus on landscape viewing, recreational fishing, and experiencing indigenous culture. Especially during the Ok Om Bok festival (New Rice Festival) of the Ede people, the lake becomes an important cultural highlight.</p>

            <p>However, tourism also poses significant challenges for conservation efforts. According to reports from the Ministry of Natural Resources and Environment, Ta Dung Lake faces pollution risks from uncontrolled agricultural and tourism activities. To address this issue, Dak Nong Province has implemented several conservation projects such as:</p>

            <ul>
                <li>Construction of wastewater treatment systems around the lake</li>
                <li>Development of sustainable ecotourism</li>
                <li>Monitoring and protection of natural ecosystems</li>
                <li>Community awareness raising campaigns</li>
            </ul>

            <h2>Scientific and Educational Value</h2>
            <p>Ta Dung Lake is a valuable "living laboratory" for scientists studying geology, ecology, and climate change. Many studies by Vietnam National University Hanoi and Ho Chi Minh City University of Science have used the lake as a model for studying tropical freshwater ecosystems.</p>

            <p>For students and university students, Ta Dung Lake is an ideal destination for hands-on learning about geography, biology, and environmental conservation. Many high schools and universities in the region have included the lake in their extracurricular programs to help young people better understand the value of natural resources.</p>

            <h2>Future Vision and Outlook</h2>
            <p>With its strategic importance in terms of economy, culture, and ecology, Ta Dung Lake is being oriented to develop into a national ecotourism area. According to Dak Nong Province's planning until 2030, the lake will become a highlight of the "Central Highlands Tourism Corridor" with distinctive tourism products such as:</p>

            <ul>
                <li>Ecotourism and indigenous cultural experiences</li>
                <li>Scientific research and environmental education</li>
                <li>Development of organic agriculture around the lake</li>
                <li>Conservation and promotion of ethnic minority cultures</li>
            </ul>

            <p>Ta Dung Lake is not just a body of water but also a symbol of sustainable development, where humans and nature coexist in harmony. Protecting Ta Dung Lake is protecting the future for future generations.</p>
            '''
        },
        'hang-dong-nui-lua-krong-no': {
            'title': 'Hang động núi lửa Krông Nô - Kỳ quan địa chất độc đáo của Lâm Đồng',
            'title_en': 'Krông Nô Volcanic Cave - Unique Geological Wonder of Lam Dong',
            'image': 'https://vnn-imgs-f.vgcloud.vn/2019/09/27/14/hang-dong-nui-lua-krong-no-duoc-de-cu-cong-vien-di-a-cha-t-toa-n-ca-u.jpg?width=0&s=U3woKIqD4MKbCin9XV0DdA',
            'content_vi': '''
            <p><strong>Hang động núi lửa Krông Nô - Kỳ quan địa chất sống của Tây Nguyên</strong></p>

            <p>Tọa lạc tại xã Nam Đà, huyện Krông Nô, tỉnh Lâm Đồng, hang động núi lửa Krông Nô là một trong những hang động núi lửa lớn nhất và ấn tượng nhất Đông Nam Á. Với chiều dài hơn 2km và cấu trúc địa chất độc đáo, hang động như một bảo tàng sống về lịch sử địa chất của vùng đất Tây Nguyên, thu hút sự chú ý của các nhà khoa học và du khách từ khắp nơi trên thế giới.</p>

            <h2>Lịch sử hình thành và quá trình địa chất</h2>
            <p>Hang động Krông Nô được hình thành từ hoạt động phun trào núi lửa mạnh mẽ cách đây khoảng 2-3 triệu năm trong kỷ Đệ Tứ. Khi dung nham nóng chảy phun trào từ lòng đất, tiếp xúc với không khí lạnh đã tạo ra lớp vỏ ngoài nguội nhanh, hình thành các đường ống dung nham (lava tubes) dài hàng kilomet.</p>

            <p>Theo nghiên cứu của các nhà địa chất học từ Đại học Khoa học Tự nhiên TP.HCM, hang động Krông Nô thuộc hệ thống núi lửa Pleistocen, tương tự như các núi lửa ở Hawaii hay Iceland. Quá trình hình thành trải qua các giai đoạn:</p>

            <ol>
                <li><strong>Giai đoạn phun trào</strong>: Dung nham bazan nóng chảy phun ra khỏi miệng núi lửa</li>
                <li><strong>Giai đoạn nguội đi</strong>: Lớp ngoài của dung nham tiếp xúc không khí, tạo vỏ cứng</li>
                <li><strong>Giai đoạn xói mòn</strong>: Nước mưa và sông suối xói mòn lớp vỏ ngoài, để lại các đường ống dung nham</li>
                <li><strong>Giai đoạn ổn định</strong>: Hình thành hang động với cấu trúc đá bazan đặc trưng</li>
            </ol>

            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSc58-cDyeMggIkuOukwqOE6OTyEGCjZduT-xSdo4UOFBSvw54-R8iQF0nJ0KexeHwd4o4&usqp=CAU" alt="Cấu trúc đá bazan trong hang động" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Các cột đá bazan hình lục giác nổi bật trong hang động, tạo nên cảnh quan kỳ ảo.</p>

            <h2>Cấu trúc địa chất và đặc điểm tự nhiên</h2>
            <p>Hang động Krông Nô có tổng chiều dài hơn 2km, được chia thành nhiều khu vực với đặc điểm địa chất riêng biệt. Phần lớn hang động được cấu tạo từ đá bazan với các cột đá hình lục giác đặc trưng, tương tự như cấu trúc đá ở Giant"s Causeway của Ireland.</p>

            <p>Các đặc điểm nổi bật của hang động bao gồm:</p>

            <ul>
                <li><strong>Cột đá bazan</strong>: Các cột đá hình lục giác, ngũ giác với đường kính từ 30-50cm</li>
                <li><strong>Vòm hang</strong>: Các mái vòm tự nhiên cao từ 5-15m</li>
                <li><strong>Suối ngầm</strong>: Dòng suối chảy qua hang với nước trong vắt</li>
                <li><strong>Hệ thống hang nhánh</strong>: Các đường hầm phụ tạo thành mạng lưới phức tạp</li>
                <li><strong>Đá stalactite</strong>: Các nhũ đá nhỏ hình thành từ khoáng chất</li>
            </ul>

            <img src="https://thanhnien.mediacdn.vn/Uploaded/tracrin/2022_11_30/1-9520.jpg" alt="Suối chảy ngầm trong hang động" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Suối nước trong vắt chảy qua các khối đá bazan, tạo nên âm thanh êm đềm trong hang.</p>

            <h2>Hệ sinh thái và đa dạng sinh học</h2>
            <p>Mặc dù là hang động núi lửa, Krông Nô vẫn duy trì một hệ sinh thái đặc trưng với nhiều loài động thực vật thích nghi với môi trường tối và ẩm. Theo khảo sát của Viện Sinh thái và Tài nguyên Sinh vật, hang động là nơi sinh sống của:</p>

            <ul>
                <li><strong>Động vật</strong>: Dơi, côn trùng, bò sát nhỏ</li>
                <li><strong>Thực vật</strong>: Rêu, địa y, dương xỉ hang động</li>
                <li><strong>Vi sinh vật</strong>: Các loài nấm và vi khuẩn đặc hữu</li>
            </ul>

            <p>Điều đặc biệt là hang động còn là nơi cư trú của loài dơi "người da" quý hiếm, một loài dơi lớn với khả năng phát ra âm thanh định vị phức tạp.</p>

            <h2>Giá trị khoa học và nghiên cứu</h2>
            <p>Hang động Krông Nô là "phòng thí nghiệm tự nhiên" quý giá cho các nhà khoa học nghiên cứu về địa chất, hang động học và cổ sinh vật học. Nhiều nghiên cứu quốc tế đã sử dụng hang động như mô hình nghiên cứu về:</p>

            <ul>
                <li>Quá trình hình thành núi lửa bazan</li>
                <li>Cấu trúc địa chất của vùng Tây Nguyên</li>
                <li>Biến đổi khí hậu qua các kỷ địa chất</li>
                <li>Đa dạng sinh học hang động nhiệt đới</li>
            </ul>

            <p>Năm 2018, một nhóm nghiên cứu từ Đại học Quốc gia Hà Nội đã phát hiện trong hang động Krông Nô những dấu vết hóa thạch của các loài động vật tiền sử, góp phần làm sáng tỏ lịch sử tiến hóa của vùng Đông Nam Á.</p>

            <h2>Ý nghĩa văn hóa và lịch sử</h2>
            <p>Đối với người dân địa phương, đặc biệt là các dân tộc thiểu số Ê Đê và M'Nông, hang động Krông Nô không chỉ là kỳ quan tự nhiên mà còn chứa đựng nhiều giá trị văn hóa tâm linh. Người dân truyền tai nhau câu chuyện về "ông tổ hang động" - một vị thần bảo vệ nguồn nước và mang lại sự sung túc.</p>

            <p>Trong thời kỳ kháng chiến chống Mỹ, hang động còn là nơi ẩn náu quan trọng của các lực lượng cách mạng, chứng kiến nhiều câu chuyện đấu tranh hào hùng của dân tộc.</p>

            <h2>Du lịch và phát triển bền vững</h2>
            <p>Hang động Krông Nô đang được phát triển thành điểm du lịch khoa học với các hoạt động tham quan có hướng dẫn viên chuyên môn. Du khách có thể trải nghiệm:</p>

            <ul>
                <li>Tham quan hang động với hệ thống chiếu sáng chuyên dụng</li>
                <li>Tìm hiểu kiến thức địa chất qua các bảng thuyết minh</li>
                <li>Tham gia các chương trình giáo dục môi trường</li>
                <li>Trải nghiệm văn hóa bản địa xung quanh hang động</li>
            </ul>

            <p>Tuy nhiên, để bảo tồn giá trị khoa học và cảnh quan, tỉnh Lâm Đồng đã áp dụng các biện pháp quản lý nghiêm ngặt:</p>

            <ul>
                <li>Giới hạn số lượng du khách (tối đa 500 người/ngày)</li>
                <li>Cấm các hoạt động gây ô nhiễm âm thanh</li>
                <li>Giám sát liên tục chất lượng không khí và độ ẩm</li>
                <li>Phát triển du lịch sinh thái xung quanh</li>
            </ul>

            <h2>Giá trị giáo dục và ý nghĩa lâu dài</h2>
            <p>Hang động Krông Nô là tài liệu sống động cho chương trình giáo dục địa lý, địa chất và môi trường trong các trường học. Nhiều trường THPT trên cả nước đã đưa hang động vào chương trình ngoại khóa, giúp học sinh hiểu rõ hơn về lịch sử hình thành trái đất và tầm quan trọng của bảo tồn tài nguyên thiên nhiên.</p>

            <p>Với giá trị khoa học và cảnh quan độc đáo, hang động Krông Nô không chỉ là di sản tự nhiên của Việt Nam mà còn góp phần vào kho tàng kiến thức địa chất của nhân loại. Bảo tồn hang động chính là bảo tồn những trang sử sống động của hành tinh chúng ta.</p>
            ''',
            'content_en': '''
            <p><strong>Krông Nô Volcanic Cave - Living Geological Wonder of the Central Highlands</strong></p>

            <p>Located in Nam Da Commune, Krông Nô District, Dak Nong Province, Krông Nô Volcanic Cave is one of the largest and most impressive volcanic caves in Southeast Asia. With a length of over 2km and unique geological structure, the cave is like a living museum of the geological history of the Central Highlands, attracting attention from scientists and tourists from around the world.</p>

            <h2>Formation History and Geological Processes</h2>
            <p>Krông Nô Cave was formed from intense volcanic eruptions approximately 2-3 million years ago during the Quaternary period. When molten lava erupted from the ground, contact with cold air created a rapidly cooling outer layer, forming long lava tubes (lava tubes).</p>

            <p>According to research by geologists from Ho Chi Minh City University of Science, Krông Nô Cave belongs to the Pleistocene volcanic system, similar to volcanoes in Hawaii or Iceland. The formation process went through several stages:</p>

            <ol>
                <li><strong>Eruption stage</strong>: Molten basalt lava erupts from the volcanic crater</li>
                <li><strong>Cooling stage</strong>: The outer layer of lava contacts air, creating a hard crust</li>
                <li><strong>Erosion stage</strong>: Rainwater and streams erode the outer crust, leaving lava tubes</li>
                <li><strong>Stabilization stage</strong>: Formation of cave with characteristic basalt structure</li>
            </ol>

            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSc58-cDyeMggIkuOukwqOE6OTyEGCjZduT-xSdo4UOFBSvw54-R8iQF0nJ0KexeHwd4o4&usqp=CAU" alt="Basalt rock structure in the cave" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Hexagonal basalt columns stand out in the cave, creating a surreal landscape.</p>

            <h2>Geological Structure and Natural Features</h2>
            <p>Krông Nô Cave has a total length of over 2km, divided into several areas with distinct geological characteristics. Most of the cave is made of basalt rock with characteristic hexagonal stone columns, similar to the rock structure at Ireland's Giant's Causeway.</p>

            <p>Prominent features of the cave include:</p>

            <ul>
                <li><strong>Basalt columns</strong>: Hexagonal and pentagonal stone columns with diameters of 30-50cm</li>
                <li><strong>Cave domes</strong>: Natural arches 5-15m high</li>
                <li><strong>Underground streams</strong>: Streams flowing through the cave with crystal-clear water</li>
                <li><strong>Branch cave system</strong>: Subsidiary tunnels creating a complex network</li>
                <li><strong>Stalactite rocks</strong>: Small dripstones formed from minerals</li>
            </ul>

            <img src="https://thanhnien.mediacdn.vn/Uploaded/tracrin/2022_11_30/1-9520.jpg" alt="Underground stream in the cave" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Crystal-clear water flows through basalt rocks, creating a soothing sound in the cave.</p>

            <h2>Ecosystem and Biodiversity</h2>
            <p>Although it is a volcanic cave, Krông Nô still maintains a distinctive ecosystem with many species of flora and fauna adapted to the dark and humid environment. According to surveys by the Institute of Ecology and Biological Resources, the cave is home to:</p>

            <ul>
                <li><strong>Animals</strong>: Bats, insects, small reptiles</li>
                <li><strong>Plants</strong>: Mosses, ferns, cave ferns</li>
                <li><strong>Microorganisms</strong>: Special fungi and bacteria species</li>
            </ul>

            <p>Particularly, the cave is also home to the rare "flying fox" bat species, a large bat with complex echolocation capabilities.</p>

            <h2>Scientific Value and Research</h2>
            <p>Krông Nô Cave is a valuable "natural laboratory" for scientists studying geology, speleology, and paleontology. Many international studies have used the cave as a research model for:</p>

            <ul>
                <li>Basalt volcanic formation processes</li>
                <li>Geological structure of the Central Highlands</li>
                <li>Climate change through geological periods</li>
                <li>Tropical cave biodiversity</li>
            </ul>

            <p>In 2018, a research team from Vietnam National University Hanoi discovered fossil traces of prehistoric animals in Krông Nô Cave, contributing to clarifying the evolutionary history of Southeast Asia.</p>

            <h2>Cultural and Historical Significance</h2>
            <p>For local people, especially the Ede and Mnong ethnic minorities, Krông Nô Cave is not only a natural wonder but also contains many spiritual and cultural values. People pass down stories about the "cave ancestor" - a deity who protects water sources and brings prosperity.</p>

            <p>During the American War resistance period, the cave was also an important hiding place for revolutionary forces, witnessing many heroic struggle stories of the nation.</p>

            <h2>Tourism and Sustainable Development</h2>
            <p>Krông Nô Cave is being developed into a scientific tourism destination with guided tour activities. Visitors can experience:</p>

            <ul>
                <li>Cave tours with specialized lighting systems</li>
                <li>Learning geological knowledge through explanatory panels</li>
                <li>Participating in environmental education programs</li>
                <li>Experiencing indigenous culture around the cave</li>
            </ul>

            <p>However, to preserve scientific value and landscape, Dak Nong Province has applied strict management measures:</p>

            <ul>
                <li>Limiting visitor numbers (maximum 500 people/day)</li>
                <li>Prohibiting noise-polluting activities</li>
                <li>Continuous monitoring of air quality and humidity</li>
                <li>Developing ecotourism around the cave</li>
            </ul>

            <h2>Educational Value and Long-term Significance</h2>
            <p>Krông Nô Cave is vivid material for geography, geology, and environmental education programs in schools. Many high schools across the country have included the cave in extracurricular programs to help students better understand Earth's formation history and the importance of natural resource conservation.</p>

            <p>With its unique scientific and landscape value, Krông Nô Cave is not only Vietnam's natural heritage but also contributes to humanity's geological knowledge treasury. Conserving the cave is conserving the living pages of our planet's history.</p>
            '''
        },
        'di-tich-n-trang-long': {
            'title': 'Di tích lịch sử N\'Trang Lơng - Chứng nhân của cuộc kháng chiến chống Mỹ',
            'title_en': 'N\'Trang Long Historical Site - Witness to the Anti-American Resistance War',
            'image': 'https://cdn2.tuoitre.vn/thumb_w/480/2022/12/20/logo-logo-f5313c1e-76f8-4974-8daf-06b60a04eff7-19921-00002a77f9cced74-16715035922951314190637.jpg',
            'content_vi': '''
            <p>Di tích lịch sử N'Trang Lơng, nằm tại huyện Krông Nô, tỉnh Lâm Đồng, là một trong những điểm nhấn quan trọng ghi dấu cuộc kháng chiến chống Mỹ cứu nước của dân tộc.</p>

            <h2>Lịch sử cách mạng</h2>
            <p>Vào những năm 1960-1970, N'Trang Lơng là căn cứ địa cách mạng quan trọng. Đây là nơi tập kết lực lượng, chuẩn bị vũ khí và tổ chức các hoạt động chống lại quân đội Mỹ và đồng minh.</p>

            <img src="https://dulich.daknong.gov.vn/DataFiles/2024/01/Places/20240118-190620-GxMjhml5.webp" alt="Công trình di tích tại N'Trang Lơng" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Các công trình di tích được bảo tồn nguyên vẹn, kể lại những câu chuyện hào hùng của thời kháng chiến.</p>

            <h2>Các công trình di tích</h2>
            <p>Di tích bao gồm các hầm bí mật, kho vũ khí, nhà ở của cán bộ cách mạng và khu tưởng niệm các anh hùng liệt sĩ. Những công trình này vẫn giữ nguyên hiện trạng lịch sử.</p>

            <img src="https://static.tuoitre.vn/tto/r/2017/06/08/03-1496885681.jpg" alt="Khu tưởng niệm liệt sĩ" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Khu tưởng niệm trang nghiêm với các bia đá khắc tên các anh hùng đã hy sinh vì độc lập dân tộc.</p>

            <h2>Giá trị giáo dục</h2>
            <p>N'Trang Lơng là nơi giáo dục truyền thống cách mạng cho thế hệ trẻ. Các chương trình tham quan, học tập tại di tích giúp thanh niên hiểu rõ giá trị của hòa bình và độc lập.</p>

            <h2>Bảo tồn và phát huy</h2>
            <p>Di tích được bảo tồn nghiêm ngặt và phát huy giá trị thông qua các hoạt động văn hóa, giáo dục. Mỗi năm, di tích đón hàng nghìn lượt khách tham quan.</p>
            ''',
            'content_en': '''
            <p>N'Trang Long Historical Site, located in Krông Nô District, Dak Nong Province, is one of the important landmarks marking the nation's anti-American resistance war for national salvation.</p>

            <h2>Revolutionary History</h2>
            <p>In the 1960s-1970s, N'Trang Long was an important revolutionary base. This was where forces were gathered, weapons prepared, and activities organized against the American army and allies.</p>

            <img src="https://dulich.daknong.gov.vn/DataFiles/2024/01/Places/20240118-190620-GxMjhml5.webp" alt="Historical structures at N'Trang Long" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">The historical structures are preserved intact, recounting the heroic stories of the resistance war.</p>

            <h2>Historical Structures</h2>
            <p>The site includes secret tunnels, weapon depots, revolutionary cadre housing, and memorial areas for heroic martyrs. These structures remain in their original historical condition.</p>

            <img src="https://static.tuoitre.vn/tto/r/2017/06/08/03-1496885681.jpg" alt="Martyrs memorial area" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">The solemn memorial area with stone stelae engraved with the names of heroes who sacrificed for national independence.</p>

            <h2>Educational Value</h2>
            <p>N'Trang Long is a place to educate revolutionary traditions for the young generation. Visit and study programs at the site help youth understand the value of peace and independence.</p>

            <h2>Conservation and Promotion</h2>
            <p>The site is strictly conserved and its value promoted through cultural and educational activities. Each year, the site welcomes thousands of visitors.</p>
            '''
        },
        'khong-gian-cong-chieng': {
            'title': 'Không gian văn hóa Cồng chiêng Tây Nguyên - Di sản phi vật thể của nhân loại',
            'title_en': 'Central Highlands Gong Culture Space - Intangible Cultural Heritage of Humanity',
            'image': 'https://vwu.vn/documents/20182/3932764/27_Jun_2022_015425_GMTPyang.jpg',
            'content_vi': '''
            <p>Không gian văn hóa Cồng chiêng Tây Nguyên là một di sản văn hóa phi vật thể của nhân loại, được UNESCO công nhận năm 2005, đại diện cho nghệ thuật âm nhạc truyền thống của các dân tộc Tây Nguyên.</p>

            <h2>Nguồn gốc và ý nghĩa</h2>
            <p>Cồng chiêng là nhạc cụ truyền thống của các dân tộc Bahnar, Ê Đê, Gia Rai, M'Nông. Đây không chỉ là nhạc cụ mà còn là phương tiện giao tiếp, cầu nối giữa con người với thần linh và thiên nhiên.</p>

            <img src="https://vpdt.vietrantour.com.vn/data/upload/2022/0421/di-san-van-hoa-cong-chieng-tay-nguyen.jpg" alt="Bộ cồng chiêng truyền thống" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Bộ cồng chiêng với các chiêng đồng được sắp xếp theo thứ tự, tạo nên âm thanh du dương.</p>

            <h2>Cấu trúc và cách chơi</h2>
            <p>Một bộ cồng chiêng gồm 12-16 chiếc chiêng đồng, được đánh theo nhịp điệu phức tạp. Mỗi dân tộc có phong cách chơi riêng, thể hiện bản sắc văn hóa độc đáo.</p>

            <img src="https://langvanhoavietnam.vn/Files/image/2019/T11_28_%20VH%20cong%20chieng/1.JPG" alt="Người nghệ nhân chơi cồng chiêng" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Nghệ nhân với bộ cồng chiêng, thể hiện sự tập trung và khéo léo trong từng nhịp đánh.</p>

            <h2>Vai trò trong đời sống</h2>
            <p>Cồng chiêng được sử dụng trong các lễ hội, tang ma, cầu mưa và các nghi lễ quan trọng. Đây là biểu tượng của sự đoàn kết và bản sắc dân tộc.</p>

            <h2>Bảo tồn và phát triển</h2>
            <p>Việc bảo tồn cồng chiêng được chú trọng thông qua các lớp học truyền dạy, lễ hội và chương trình văn hóa. Nhiều nghệ nhân trẻ đang học hỏi để duy trì di sản này.</p>
            ''',
            'content_en': '''
            <p>The Central Highlands Gong Culture Space is an intangible cultural heritage of humanity, recognized by UNESCO in 2005, representing the traditional musical art of the Central Highlands ethnic groups.</p>

            <h2>Origin and Meaning</h2>
            <p>Gongs are traditional musical instruments of the Bahnar, Ede, Gia Rai, and Mnong ethnic groups. These are not just musical instruments but also means of communication, bridges between humans and spirits and nature.</p>

            <img src="https://vpdt.vietrantour.com.vn/data/upload/2022/0421/di-san-van-hoa-cong-chieng-tay-nguyen.jpg" alt="Traditional gong set" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">A set of gongs with bronze gongs arranged in order, creating melodious sounds.</p>

            <h2>Structure and Playing Method</h2>
            <p>A gong set consists of 12-16 bronze gongs, played in complex rhythms. Each ethnic group has its own playing style, expressing unique cultural identity.</p>

            <img src="https://langvanhoavietnam.vn/Files/image/2019/T11_28_%20VH%20cong%20chieng/1.JPG" alt="Artisan playing gongs" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">The artisan with a gong set, showing concentration and skill in each beat.</p>

            <h2>Role in Daily Life</h2>
            <p>Gongs are used in festivals, funerals, rain prayers, and important ceremonies. They are symbols of unity and national identity.</p>

            <h2>Conservation and Development</h2>
            <p>Gong conservation is emphasized through teaching classes, festivals, and cultural programs. Many young artisans are learning to maintain this heritage.</p>
            '''
        },
        'le-hoi-nguoi-ma': {
            'title': 'Lễ hội truyền thống của người Mạ - Nét đẹp văn hóa Tây Nguyên',
            'title_en': 'Traditional Festivals of the Ma People - Cultural Beauty of the Central Highlands',
            'image': 'https://dantocmiennui-media.baotintuc.vn/images/84426cb421b40f0fbef0009243df48a9f2e06a52a677497f77784b603419ad0a5ce9c37935954a227004d4936ad77ebcaaec2b4361bd921c2900d8e534226f4b/IMGL0512.jpg',
            'content_vi': '''
            <p>Lễ hội truyền thống của người Mạ là một phần không thể thiếu trong đời sống văn hóa của dân tộc Mạ tại tỉnh Lâm Đồng, phản ánh tín ngưỡng, phong tục và bản sắc dân tộc.</p>

            <h2>Các loại lễ hội</h2>
            <p>Người Mạ có nhiều lễ hội truyền thống như lễ mừng lúa mới, lễ cưới hỏi, lễ tang ma và các lễ cầu mưa, cầu mùa. Mỗi lễ hội đều có ý nghĩa sâu sắc và được tổ chức trang trọng.</p>

            <img src="https://media.baovanhoa.vn/zoom/600_500/Portals/0/EasyGalleryImages/1/62285/1.JPG" alt="Lễ hội mừng lúa mới" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Người dân trong trang phục truyền thống tham gia lễ hội mừng lúa mới, thể hiện niềm vui sau mùa thu hoạch.</p>

            <h2>Lễ mừng lúa mới</h2>
            <p>Lễ hội quan trọng nhất là lễ mừng lúa mới, diễn ra vào tháng 11-12 âm lịch. Đây là dịp để cảm ơn thần linh, tổ tiên và chia sẻ thành quả lao động.</p>

            <img src="https://media.baovanhoa.vn/zoom/600_500/Portals/0/EasyGalleryImages/1/62285/5-(1).JPG" alt="Các điệu múa truyền thống" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Các điệu múa truyền thống với trang phục lộng lẫy, thể hiện sự vui tươi và đoàn kết.</p>

            <h2>Phong tục và nghi lễ</h2>
            <p>Các lễ hội bao gồm các nghi lễ cúng bái, múa hát, thi đấu truyền thống. Âm nhạc cồng chiêng và khèn lá là không thể thiếu trong các buổi lễ.</p>

            <h2>Ý nghĩa văn hóa</h2>
            <p>Lễ hội không chỉ là dịp vui chơi mà còn là nơi giáo dục truyền thống, gắn kết cộng đồng và bảo tồn bản sắc dân tộc.</p>

            <h2>Bảo tồn và phát triển</h2>
            <p>Với sự phát triển của xã hội, các lễ hội truyền thống đang được bảo tồn và phát triển, trở thành điểm nhấn văn hóa của tỉnh Lâm Đồng.</p>
            ''',
            'content_en': '''
            <p>Traditional festivals of the Ma people are an indispensable part of the cultural life of the Ma ethnic group in Dak Nong Province, reflecting beliefs, customs, and national identity.</p>

            <h2>Types of Festivals</h2>
            <p>The Ma people have many traditional festivals such as the new rice festival, wedding ceremonies, funerals, and rain and harvest prayer ceremonies. Each festival has profound meaning and is organized solemnly.</p>

            <img src="https://media.baovanhoa.vn/zoom/600_500/Portals/0/EasyGalleryImages/1/62285/1.JPG" alt="New rice festival" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">People in traditional costumes participate in the new rice festival, expressing joy after the harvest season.</p>

            <h2>New Rice Festival</h2>
            <p>The most important festival is the new rice festival, held in November-December of the lunar calendar. This is an occasion to thank the spirits, ancestors, and share the fruits of labor.</p>

            <img src="https://media.baovanhoa.vn/zoom/600_500/Portals/0/EasyGalleryImages/1/62285/5-(1).JPG" alt="Traditional dances" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Traditional dances with splendid costumes, expressing joy and unity.</p>

            <h2>Customs and Rituals</h2>
            <p>The festivals include worship rituals, singing and dancing, and traditional competitions. Gong music and leaf flutes are indispensable in the ceremonies.</p>

            <h2>Cultural Significance</h2>
            <p>Festivals are not only entertainment occasions but also places to educate traditions, connect communities, and preserve national identity.</p>

            <h2>Conservation and Development</h2>
            <p>With social development, traditional festivals are being conserved and developed, becoming cultural highlights of Dak Nong Province.</p>
            '''
        },
        'nghe-det-tho-cam': {
            'title': 'Nghề dệt thổ cẩm Ê Đê – M\'Nông - Bảo tồn bản sắc dân tộc',
            'title_en': 'Traditional Brocade Weaving of Ede – Mnong People - Preserving National Identity',
            'image': 'https://dantocmiennui-media.baotintuc.vn/images/57c5aab70c5efc5a98d240302ffc6edb6f987fad6e27a995586a0c17e03923dfda99af3cb3695f77930d76b788e51bee6b3a58603e2551456a75a9c9d13e2b0f/050-1.jpg',
            'content_vi': '''
            <p>Nghề dệt thổ cẩm của người Ê Đê và M'Nông là một nghệ thuật thủ công truyền thống, thể hiện bản sắc văn hóa và sự khéo léo của các dân tộc Tây Nguyên.</p>

            <h2>Lịch sử và truyền thống</h2>
            <p>Nghề dệt thổ cẩm có từ xa xưa, là phương tiện để tạo ra trang phục, khăn piêu, túi xách và các vật dụng gia đình. Mỗi hoa văn đều chứa đựng ý nghĩa văn hóa sâu sắc.</p>

            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSR13S4cHFUY1PkHfOb52lWSUTWttcC2vrZwg&s" alt="Người phụ nữ dệt thổ cẩm" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Người phụ nữ Ê Đê tập trung dệt thổ cẩm trên khung cửi truyền thống.</p>

            <h2>Nguyên liệu và kỹ thuật</h2>
            <p>Thổ cẩm được dệt từ bông, sợi màu tự nhiên. Quy trình phức tạp bao gồm trồng bông, kéo sợi, nhuộm màu và dệt. Mỗi sản phẩm mất hàng tháng để hoàn thành.</p>

            <img src="https://images2.thanhnien.vn/528068263637045248/2023/5/18/edit-det-tho-cam-16843958441061959326868.png" alt="Các hoa văn thổ cẩm" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Các hoa văn thổ cẩm tinh xảo với màu sắc tươi sáng, thể hiện sự sáng tạo của người dệt.</p>

            <h2>Ý nghĩa văn hóa</h2>
            <p>Mỗi hoa văn thổ cẩm kể một câu chuyện, thể hiện lịch sử, tín ngưỡng và cuộc sống của dân tộc. Đây là di sản văn hóa phi vật thể quý báu.</p>

            <h2>Bảo tồn và phát triển</h2>
            <p>Nghề dệt thổ cẩm đang được bảo tồn thông qua các lớp học, hợp tác xã. Sản phẩm thổ cẩm ngày càng được ưa chuộng trên thị trường.</p>
            ''',
            'content_en': '''
            <p>The traditional brocade weaving of the Ede and Mnong people is a traditional handicraft art, expressing the cultural identity and dexterity of the Central Highlands ethnic groups.</p>

            <h2>History and Tradition</h2>
            <p>The brocade weaving profession has existed since ancient times, serving as a means to create clothing, scarves, bags, and household items. Each pattern contains profound cultural meaning.</p>

            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSR13S4cHFUY1PkHfOb52lWSUTWttcC2vrZwg&s" alt="Woman weaving brocade" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">An Ede woman concentrates on weaving brocade on a traditional loom.</p>

            <h2>Materials and Techniques</h2>
            <p>Brocade is woven from cotton and natural colored threads. The complex process includes growing cotton, spinning, dyeing, and weaving. Each product takes months to complete.</p>

            <img src="https://images2.thanhnien.vn/528068263637045248/2023/5/18/edit-det-tho-cam-16843958441061959326868.png" alt="Brocade patterns" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Exquisite brocade patterns with bright colors, expressing the creativity of the weavers.</p>

            <h2>Cultural Significance</h2>
            <p>Each brocade pattern tells a story, expressing the history, beliefs, and life of the ethnic group. This is a precious intangible cultural heritage.</p>

            <h2>Conservation and Development</h2>
            <p>The brocade weaving profession is being conserved through classes and cooperatives. Brocade products are increasingly favored in the market.</p>
            '''
        },
        'van-hoa-nha-dai': {
            'title': 'Văn hóa nhà dài – nhà sàn Tây Nguyên - Kiến trúc độc đáo của các dân tộc thiểu số',
            'title_en': 'Longhouse Culture of the Central Highlands - Unique Architecture of Ethnic Minorities',
            'image': 'https://madagui.com.vn/assets/uploads/2016/04/KI%E1%BA%BEN-TR%C3%9AC-T%C3%82Y-NGUY%C3%8AN-7.jpg',
            'content_vi': '''
            <p>Nhà dài - nhà sàn là biểu tượng kiến trúc của các dân tộc Tây Nguyên, phản ánh triết lý sống hài hòa với thiên nhiên và cộng đồng.</p>

            <h2>Kiến trúc đặc sắc</h2>
            <p>Nhà sàn được xây dựng trên cột gỗ cao 2-3m, mái tranh hoặc lá, dài hàng chục mét. Bên trong chia thành nhiều gian, phục vụ cho nhiều hộ gia đình.</p>

            <img src="https://madagui.com.vn/assets/uploads/2016/04/KI%E1%BA%BEN-TR%C3%9AC-T%C3%82Y-NGUY%C3%8AN-3.jpg" alt="Nhà sàn truyền thống" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Nhà sàn dài với mái tranh cong vút, thể hiện sự khéo léo trong kiến trúc gỗ.</p>

            <h2>Ý nghĩa văn hóa</h2>
            <p>Nhà sàn thể hiện tinh thần đoàn kết, nơi con người sống gần gũi với thiên nhiên. Đây là nơi tổ chức lễ hội, họp làng và bảo tồn văn hóa.</p>

            <img src="https://cly.1cdn.vn/2023/10/22/nha-san-dai-kien-truc-doc-dao-cua-nguoi-e-de-o-tay-nguyen.hinh-1.jpg" alt="Bên trong nhà sàn" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Bên trong nhà sàn với các gian phòng ngăn bằng gỗ, tạo không gian sống ấm cúng.</p>

            <h2>Vật liệu và kỹ thuật</h2>
            <p>Nhà sàn sử dụng gỗ, tre, nứa và lá rừng. Việc xây dựng đòi hỏi kỹ thuật cao và sự hợp tác của cả làng.</p>

            <h2>Bảo tồn và phát triển</h2>
            <p>Văn hóa nhà sàn đang được bảo tồn thông qua các làng văn hóa, du lịch cộng đồng. Đây là điểm nhấn văn hóa của Tây Nguyên.</p>
            ''',
            'content_en': '''
            <p>Longhouse - stilt house is the architectural symbol of the Central Highlands ethnic groups, reflecting the philosophy of living in harmony with nature and community.</p>

            <h2>Unique Architecture</h2>
            <p>Stilt houses are built on wooden pillars 2-3 meters high, with thatched or leaf roofs, spanning dozens of meters. Inside is divided into many compartments, serving multiple households.</p>

            <img src="https://madagui.com.vn/assets/uploads/2016/04/KI%E1%BA%BEN-TR%C3%9AC-T%C3%82Y-NGUY%C3%8AN-3.jpg" alt="Traditional stilt house" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Long stilt house with curved thatched roof, showing skill in wooden architecture.</p>

            <h2>Cultural Significance</h2>
            <p>Stilt houses express the spirit of unity, where people live close to nature. This is where festivals are organized, village meetings held, and culture preserved.</p>

            <img src="https://cly.1cdn.vn/2023/10/22/nha-san-dai-kien-truc-doc-dao-cua-nguoi-e-de-o-tay-nguyen.hinh-1.jpg" alt="Inside the stilt house" style="max-width: 500px; height: auto; display: block; margin: 20px auto;">
            <p style="text-align: center; font-style: italic;">Inside the stilt house with wooden partitioned rooms, creating a cozy living space.</p>

            <h2>Materials and Techniques</h2>
            <p>Stilt houses use wood, bamboo, rattan, and forest leaves. Construction requires high technical skills and cooperation from the entire village.</p>

            <h2>Conservation and Development</h2>
            <p>Longhouse culture is being conserved through cultural villages and community tourism. This is a cultural highlight of the Central Highlands.</p>
            '''
        }
    }

    if site not in sites:
        return redirect(url_for('daknong'))

    data = sites[site].copy()
    data['date'] = '01/12/2025'  # Current date
    data['current_lang'] = lang

    # Select content and title based on language
    if lang == 'en':
        data['title'] = data.get('title_en', data['title'])
        data['content'] = data.get('content_en', data.get('content_vi', ''))
    else:
        data['content'] = data.get('content_vi', data.get('content', ''))

    return render_template('site_detail.html', **data)

@app.route('/chat', methods=['POST'])
def chat():
    msg = request.json.get('message', '').strip()
    if not msg:
        return jsonify({'response': format_response('Hãy hỏi gì đó nhé!')})

    history = get_history()
    history.append(f"Bạn: {msg}")

    context = retrieve_context(msg)
    recent = "\n".join(history[-10:])

    prompt = f"""
    Tài liệu RAG:
    {context}
    Lịch sử nhắn tin để theo dõi và trả lời:
    {recent}

Bạn là Trợ Lý ảo văn hóa song ngữ hỗ trợ văn hóa - di tích lịch sử Tây Nguyên - Lâm Đồng, chuyên dành cho học sinh học tiếng Anh.
Nhiệm vụ của bạn là hỗ trợ người dùng tìm hiểu về văn hóa và di tích lịch sử các tỉnh Tây Nguyên (bao gồm Đăk Nông cũ và Lâm Đồng) bằng cách cung cấp thông tin song ngữ (tiếng Việt và tiếng Anh) theo từng đoạn ngắn.
Bạn trả lời dựa trên tài liệu RAG và kiến thức về văn hóa, lịch sử Tây Nguyên - Lâm Đồng.
Yêu cầu trả lời:
- Cấu trúc phản hồi theo từng đoạn ngắn: Mỗi đoạn gồm 1-2 câu tiếng Việt, theo sau là bản dịch tiếng Anh chính xác và tự nhiên.
- Sử dụng ngôn ngữ đơn giản, phù hợp cho học sinh học tiếng Anh, giải thích từ vựng khó nếu cần.
- Sử dụng <strong>, <em>, • cho định dạng.
- Thân thiện, khuyến khích khám phá văn hóa địa phương và học tiếng Anh.
- Giữ phản hồi ngắn gọn, dưới 500 từ.
- Luôn kèm "Thông tin này chỉ mang tính tham khảo!" in đậm ở cuối.

Câu hỏi: {msg}


"""

    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        res = model.generate_content(prompt)
        ai_text = res.text
        history.append(f"AI: {ai_text}")
        save_history(history)
        return jsonify({'response': format_response(ai_text)})
    except Exception as e:
        print("Lỗi chat:", e)
        return jsonify({'response': format_response('AI đang bận, thử lại sau!')})

# ================== ADMIN ==================
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if (request.form.get('username') == 'tranquocgiang' and
            request.form.get('password') == 'tranquocgiang'):
            session['admin'] = True
            flash('Đăng nhập thành công!', 'success')
            return redirect(url_for('admin_panel'))
        flash('Sai tài khoản/mật khẩu.', 'error')
    return render_template('admin_login.html')

@app.route('/admin/panel')
def admin_panel():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))

    pdfs = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith(('.pdf', '.txt'))]
    histories = []
    for k in session.keys():
        if k.startswith('hist_'):
            ip = k[5:]
            h = session[k]
            if h:
                histories.append({
                    'ip': ip,
                    'messages': len(h),
                    'latest': h[-1],
                    'history': '<br>'.join(h[-10:])
                })

    rag_status = "Sẵn sàng" if RAG_DATA["is_ready"] else "Chưa tải"
    return render_template('admin.html',
                           pdf_files=pdfs,
                           histories=histories,
                           total_users=len(histories),
                           rag_status=rag_status)

@app.route('/admin/upload', methods=['POST'])
def admin_upload():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    file = request.files.get('file')
    if file and allowed_file(file.filename):
        path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(path)
        flash(f'Upload thành công: {file.filename}', 'success')
        init_rag()  # 🔁 Tải lại RAG sau upload
    else:
        flash('Chỉ chấp nhận PDF!', 'error')
    return redirect(url_for('admin_panel'))

@app.route('/admin/delete/<filename>', methods=['POST'])
def admin_delete(filename):
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if os.path.exists(path):
        os.remove(path)
        flash(f'Đã xóa: {filename}', 'success')
        init_rag()  # 🔁 Tải lại RAG sau khi xóa
    return redirect(url_for('admin_panel'))

@app.route('/admin/export_csv')
def export_csv():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    data = []
    for k in session.keys():
        if k.startswith('hist_'):
            ip = k[5:]
            h = session.get(k, [])
            if h:
                data.append({
                    'IP': ip,
                    'Số tin': len(h),
                    'Mới nhất': h[-1],
                    '10 tin cuối': ' | '.join(h[-10:])
                })
    df = pd.DataFrame(data or [{'IP': '-', 'Số tin': 0, 'Mới nhất': '', '10 tin cuối': ''}])
    output = BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='lich_su_chat.csv')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    flash('Đã đăng xuất.', 'success')
    return redirect(url_for('admin_login'))

# ================== RUN ==================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
