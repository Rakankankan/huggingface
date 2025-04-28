import streamlit as st
import requests
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from streamlit_autorefresh import st_autorefresh
import cv2
import torch
import time
import datetime
from PIL import Image
import numpy as np
import io
import os
from telegram.ext import Application
from telegram import Bot
import asyncio
import logging

# Atur logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Atur cache PyTorch untuk lingkungan deployment
os.environ['TORCH_HOME'] = '/tmp/torch_hub'

# --- CONFIG ---
UBIDOTS_TOKEN = "BBUS-4dkNId6LDOVysK48pdwW8cUGBfAQTK"
DEVICE_LABEL = "hsc345"
VARIABLES = ["mq2", "humidity", "temperature", "lux", "sound"]  # Tambah sound
TELEGRAM_BOT_TOKEN = "7941979379:AAEWGtlb87RYkvht8GzL8Ber29uosKo3e4s"
TELEGRAM_CHAT_ID = "5721363432"
NOTIFICATION_INTERVAL = 300  # 5 menit dalam detik
ALERT_COOLDOWN = 60  # 1 menit cooldown untuk notifikasi langsung
CAMERA_URL = "http://192.168.1.12:81/stream"  # URL kamera ESP32-CAM
GEMINI_API_KEY = "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"
GEMINI_MODEL = "google/gemini-flash-1.5"

# --- STYLE ---
st.markdown("""
    <style>
        .main-title {
            background-color: #001f3f;
            color: white;
            padding: 15px;
            text-align: center;
            font-size: 32px;
            border-radius: 8px;
            margin-bottom: 25px;
        }
        .data-box {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px;
            margin-bottom: 10px;
            font-size: 22px;
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
        }
        .label {
            font-weight: bold;
        }
        .data-value {
            font-size: 24px;
            font-weight: bold;
        }
        .refresh-btn {
            position: absolute;
            top: 30px;
            right: 30px;
            background-color: #0078d4;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }
        .refresh-btn:hover {
            background-color: #005a8d;
        }
        .tab-content {
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
        }
        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 8px;
        }
        .user-message {
            background-color: #e3f2fd;
            padding: 8px 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            max-width: 80%;
            margin-left: auto;
        }
        .assistant-message {
            background-color: #ffffff;
            padding: 8px 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            max-width: 80%;
            margin-right: auto;
        }
    </style>
""", unsafe_allow_html=True)

# --- TELEGRAM FUNCTIONS ---
async def send_telegram_message(message):
    try:
        logger.info("Mengirim pesan ke Telegram...")
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode="Markdown")
        logger.info("Pesan berhasil dikirim ke Telegram")
        st.success("Pesan berhasil dikirim ke Telegram!")
    except Exception as e:
        logger.error(f"Gagal mengirim pesan ke Telegram: {str(e)}")
        st.error(f"Gagal mengirim pesan ke Telegram: {str(e)}")

async def send_telegram_photo(photo, caption):
    try:
        logger.info("Mengirim foto ke Telegram...")
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo, caption=caption, parse_mode="Markdown")
        logger.info("Foto berhasil dikirim ke Telegram")
        st.success("Foto berhasil dikirim ke Telegram!")
    except Exception as e:
        logger.error(f"Gagal mengirim foto ke Telegram: {str(e)}")
        st.error(f"Gagal mengirim foto ke Telegram: {str(e)}")

# --- DATA FETCH ---
def get_ubidots_data(variable_label):
    url = f"https://industrial.api.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}/{variable_label}/values"
    headers = {
        "X-Auth-Token": UBIDOTS_TOKEN,
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get("results", [])
        else:
            logger.error(f"Gagal mengambil data dari Ubidots untuk {variable_label}: {response.status_code}")
            st.error(f"Gagal mengambil data dari Ubidots untuk {variable_label}: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error saat mengambil data Ubidots: {str(e)}")
        st.error(f"Error saat mengambil data Ubidots: {str(e)}")
        return None

# --- DATA FETCH HELPER ---
def fetch_latest_sensor_data():
    """
    Mengambil data terbaru dari Ubidots untuk semua variabel sensor.
    Mengembalikan dictionary dengan nilai dan status untuk setiap sensor.
    """
    data_values = {}
    statuses = {}
    
    for var_name in VARIABLES:
        data = get_ubidots_data(var_name)
        if data and len(data) > 0:
            value = round(data[0]['value'], 2)
            data_values[var_name] = value
        else:
            data_values[var_name] = None
            logger.error(f"Gagal mengambil data untuk {var_name}")
            st.error(f"Gagal mengambil data untuk {var_name}")
    
    # Tentukan status berdasarkan nilai
    mq2_value = data_values.get("mq2")
    lux_value = data_values.get("lux")
    temperature_value = data_values.get("temperature")
    humidity_value = data_values.get("humidity")
    sound_value = data_values.get("sound")
    
    statuses["mq2"] = predict_smoke_status(mq2_value) if mq2_value is not None else "Data asap tidak tersedia"
    statuses["lux"] = evaluate_lux_condition(lux_value, mq2_value) if lux_value is not None else "Data cahaya tidak tersedia"
    statuses["temperature"] = evaluate_temperature_condition(temperature_value) if temperature_value is not None else "Data suhu tidak tersedia"
    statuses["humidity"] = (
        f"Kelembapan {humidity_value}%: {'tinggi' if humidity_value > 70 else 'normal' if humidity_value >= 30 else 'rendah'}"
        if humidity_value is not None else "Data kelembapan tidak tersedia"
    )
    statuses["sound"] = evaluate_sound_condition(sound_value) if sound_value is not None else "Data suara tidak tersedia"
    
    return {
        "values": data_values,
        "statuses": statuses
    }

# --- SIMULASI DATA DAN MODEL ---
@st.cache_data
def generate_mq2_simulation_data(n_samples=100):
    data = []
    for _ in range(n_samples):
        label = random.choices([0, 1], weights=[0.7, 0.3])[0]
        value = random.randint(400, 1000) if label == 1 else random.randint(100, 400)
        data.append((value, label))
    df = pd.DataFrame(data, columns=["mq2_value", "label"])
    return df

@st.cache_resource
def train_mq2_model():
    df = generate_mq2_simulation_data()
    X = df[['mq2_value']]
    y = df['label']
    model = LogisticRegression()
    model.fit(X, y)
    return model

model_iot = train_mq2_model()

# --- AI LOGIC ---
def predict_smoke_status(mq2_value):
    if mq2_value is None:
        return "Data asap tidak tersedia."
    if mq2_value > 800:
        return "Bahaya! Terdeteksi asap rokok!"
    elif mq2_value >= 500:
        return "Mencurigakan: kemungkinan ada asap, tapi belum pasti rokok."
    else:
        return "Semua aman, tidak terdeteksi asap mencurigakan."

def evaluate_lux_condition(lux_value, mq2_value):
    if lux_value is None:
        return "Data cahaya tidak tersedia."
    if lux_value <= 50:
        if "Bahaya" in predict_smoke_status(mq2_value):
            return "Agak mencurigakan: gelap dan ada indikasi asap rokok!"
        elif "Mencurigakan" in predict_smoke_status(mq2_value):
            return "Ruangan gelap dan ada kemungkinan asap, perlu dipantau."
        else:
            return "Ruangan dalam kondisi gelap, tapi tidak ada asap. Masih aman."
    else:
        return "Lampu menyala, kondisi ruangan terang."

def evaluate_temperature_condition(temp_value):
    if temp_value is None:
        return "Data suhu tidak tersedia."
    if temp_value >= 31:
        return "Suhu sangat panas, bisa tidak nyaman, bisa berbahaya!"
    elif temp_value >= 29:
        return "Suhu cukup panas, kurang nyaman."
    elif temp_value <= 28:
        return "Suhu normal dan nyaman."
    else:
        return "Suhu terlalu dingin, bisa tidak nyaman."

def evaluate_sound_condition(sound_value):
    if sound_value is None:
        return "Data suara tidak tersedia."
    if sound_value >= 80:
        return f"Suara {sound_value} dB: Tinggi, kemungkinan aktivitas mencurigakan."
    elif sound_value >= 50:
        return f"Suara {sound_value} dB: Sedang, mungkin ada aktivitas normal."
    else:
        return f"Suara {sound_value} dB: Rendah, ruangan relatif tenang."

def generate_narrative_report(mq2_status, mq2_value, lux_status, lux_value, temp_status, temp_value, humidity_status, humidity_value, sound_status, sound_value):
    intro_templates = [
        "Saat ini, kondisi ruangan terpantau sebagai berikut.",
        "Berikut laporan terbaru dari sistem pengawasan ruangan.",
        "Mari kita lihat bagaimana keadaan ruangan saat ini."
    ]
    smoke_templates = {
        "Bahaya": [
            f"Perhatian! Sensor mendeteksi asap rokok dengan nilai MQ2 {mq2_value}. Segera periksa ruangan!",
            f"Nilai MQ2 mencapai {mq2_value}, menunjukkan adanya asap rokok. Tindakan cepat diperlukan."
        ],
        "Mencurigakan": [
            f"Ada indikasi asap dengan nilai MQ2 {mq2_value}. Meski belum pasti rokok, sebaiknya waspada.",
            f"Sensor MQ2 mencatat {mq2_value}, kemungkinan ada asap. Perlu pemantauan lebih lanjut."
        ],
        "Semua aman": [
            f"Semuanya aman, sensor MQ2 hanya mencatat {mq2_value}. Tidak ada tanda-tanda asap rokok.",
            f"Ruangan bebas asap dengan nilai MQ2 {mq2_value}. Kondisi terkendali."
        ]
    }
    light_templates = {
        "mencurigakan": [
            f"Pencahayaan sangat rendah ({lux_value} lux), ditambah ada indikasi asap. Situasi perlu diperhatikan.",
            f"Dengan lux hanya {lux_value}, ruangan gelap dan ada tanda asap. Sebaiknya periksa."
        ],
        "gelap": [
            f"Ruangan gelap dengan intensitas cahaya {lux_value} lux, tapi tidak ada asap. Aman untuk saat ini.",
            f"Pencahayaan rendah ({lux_value} lux), namun tidak ada masalah asap."
        ],
        "terang": [
            f"Ruangan terang dengan cahaya {lux_value} lux. Semua terlihat jelas dan baik.",
            f"Dengan {lux_value} lux, pencahayaan ruangan sangat mendukung visibilitas."
        ]
    }
    temp_templates = {
        "panas": [
            f"Suhu cukup tinggi, mencapai {temp_value}°C. Mungkin perlu ventilasi tambahan.",
            f"Ruangan terasa panas dengan suhu {temp_value}°C. Perhatikan kenyamanan."
        ],
        "normal": [
            f"Suhu nyaman di {temp_value}°C. Ideal untuk aktivitas sehari-hari.",
            f"Dengan suhu {temp_value}°C, ruangan dalam kondisi menyenangkan."
        ],
        "dingin": [
            f"Suhu agak dingin, hanya {temp_value}°C. Mungkin perlu penghangat.",
            f"Ruangan terasa sejuk di {temp_value}°C. Sesuaikan jika diperlukan."
        ]
    }
    humidity_templates = {
        "tinggi": [
            f"Kelembapan tinggi di {humidity_value}%. Pertimbangkan untuk menggunakan dehumidifier.",
            f"Dengan kelembapan {humidity_value}%, ruangan terasa agak lembap."
        ],
        "normal": [
            f"Kelembapan normal di {humidity_value}%. Kondisi cukup seimbang.",
            f"Level kelembapan {humidity_value}% menunjukkan keseimbangan yang baik."
        ],
        "rendah": [
            f"Kelembapan rendah, hanya {humidity_value}%. Mungkin perlu humidifier.",
            f"Ruangan agak kering dengan kelembapan {humidity_value}%."
        ]
    }
    sound_templates = {
        "tinggi": [
            f"Tingkat suara tinggi di {sound_value} dB. Mungkin ada aktivitas mencurigakan.",
            f"Dengan {sound_value} dB, ruangan cukup bising. Perlu diperiksa."
        ],
        "sedang": [
            f"Tingkat suara sedang di {sound_value} dB. Aktivitas normal kemungkinan terjadi.",
            f"Suara di {sound_value} dB menunjukkan kondisi cukup aktif."
        ],
        "rendah": [
            f"Tingkat suara rendah di {sound_value} dB. Ruangan relatif tenang.",
            f"Dengan {sound_value} dB, suasana ruangan sangat damai."
        ]
    }

    intro = random.choice(intro_templates)
    smoke = random.choice(smoke_templates.get(mq2_status.split()[0], ["Data asap tidak tersedia."]))
    light_key = "mencurigakan" if "mencurigakan" in lux_status.lower() else "gelap" if "gelap" in lux_status.lower() else "terang"
    light = random.choice(light_templates.get(light_key, ["Data cahaya tidak tersedia."]))
    temp_key = "panas" if "panas" in temp_status.lower() else "dingin" if "dingin" in temp_status.lower() else "normal"
    temp = random.choice(temp_templates.get(temp_key, ["Data suhu tidak tersedia."]))
    humidity_key = "tinggi" if "tinggi" in humidity_status.lower() else "rendah" if "rendah" in humidity_status.lower() else "normal"
    humidity = random.choice(humidity_templates.get(humidity_key, ["Data kelembapan tidak tersedia."]))
    sound_key = "tinggi" if "tinggi" in sound_status.lower() else "sedang" if "sedang" in sound_status.lower() else "rendah"
    sound = random.choice(sound_templates.get(sound_key, ["Data suara tidak tersedia."]))

    narrative = (
        f"📊 *Laporan Status Ruangan*\n"
        f"{intro}\n"
        f"- 🚨 {smoke}\n"
        f"- 💡 {light}\n"
        f"- 🌡️ {temp}\n"
        f"- 💧 {humidity}\n"
        f"- 🎙️ {sound}\n"
        f"🕒 *Waktu*: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    return narrative

def predict_smoking_risk_rule_based(mq2_value, lux_value, sound_value):
    if mq2_value is None or lux_value is None or sound_value is None:
        return "Data tidak cukup untuk prediksi risiko merokok."
    
    risk_score = 0
    risk_messages = []
    
    if mq2_value > 800:
        risk_score += 50
        risk_messages.append("Nilai asap sangat tinggi (MQ2 > 800), besar kemungkinan ada aktivitas merokok.")
    elif mq2_value >= 500:
        risk_score += 30
        risk_messages.append("Asap mencurigakan terdeteksi (MQ2 500-800), mungkin ada risiko merokok.")
    else:
        risk_messages.append("Asap rendah (MQ2 < 500), tidak ada indikasi kuat merokok.")
    
    if lux_value <= 50:
        risk_score += 20
        risk_messages.append("Ruangan gelap (lux ≤ 50), sering dikaitkan dengan aktivitas tersembunyi seperti merokok.")
    elif lux_value <= 100:
        risk_score += 10
        risk_messages.append("Pencahayaan rendah (lux ≤ 100), bisa memudahkan aktivitas merokok tanpa terdeteksi.")
    else:
        risk_messages.append("Ruangan terang (lux > 100), mengurangi kemungkinan merokok tersembunyi.")
    
    if sound_value >= 80:
        risk_score += 15
        risk_messages.append("Tingkat suara tinggi (≥ 80 dB), bisa menandakan aktivitas mencurigakan.")
    elif sound_value >= 50:
        risk_score += 5
        risk_messages.append("Tingkat suara sedang (50-80 dB), mungkin aktivitas normal.")
    else:
        risk_messages.append("Tingkat suara rendah (< 50 dB), ruangan tenang.")

    if risk_score >= 60:
        risk_level = "Tinggi"
        recommendation = "Segera periksa ruangan dan pastikan ventilasi baik. Aktivitas merokok sangat mungkin terjadi."
    elif risk_score >= 40:
        risk_level = "Sedang"
        recommendation = "Pantau ruangan lebih sering, terutama jika asap tetap terdeteksi. Pertimbangkan pemeriksaan manual."
    else:
        risk_level = "Rendah"
        recommendation = "Kondisi aman untuk saat ini. Tetap pertahankan pemantauan rutin."
    
    report = (
        f"🔍 *Prediksi Risiko Merokok*\n"
        f"Tingkat Risiko: **{risk_level}** (Skor: {risk_score})\n"
        f"Rincian:\n- {risk_messages[0]}\n- {risk_messages[1]}\n- {risk_messages[2]}\n"
        f"Rekomendasi: {recommendation}"
    )
    return report

def get_room_condition_summary(mq2_value, lux_value, temperature_value, humidity_value, sound_value):
    mq2_status = predict_smoke_status(mq2_value)
    lux_status = evaluate_lux_condition(lux_value, mq2_value)
    temp_status = evaluate_temperature_condition(temperature_value)
    humidity_status = (f"Kelembapan {humidity_value}%: {'tinggi' if humidity_value > 70 else 'normal' if humidity_value >= 30 else 'rendah'}"
                      if humidity_value is not None else "Data kelembapan tidak tersedia.")
    sound_status = evaluate_sound_condition(sound_value)

    if "Bahaya" in mq2_status or "berbahaya" in temp_status.lower() or "tinggi" in sound_status.lower():
        overall_status = "Bahaya"
        color = "red"
        suggestion = "Segera ambil tindakan: periksa sumber asap, suhu, atau suara mencurigakan."
    elif "Mencurigakan" in mq2_status or "mencurigakan" in lux_status.lower() or "panas" in temp_status.lower() or "tinggi" in humidity_status.lower() or "sedang" in sound_status.lower():
        overall_status = "Waspada"
        color = "orange"
        suggestion = "Pantau ruangan lebih sering dan pertimbangkan ventilasi, penyesuaian cahaya, atau pemeriksaan suara."
    else:
        overall_status = "Aman"
        color = "green"
        suggestion = "Kondisi ruangan baik, lanjutkan pemantauan rutin."

    return {
        "status": overall_status,
        "color": color,
        "suggestion": suggestion,
        "details": {
            "Asap": mq2_status,
            "Cahaya": lux_status,
            "Suhu": temp_status,
            "Kelembapan": humidity_status,
            "Suara": sound_status
        }
    }

# --- GEMINI AI CHATBOT ---
def get_gemini_response(messages):
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": GEMINI_MODEL,
        "messages": messages,
        "stream": False,
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            logger.error(f"Error {response.status_code} dari Gemini API: {response.text}")
            return f"Maaf, terjadi kesalahan saat menghubungi AI. (Error {response.status_code})"
    except Exception as e:
        logger.error(f"Exception saat menghubungi Gemini API: {str(e)}")
        return "Maaf, terjadi kesalahan saat menghubungi AI. Silakan coba lagi nanti."

def generate_chatbot_context(mq2_value, lux_value, temperature_value, humidity_value, sound_value):
    """Generate context for the chatbot based on sensor data"""
    mq2_status = predict_smoke_status(mq2_value)
    lux_status = evaluate_lux_condition(lux_value, mq2_value)
    temp_status = evaluate_temperature_condition(temperature_value)
    humidity_status = (f"Kelembapan {humidity_value}%: {'tinggi' if humidity_value > 70 else 'normal' if humidity_value >= 30 else 'rendah'}"
                      if humidity_value is not None else "Data kelembapan tidak tersedia.")
    sound_status = evaluate_sound_condition(sound_value)
    
    return (
        "Anda adalah asisten AI untuk sistem deteksi merokok. Berikut data sensor terbaru:\n"
        f"- Sensor Asap (MQ2): {mq2_value} ({mq2_status})\n"
        f"- Sensor Cahaya: {lux_value} lux ({lux_status})\n"
        f"- Sensor Suhu: {temperature_value}°C ({temp_status})\n"
        f"- Sensor Kelembapan: {humidity_value}% ({humidity_status})\n"
        f"- Sensor Suara: {sound_value} dB ({sound_status})\n\n"
        "Anda bisa menjawab pertanyaan tentang kondisi ruangan, deteksi asap rokok, "
        "atau memberikan saran berdasarkan data sensor. Gunakan bahasa yang jelas dan informatif."
    )

# --- ESP32-CAM DETECTION ---
@st.cache_resource
def load_yolo_model():
    try:
        logger.info("Mencoba memuat model YOLOv5...")
        st.write("Mencoba memuat model YOLOv5...")
        st.write(f"Current working directory: {os.getcwd()}")
        st.write(f"Model file exists: {os.path.exists('model/best.pt')}")
        if os.path.exists('model/best.pt'):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best.pt', force_reload=True)
        else:
            logger.warning("File model/best.pt tidak ditemukan, menggunakan model yolov5s")
            st.warning("File model/best.pt tidak ditemukan, menggunakan model yolov5s")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
        logger.info("Model YOLOv5 berhasil dimuat")
        st.write("Model YOLOv5 berhasil dimuat")
        return model
    except Exception as e:
        logger.error(f"Gagal memuat model YOLOv5: {str(e)}")
        st.error(f"Gagal memuat model YOLOv5: {str(e)}")
        return None

def run_camera_detection(frame_placeholder, status_placeholder):
    try:
        logger.info(f"Mencoba membuka stream kamera di {CAMERA_URL}")
        cap = cv2.VideoCapture(CAMERA_URL)
        if not cap.isOpened():
            logger.error(f"Tidak dapat membuka stream kamera di {CAMERA_URL}")
            status_placeholder.error(
                f"Tidak dapat membuka stream kamera di {CAMERA_URL}. "
                "Periksa apakah ESP32-CAM aktif, URL benar, atau port/streaming path diperlukan (misalnya, :81/stream)."
            )
            return
        last_saved_time = 0
        last_smoking_notification = 0
        save_interval = 600  # 10 menit untuk penyimpanan gambar lokal

        while st.session_state.get("cam_running", False) and st.session_state.get("cam_refresh", False):
            ret, frame = cap.read()
            if not ret:
                logger.error("Gagal membaca frame dari kamera")
                status_placeholder.error("Gagal membaca frame dari kamera. Stream mungkin terputus.")
                break

            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if st.session_state.get("model_cam") is not None:
                results = st.session_state.model_cam(img_pil)
                results.render()
                rendered = results.ims[0]
                frame = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

                df = results.pandas().xyxy[0]
                found_person = 'person' in df['name'].values
                found_smoke = 'smoke' in df['name'].values
            else:
                found_person = False
                found_smoke = False

            current_time = time.time()

            _, buffer = cv2.imencode('.jpg', frame)
            st.session_state.latest_frame = buffer.tobytes()

            if found_person and found_smoke:
                logger.warning("Merokok terdeteksi di ruangan")
                status_placeholder.warning("Merokok terdeteksi di ruangan!")
                if current_time - last_smoking_notification > ALERT_COOLDOWN:
                    caption = (
                        f"🚨 *Peringatan*: Aktivitas merokok terdeteksi di ruangan!\n"
                        f"🕒 *Waktu*: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    asyncio.run(send_telegram_photo(st.session_state.latest_frame, caption))
                    last_smoking_notification = current_time

                    sensor_data = fetch_latest_sensor_data()
                    values = sensor_data["values"]
                    statuses = sensor_data["statuses"]
                    narrative = generate_narrative_report(
                        statuses["mq2"], values.get("mq2", "N/A"),
                        statuses["lux"], values.get("lux", "N/A"),
                        statuses["temperature"], values.get("temperature", "N/A"),
                        statuses["humidity"], values.get("humidity", "N/A"),
                        statuses["sound"], values.get("sound", "N/A")
                    )
                    asyncio.run(send_telegram_photo(st.session_state.latest_frame, narrative))

                if current_time - last_saved_time > save_interval:
                    filename = datetime.datetime.now().strftime("smoking_%Y%m%d_%H%M%S.jpg")
                    cv2.imwrite(filename, frame)
                    last_saved_time = current_time
                    logger.info(f"Gambar disimpan: {filename}")
                    status_placeholder.info(f"Gambar disimpan: {filename}")
            else:
                status_placeholder.success("Tidak ada aktivitas merokok terdeteksi di ruangan.")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_placeholder.image(frame_pil, channels="RGB", use_container_width=True)
            st.session_state.last_frame = frame_pil

            time.sleep(0.1)

        cap.release()
    except Exception as e:
        logger.error(f"Error kamera: {str(e)}")
        status_placeholder.error(f"Error kamera: {str(e)}")
    finally:
        pass

# --- PERIODIC NOTIFICATION FUNCTION ---
def send_periodic_notification():
    """
    Mengirim notifikasi periodik ke Telegram setiap 5 menit dengan data suhu, kelembapan, asap, intensitas cahaya, dan suara.
    """
    current_time = time.time()
    if current_time - st.session_state.last_notification['last_sent'] < NOTIFICATION_INTERVAL:
        return
    
    logger.info("Mengirim notifikasi periodik...")
    
    sensor_data = fetch_latest_sensor_data()
    values = sensor_data["values"]
    statuses = sensor_data["statuses"]
    
    st.session_state.last_notification['mq2']['status'] = statuses["mq2"]
    st.session_state.last_notification['mq2']['value'] = values.get("mq2")
    st.session_state.last_notification['lux']['status'] = statuses["lux"]
    st.session_state.last_notification['lux']['value'] = values.get("lux")
    st.session_state.last_notification['temperature']['status'] = statuses["temperature"]
    st.session_state.last_notification['temperature']['value'] = values.get("temperature")
    st.session_state.last_notification['humidity']['status'] = statuses["humidity"]
    st.session_state.last_notification['humidity']['value'] = values.get("humidity")
    st.session_state.last_notification['sound']['status'] = statuses["sound"]
    st.session_state.last_notification['sound']['value'] = values.get("sound")
    
    caption = (
        f"📊 *Laporan Status Ruangan* ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"
        f"💨 *Asap*: {statuses['mq2']} (Nilai: {values.get('mq2', 'N/A')})\n"
        f"💡 *Cahaya*: {statuses['lux']} (Nilai: {values.get('lux', 'N/A')} lux)\n"
        f"🌡️ *Suhu*: {statuses['temperature']} (Nilai: {values.get('temperature', 'N/A')}°C)\n"
        f"💧 *Kelembapan*: {statuses['humidity']} (Nilai: {values.get('humidity', 'N/A')}%)\n"
        f"🎙️ *Suara*: {statuses['sound']} (Nilai: {values.get('sound', 'N/A')} dB)\n"
    )
    
    if st.session_state.latest_frame is not None:
        asyncio.run(send_telegram_photo(st.session_state.latest_frame, caption))
    else:
        asyncio.run(send_telegram_message(caption + "\n⚠️ *Foto*: Kamera tidak aktif"))
    
    st.session_state.last_notification['last_sent'] = current_time
    logger.info("Notifikasi periodik dikirim")

# --- UI START ---
st.markdown('<div class="main-title">Sistem Deteksi Merokok Terintegrasi</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["IoT Sensor", "ESP32-CAM"])

# --- IOT TAB ---
with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.header("Live Stream Data + AI Deteksi Rokok & Cahaya")

    mq2_value_latest = None
    lux_value_latest = None
    temperature_value_latest = None
    humidity_value_latest = None
    sound_value_latest = None

    auto_refresh_iot = st.checkbox("Aktifkan Auto-Refresh Data IoT", value=True, key="iot_refresh")
    if auto_refresh_iot:
        st_autorefresh(interval=5000, key="iot_auto_refresh")

    if 'last_notification' not in st.session_state:
        st.session_state.last_notification = {
            'mq2': {'status': None, 'value': None, 'last_alert_sent': 0},
            'lux': {'status': None, 'value': None},
            'temperature': {'status': None, 'value': None},
            'humidity': {'status': None, 'value': None},
            'sound': {'status': None, 'value': None},
            'last_sent': 0
        }

    if 'latest_frame' not in st.session_state:
        st.session_state.latest_frame = None

    if 'last_frame' not in st.session_state:
        st.session_state.last_frame = None

    # Initialize chat history if not exists
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{
            "role": "system",
            "content": "Anda adalah asisten AI untuk sistem deteksi merokok. Anda bisa menjawab pertanyaan tentang kondisi ruangan, deteksi asap rokok, atau memberikan saran berdasarkan data sensor."
        }]

    for var_name in VARIABLES:
        data = get_ubidots_data(var_name)
        if data:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            value = round(df.iloc[0]['value'], 2)

            if var_name == "mq2":
                var_label = "ASAP/GAS"
                emoji = "💨"
                mq2_value_latest = value
            elif var_name == "humidity":
                var_label = "KELEMBAPAN"
                emoji = "💧"
                humidity_value_latest = value
            elif var_name == "temperature":
                var_label = "SUHU"
                emoji = "🌡️"
                temperature_value_latest = value
            elif var_name == "lux":
                var_label = "INTENSITAS CAHAYA"
                emoji = "💡"
                lux_value_latest = value
            elif var_name == "sound":
                var_label = "SUARA"
                emoji = "🎙️"
                sound_value_latest = value

            st.markdown(
                f'<div class="data-box"><span class="label">{emoji} {var_label}</span><span class="data-value">{value}</span></div>',
                unsafe_allow_html=True
            )

            st.line_chart(df[['timestamp', 'value']].set_index('timestamp'))

            current_time = time.time()

            if var_name == "mq2":
                status = predict_smoke_status(value)
                if "Bahaya" in status and \
                   current_time - st.session_state.last_notification['mq2']['last_alert_sent'] > ALERT_COOLDOWN:
                    caption = (
                        f"🚨 *Peringatan Asap*: {status}\n"
                        f"📊 *Nilai MQ2*: {value}\n"
                        f"🕒 *Waktu*: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    if st.session_state.latest_frame is not None:
                        asyncio.run(send_telegram_photo(st.session_state.latest_frame, caption))
                    else:
                        asyncio.run(send_telegram_message(caption + "\n⚠️ *Foto*: Kamera tidak aktif"))
                    st.session_state.last_notification['mq2']['last_alert_sent'] = current_time

                    sensor_data = fetch_latest_sensor_data()
                    values = sensor_data["values"]
                    statuses = sensor_data["statuses"]
                    narrative = generate_narrative_report(
                        statuses["mq2"], values.get("mq2", "N/A"),
                        statuses["lux"], values.get("lux", "N/A"),
                        statuses["temperature"], values.get("temperature", "N/A"),
                        statuses["humidity"], values.get("humidity", "N/A"),
                        statuses["sound"], values.get("sound", "N/A")
                    )
                    if st.session_state.latest_frame is not None:
                        asyncio.run(send_telegram_photo(st.session_state.latest_frame, narrative))
                    else:
                        asyncio.run(send_telegram_message(narrative + "\n⚠️ *Foto*: Kamera tidak aktif"))

                st.session_state.last_notification['mq2']['status'] = status
                st.session_state.last_notification['mq2']['value'] = value
                if "Bahaya" in status:
                    st.error(status)
                elif "Mencurigakan" in status:
                    st.warning(status)
                else:
                    st.success(status)

            if var_name == "lux":
                lux_status = evaluate_lux_condition(value, mq2_value_latest)
                st.session_state.last_notification['lux']['status'] = lux_status
                st.session_state.last_notification['lux']['value'] = value
                if "mencurigakan" in lux_status.lower():
                    st.warning(lux_status)
                else:
                    st.info(lux_status)

            if var_name == "temperature":
                temp_status = evaluate_temperature_condition(value)
                st.session_state.last_notification['temperature']['status'] = temp_status
                st.session_state.last_notification['temperature']['value'] = value
                if "panas" in temp_status.lower() or "berbahaya" in temp_status.lower():
                    st.warning(temp_status)
                elif "dingin" in temp_status.lower():
                    st.info(temp_status)
                else:
                    st.success(temp_status)

            if var_name == "humidity":
                humidity_status = f"Kelembapan {value}%: {'tinggi' if value > 70 else 'normal' if value >= 30 else 'rendah'}"
                st.session_state.last_notification['humidity']['status'] = humidity_status
                st.session_state.last_notification['humidity']['value'] = value
                st.info(humidity_status)

            if var_name == "sound":
                sound_status = evaluate_sound_condition(value)
                st.session_state.last_notification['sound']['status'] = sound_status
                st.session_state.last_notification['sound']['value'] = value
                if "tinggi" in sound_status.lower():
                    st.warning(sound_status)
                elif "sedang" in sound_status.lower():
                    st.info(sound_status)
                else:
                    st.success(sound_status)

        else:
            st.error(f"Gagal mengambil data dari variabel: {var_name}")

    if all(v is not None for v in [mq2_value_latest, lux_value_latest, temperature_value_latest, humidity_value_latest, sound_value_latest]):
        summary = get_room_condition_summary(
            mq2_value_latest, lux_value_latest, temperature_value_latest, humidity_value_latest, sound_value_latest
        )
        st.markdown(
            f"""
            <div style="background-color: #f0f0f0; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <h3 style="color: {summary['color']};">Status Ruangan: {summary['status']}</h3>
                <p><strong>Saran:</strong> {summary['suggestion']}</p>
                <p><strong>Detail:</strong></p>
                <ul>
                    <li>Asap: {summary['details']['Asap']}</li>
                    <li>Cahaya: {summary['details']['Cahaya']}</li>
                    <li>Suhu: {summary['details']['Suhu']}</li>
                    <li>Kelembapan: {summary['details']['Kelembapan']}</li>
                    <li>Suara: {summary['details']['Suara']}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    send_periodic_notification()

    st.subheader("Uji Notifikasi Telegram")
    if st.button("Kirim Pesan Uji ke Telegram"):
        asyncio.run(send_telegram_message("🔍 *Pesan Uji*: Sistem deteksi merokok berfungsi dengan baik!"))

    if all(v is not None for v in [mq2_value_latest, lux_value_latest, temperature_value_latest, humidity_value_latest, sound_value_latest]):
        st.markdown("---")
        st.subheader("💬 Chatbot AI (Gemini Flash)")
        
        # Chatbot dalam form untuk menghindari refresh
        with st.form(key="chat_form", clear_on_submit=True):
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Tampilkan hanya pesan terbaru (jika ada)
            if len(st.session_state.chat_messages) > 1:
                for message in st.session_state.chat_messages[1:]:  # Skip system message
                    if message["role"] == "user":
                        st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="assistant-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="assistant-message">🤖 Tulis pertanyaan Anda untuk memulai!</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Input dan tombol submit
            user_input = st.text_input("Tulis pesan kamu...", key="chat_input")
            submit_button = st.form_submit_button("Kirim")

            if submit_button and user_input:
                # Reset chat history, hanya simpan system message
                st.session_state.chat_messages = [{
                    "role": "system",
                    "content": generate_chatbot_context(
                        mq2_value_latest,
                        lux_value_latest,
                        temperature_value_latest,
                        humidity_value_latest,
                        sound_value_latest
                    )
                }]
                
                # Tambahkan pesan pengguna
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                
                # Dapatkan respons AI
                with st.spinner("Menunggu respon AI..."):
                    ai_response = get_gemini_response(st.session_state.chat_messages)
                    
                    # Tambahkan respons AI ke chat history
                    st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
                
                # Refresh form untuk menampilkan pesan baru
                st.rerun()

    st.markdown("---")
    st.subheader("🔍 Prediksi Risiko Merokok")
    if st.button("Analisis Risiko Merokok"):
        prediction = predict_smoking_risk_rule_based(mq2_value_latest, lux_value_latest, sound_value_latest)
        st.write(prediction)
        if "Tinggi" in prediction:
            asyncio.run(send_telegram_message(f"🚨 *Peringatan Risiko*: {prediction}"))

    st.markdown('</div>', unsafe_allow_html=True)

# --- ESP32-CAM TAB ---
with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.header("Deteksi Merokok dengan ESP32-CAM")
    st.write("Mendeteksi aktivitas merokok secara real-time menggunakan kamera ESP32-CAM dan model YOLOv5.")

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    col1, col2 = st.columns(2)
    with col1:
        start_cam = st.checkbox("Mulai Deteksi", value=False, key="cam_start")
    with col2:
        auto_refresh_cam = st.checkbox("Aktifkan Auto-Refresh Kamera", value=True, key="cam_refresh")

    if start_cam:
        st.session_state.cam_running = True
        if 'model_cam' not in st.session_state or st.session_state.model_cam is None:
            st.session_state.model_cam = load_yolo_model()
        if auto_refresh_cam:
            run_camera_detection(frame_placeholder, status_placeholder)
        elif st.session_state.last_frame is not None:
            frame_placeholder.image(st.session_state.last_frame, channels="RGB", use_container_width=True)
            status_placeholder.info("Auto-refresh kamera dimatikan. Menampilkan gambar terakhir.")
        else:
            status_placeholder.warning("Tidak ada gambar terakhir untuk ditampilkan.")
    else:
        st.session_state.cam_running = False
        if st.session_state.last_frame is not None:
            frame_placeholder.image(st.session_state.last_frame, channels="RGB", use_container_width=True)
            status_placeholder.info("Kamera dimatikan. Menampilkan gambar terakhir.")
        else:
            status_placeholder.info("Klik 'Mulai Deteksi' untuk memulai streaming dari kamera.")

    st.markdown('</div>', unsafe_allow_html=True)
