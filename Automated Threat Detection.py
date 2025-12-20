import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os
import time
from datetime import datetime
from collections import deque
import pandas as pd
import io
import base64
import threading
import pyttsx3

def speak_alert(message):
    """Speaks the message out loud automatically."""
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 165)   # Adjust speed
            engine.setProperty('volume', 1.0) # Max volume
            engine.say(message)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    threading.Thread(target=_speak, daemon=True).start()


# Twilio config
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_MESSAGING_SERVICE_SID = os.getenv('TWILIO_MESSAGING_SERVICE_SID')
TWILIO_TO_NUMBER = os.getenv('TWILIO_TO_NUMBER')

# Optional heavy imports (wrapped in try/except to keep app usable if missing)
try:
    from moviepy.editor import VideoFileClip
except Exception:
    VideoFileClip = None

try:
    import librosa
    import soundfile as sf
except Exception:
    librosa = None
    sf = None

try:
    from fpdf import FPDF
except Exception:
    FPDF = None

try:
    import requests
except Exception:
    requests = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    from twilio.rest import Client as TwilioClient
except Exception:
    TwilioClient = None

# ========================
# CONFIG
# ========================
IMG_SIZE = 64
FRAMES_PER_VIDEO = 30
LABELS = ["Non-Violence", "Violence"]
DETECTION_CONF_THRESHOLD = 0.6  # base threshold for video model
FUSED_THRESHOLD = 0.6           # fused video+audio threshold to trigger alarms
SAVE_DIR = "detections"
EVENT_LOG = "event_log.csv"
REPORTS_DIR = "reports"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ========================
# LOAD MODEL
# ========================

# ========================
# LOAD MODEL
# ========================
@st.cache_resource
def load_model_cnn_lstm():
    model_path = "violence_detection_model.keras"
    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
        st.stop()
    return load_model(model_path, compile=False)

model = load_model_cnn_lstm()

# ========================
# HELPERS
# ========================
def preprocess_frames(frames):
    processed = [cv2.resize(f, (IMG_SIZE, IMG_SIZE)) for f in frames]
    processed = np.array(processed) / 255.0
    if len(processed) < FRAMES_PER_VIDEO:
        pad_len = FRAMES_PER_VIDEO - len(processed)
        processed = np.concatenate([processed, np.zeros((pad_len, IMG_SIZE, IMG_SIZE, 3))])
    else:
        processed = processed[:FRAMES_PER_VIDEO]
    return np.expand_dims(processed, axis=0)

def predict_from_frames(frames):
    input_data = preprocess_frames(frames)
    pred = model.predict(input_data, verbose=0)[0]
    if pred.shape[0] == 1:
        label = LABELS[int(pred[0] > 0.5)]
        conf = float(pred[0] if pred[0] > 0.5 else 1 - pred[0])
    else:
        label = LABELS[np.argmax(pred)]
        conf = float(np.max(pred))
    return label, conf

def send_sms_alert(cam_location):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            messaging_service_sid=TWILIO_MESSAGING_SERVICE_SID,
            body=f"ALERT: Violence detected at {cam_location}!",
            to=TWILIO_TO_NUMBER
        )
        st.success(f"SMS sent! SID: {message.sid}")
    except Exception as e:
        st.error(f"Twilio SMS failed: {e}")

def sample_frames(frames, k=FRAMES_PER_VIDEO):
    n = len(frames)
    if n <= k:
        return frames + [frames[-1]]*(k-n)
    idxs = np.linspace(0, n-1, k).astype(int)
    return [frames[i] for i in idxs]

# ========================
# HELPERS: Video / Model
# ========================

def preprocess_frames(frames, img_size=IMG_SIZE, max_frames=FRAMES_PER_VIDEO):
    processed = [cv2.resize(f, (img_size, img_size)) for f in frames]
    processed = np.array(processed) / 255.0

    if len(processed) < max_frames:
        pad_len = max_frames - len(processed)
        processed = np.concatenate([processed, np.zeros((pad_len, img_size, img_size, 3))])
    else:
        processed = processed[:max_frames]

    return np.expand_dims(processed, axis=0)

def predict_from_frames(frames):
    input_data = preprocess_frames(frames)
    pred = model.predict(input_data, verbose=0)[0]

    if pred.shape[0] == 1:
        label = LABELS[int(pred[0] > 0.5)]
        conf = float(pred[0] if pred[0] > 0.5 else 1 - pred[0])
    else:
        label = LABELS[np.argmax(pred)]
        conf = float(np.max(pred))
    return label, conf

def sample_frames(frames, k=FRAMES_PER_VIDEO):
    # evenly sample k frames from frames list
    n = len(frames)
    if n == 0:
        return []
    if n <= k:
        return frames + [frames[-1]] * (k - n)
    idxs = np.linspace(0, n - 1, k).astype(int)
    return [frames[i] for i in idxs]

# =============================
# HELPERS: Audio
# =============================

def extract_audio_from_video(video_path, out_audio_path):
    if VideoFileClip is None:
        raise RuntimeError("moviepy not installed — can't extract audio from video.")
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        return None
    clip.audio.write_audiofile(out_audio_path, verbose=False, logger=None)
    return out_audio_path

def analyze_audio_heuristic(audio_path):
    """
    Simple audio heuristic returning score 0..1 where higher means more likely violence (screams/loud events).
    Uses RMS energy, spectral centroid and zero-crossing rate.
    This is a heuristic demo — replace with trained audio model for production.
    """
    if librosa is None:
        return 0.0
    try:
        y, sr = librosa.load(audio_path, sr=None)
        # short-time features
        rmse = librosa.feature.rms(y=y).mean()
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()

        # heuristic normalize
        # choose rough scales observed empirically
        rmse_score = min(1.0, rmse / 0.02)
        centroid_score = min(1.0, centroid / 3000.0)
        zcr_score = min(1.0, zcr / 0.1)

        # combine giving more weight to energy + centroid
        score = (0.6 * rmse_score) + (0.3 * centroid_score) + (0.1 * zcr_score)
        score = float(np.clip(score, 0.0, 1.0))
        return score
    except Exception as e:
        st.warning("Audio analysis failed: " + str(e))
        return 0.0
# =============================
# HELPERS: Save frame & clip
# =============================
def save_detection_artifacts(frames, image_prefix="capture"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(SAVE_DIR, f"{image_prefix}_{timestamp}.jpg")
    # save middle frame as jpg
    mid = frames[len(frames) // 2]
    mid_bgr = cv2.cvtColor(mid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, mid_bgr)

    # save short clip
    clip_path = os.path.join(SAVE_DIR, f"{image_prefix}_{timestamp}.mp4")
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(clip_path, fourcc, 10.0, (width, height))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

    return img_path, clip_path

# ========================
# HELPER: Twilio SMS (One-Time Alert)
# ========================
from twilio.rest import Client

# Twilio credentials
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
messaging_service_sid = os.getenv('TWILIO_MESSAGING_SERVICE_SID')
to_number = os.getenv('TWILIO_TO_NUMBER')

# Create Twilio client
client = Client(account_sid, auth_token)

# 🔒 One-time alert flag
alert_sent = False

def send_alert_one_time():
    global alert_sent

    if alert_sent:
        return   # ❌ already sent, do nothingf

    message = client.messages.create(
        messaging_service_sid=messaging_service_sid,
        body='Sir, suspicious activity detected in Cam 1!',
        to=to_number
    )

    print("SMS sent, SID:", message.sid)
    alert_sent = True   # ✅ block future alerts

# Example usage:
# Call this function only when your detection model triggers
send_alert_one_time()

# ========================
# HELPERS: Event Logging
# ========================
def log_event_row(row: dict):
    df = pd.DataFrame([row])
    if not os.path.exists(EVENT_LOG):
        df.to_csv(EVENT_LOG, index=False)
    else:
        df.to_csv(EVENT_LOG, mode='a', header=False, index=False)

# ========================
# HELPERS: PDF Report
# ========================
def create_pdf_report(event_row, image_path, output_pdf_path):
    if FPDF is None:
        raise RuntimeError("fpdf not installed — can't create PDF report.")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "AI-Powered Violence Detection Report", ln=True, align='C')
    pdf.ln(5)

    for k, v in event_row.items():
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 8, f"{k}: {v}", ln=True)

    pdf.ln(5)
    try:
        pdf.image(image_path, x=10, w=180)
    except Exception:
        pdf.ln(40)
        pdf.cell(0, 8, "[Image could not be embedded]", ln=True)

    pdf.output(output_pdf_path)
    return output_pdf_path

# ========================
# HELPERS: Notifications (Email, Telegram, Twilio)
# ========================
def send_email_alert(smtp_server, smtp_port, smtp_user, smtp_password, recipient, subject, body, attachments=[]):
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = recipient
    msg.set_content(body)

    for path in attachments:
        try:
            with open(path, 'rb') as f:
                data = f.read()
                maintype = 'application'
                subtype = 'octet-stream'
                filename = os.path.basename(path)
                msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
        except Exception as e:
            st.warning(f"Failed to attach {path}: {e}")

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email send failed: {e}")
        return False

def send_telegram_alert(bot_token, chat_id, message, photo_path=None):
    if requests is None:
        st.warning("Requests library not available — cannot send Telegram message.")
        return False
    try:
        if photo_path is None:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            resp = requests.post(url, data={"chat_id": chat_id, "text": message})
        else:
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            with open(photo_path, 'rb') as img:
                resp = requests.post(url, data={"chat_id": chat_id, "caption": message}, files={"photo": img})
        return resp.status_code == 200
    except Exception as e:
        st.warning(f"Telegram send failed: {e}")
        return False

def send_twilio_sms(account_sid, auth_token, from_number, to_number, message):
    if TwilioClient is None:
        st.warning("twilio library not available — cannot send SMS via Twilio.")
        return False
    try:
        client = TwilioClient(account_sid, auth_token)
        client.messages.create(body=message, from_=from_number, to=to_number)
        return True
    except Exception as e:
        st.warning(f"Twilio SMS failed: {e}")
        return False

# ========================
# HELPERS: TTS (speak and create audio playable in browser)
# ========================

def auto_tts_alert(message, save_path="tts_alert.mp3"):
    """
    Automatically speaks the given message using pyttsx3 
    (no user interaction) and saves it as an MP3.
    """
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 165)   # speed of speech
            engine.setProperty('volume', 1.0) # max volume
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)

            # Save and speak together
            engine.save_to_file(message, save_path)
            engine.say(message)
            engine.runAndWait()
            
            print(f"[TTS] Spoke alert: {message}")
        except Exception as e:
            print(f"[TTS Error] {e}")

    # Run in background so Streamlit doesn't freeze
    threading.Thread(target=_speak, daemon=True).start()


# ========================
# STREAMLIT UI
# ========================
st.set_page_config(page_title="AI-Automated Threat Recognition System", layout="wide")
st.title("Automated Threat Recognition System")
st.markdown("Detect Violence / Non-Violence using webcam, image, or video upload. Now with audio fusion, logging, PDF reports, TTS and notifications.")

# Sidebar: notification settings
st.sidebar.header("Settings & Notifications")
cam_location = st.sidebar.text_input("Camera Location / Label", value="Cam-1")
use_email = st.sidebar.checkbox("Enable Email Alerts")
smtp_server = st.sidebar.text_input("SMTP Server (e.g. smtp.gmail.com)", value="smtp.gmail.com")
smtp_port = st.sidebar.number_input("SMTP Port", value=465)
smtp_user = st.sidebar.text_input("SMTP User (sender email)")
smtp_password = st.sidebar.text_input("SMTP Password (or app password)", type="password")
email_recipient = st.sidebar.text_input("Recipient Email")

use_telegram = st.sidebar.checkbox("Enable Telegram Alerts")
telegram_token = st.sidebar.text_input("Telegram Bot Token")
telegram_chat = st.sidebar.text_input("Telegram Chat ID")

use_twilio = st.sidebar.checkbox("Enable SMS (Twilio)")
twilio_sid = st.sidebar.text_input("Twilio SID")
twilio_token = st.sidebar.text_input("Twilio Auth Token")
twilio_from = st.sidebar.text_input("Twilio From Number")
twilio_to = st.sidebar.text_input("Twilio To Number")

enable_pdf = st.sidebar.checkbox("Generate PDF Report on Detection", value=True)
enable_tts = st.sidebar.checkbox("Enable Voice Assistant (TTS)", value=True)

st.markdown("---")

# CSS for alert box (reused)
st.markdown("""
    <style>
    .alert-box {
        background: linear-gradient(90deg, #ff0000, #ff3333, #ff0000);
        color: white;
        padding: 18px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        border-radius: 15px;
        box-shadow: 0px 0px 25px rgba(255, 0, 0, 0.8);
        animation: pulse 1s infinite alternate;
        letter-spacing: 1.2px;
        margin-top: 20px;
    }
    @keyframes pulse {
        0% {opacity: 1; transform: scale(1);} 
        100% {opacity: 0.75; transform: scale(1.05);} 
    }
    .alert-text {
        text-shadow: 0 0 10px #fff, 0 0 20px #ff4d4d, 0 0 30px #ff0000;
    }
    </style>
""", unsafe_allow_html=True)

mode = st.radio("Choose Mode:", ["🎥 Real-Time Webcam", "📸 Image Upload", "🎬 Video Upload"])

# utility: load current event log
def load_event_log():
    import pandas as pd
    if os.path.exists(EVENT_LOG):
        try:
            # skip bad lines automatically
            return pd.read_csv(EVENT_LOG, on_bad_lines='skip', engine='python')
        except Exception as e:
            st.error(f"Failed to read event log: {e}")
            return pd.DataFrame(columns=[
                "timestamp", "camera", "label", "video_conf", 
                "audio_score", "fused_score", "image_path", "clip_path", "detection_source"
            ])
    else:
        return pd.DataFrame(columns=[
            "timestamp", "camera", "label", "video_conf", 
            "audio_score", "fused_score", "image_path", "clip_path", "detection_source"
        ])


# ===============================
# MODE: WEBCAM
# ===============================
if mode == "🎥 Real-Time Webcam":
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    prediction_text = st.empty()
    alert_box = st.empty()

    # Optionally allow uploading a separate audio file to fuse with webcam predictions
    webcam_audio = st.file_uploader("(Optional) Upload audio file to fuse with webcam predictions (wav, mp3)", type=["wav", "mp3"])

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access webcam.")
            st.stop()

        frame_buffer = deque(maxlen=FRAMES_PER_VIDEO)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read frame from webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(rgb_frame)
            frame_buffer.append(rgb_frame)

            # process when buffer full
            if len(frame_buffer) == FRAMES_PER_VIDEO:
                frames = list(frame_buffer)
                label, conf = predict_from_frames(frames)
                audio_score = 0.0
                if webcam_audio is not None and VideoFileClip is not None:
                    # save uploaded audio to temp and analyze
                    tmp_a = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(webcam_audio.name)[1])
                    tmp_a.write(webcam_audio.read())
                    tmp_a.close()
                    audio_score = analyze_audio_heuristic(tmp_a.name) if librosa is not None else 0.0

                # fuse
                fused = (0.75 * conf) + (0.25 * audio_score)

                prediction_text.markdown(f"### Prediction: {label} (Video Conf: {conf:.2f}, AudioScore: {audio_score:.2f}, Fused: {fused:.2f})")

                if fused >= FUSED_THRESHOLD and label == "Violence":
                    alert_box.markdown("""
                        <div class='alert-box'>
                            🚨 <span class='alert-text'>VIOLENCE DETECTED!</span> ⚠
                        </div>
                        """, unsafe_allow_html=True)

                    # save artifacts
                    img_path, clip_path = save_detection_artifacts(frames, image_prefix=cam_location.replace(' ', '_'))
                    # log event
                    row = {
                        'timestamp': datetime.now().isoformat(),
                        'camera': cam_location,
                        'label': label,
                        'video_conf': float(conf),
                        'audio_score': float(audio_score),
                        'fused_score': float(fused),
                        'image_path': img_path,
                        'clip_path': clip_path
                    }
                    log_event_row(row)

                    # PDF
                    pdf_path = None
                    if enable_pdf:
                        try:
                            pdf_path = create_pdf_report(row, img_path, os.path.join(REPORTS_DIR, os.path.basename(img_path).replace('.jpg', '.pdf')))
                        except Exception as e:
                            st.warning(f"PDF creation failed: {e}")

                    # notifications
                    subject = f"ALERT: Violence Detected at {cam_location}"
                    body = f"Time: {row['timestamp']}\nCamera: {cam_location}\nFused Score: {fused:.2f}"

                    if use_email and smtp_user and smtp_password and email_recipient:
                        attachments = [p for p in [img_path, pdf_path] if p]
                        send_email_alert(smtp_server, int(smtp_port), smtp_user, smtp_password, email_recipient, subject, body, attachments)

                    if use_telegram and telegram_token and telegram_chat:
                        send_telegram_alert(telegram_token, telegram_chat, body, photo_path=img_path)

                    if use_twilio and twilio_sid and twilio_token and twilio_from and twilio_to:
                        send_twilio_sms(twilio_sid, twilio_token, twilio_from, twilio_to, subject + " — " + body)

                    # TTS
                    if enable_tts:
                        tts_msg = f"Sir, I've detected unusual activity in {cam_location}."
                        auto_tts_alert(tts_msg)
                else:
                    alert_box.empty()

                frame_buffer.clear()

            # small delay
            time.sleep(0.02)

        cap.release()

# ========================
# MODE: IMAGE UPLOAD
# ========================
elif mode == "📸 Image Upload":
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption="Uploaded Image", width="stretch")

        frames = [image] * FRAMES_PER_VIDEO 
        label, conf = predict_from_frames(frames)
        audio_score = 0.0
        fused = (0.85 * conf) + (0.15 * audio_score)

        st.markdown(f"### Prediction: {label} (Video Conf: {conf:.2f}, Fused: {fused:.2f})")

        if fused >= FUSED_THRESHOLD and label == "Violence":
            st.markdown("""
                <div class='alert-box'>
                    🚨 <span class='alert-text'>VIOLENCE DETECTED!</span> ⚠
                </div>
                """, unsafe_allow_html=True)

            img_path, clip_path = save_detection_artifacts(frames, image_prefix=cam_location.replace(' ', '_'))
            row = {
                'timestamp': datetime.now().isoformat(),
                'camera': cam_location,
                'label': label,
                'video_conf': float(conf),
                'audio_score': float(audio_score),
                'fused_score': float(fused),
                'image_path': img_path,
                'clip_path': clip_path
            }
            log_event_row(row)

            pdf_path = None
            if enable_pdf:
                try:
                    pdf_path = create_pdf_report(
                        row, img_path,
                        os.path.join(REPORTS_DIR, os.path.basename(img_path).replace('.jpg', '.pdf'))
                    )
                except Exception as e:
                    st.warning(f"PDF creation failed: {e}")

            # Notifications
            subject = f"ALERT: Violence Detected at {cam_location}"
            body = f"Time: {row['timestamp']}\nCamera: {cam_location}\nFused Score: {fused:.2f}"

            if use_email and smtp_user and smtp_password and email_recipient:
                attachments = [p for p in [img_path, pdf_path] if p]
                send_email_alert(
                    smtp_server, int(smtp_port), smtp_user, smtp_password,
                    email_recipient, subject, body, attachments
                )

            if use_telegram and telegram_token and telegram_chat:
                send_telegram_alert(telegram_token, telegram_chat, body, photo_path=img_path)

            if use_twilio and twilio_sid and twilio_token and twilio_from and twilio_to:
                send_twilio_sms(twilio_sid, twilio_token, twilio_from, twilio_to, subject + " — " + body)

            # ✅ TTS speaks ONLY when violence is detected
            if enable_tts:
                tts_msg = f"Sir, I've detected unusual activity in {cam_location}."
                speak_alert(tts_msg)

        else:
            # ✅ No violence detected — silent mode
            st.success("✅ No violence detected in the uploaded image.")


# ========================
# MODE: VIDEO UPLOAD
# ========================
elif mode == "🎬 Video Upload":
    uploaded_video = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())
        temp_video.close()

        st.video(temp_video.name)

        cap = cv2.VideoCapture(temp_video.name)
        frames = []
        success, frame = cap.read()
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success, frame = cap.read()
        cap.release()

        if len(frames) == 0:
            st.error("❌ No frames found in video.")
        else:
            sampled = sample_frames(frames, k=FRAMES_PER_VIDEO)
            label, conf = predict_from_frames(sampled)

            # Extract and analyze audio
            audio_score = 0.0
            if VideoFileClip is not None and librosa is not None:
                try:
                    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    tmp_audio.close()
                    extract_audio_from_video(temp_video.name, tmp_audio.name)
                    audio_score = analyze_audio_heuristic(tmp_audio.name)
                    os.remove(tmp_audio.name)
                except Exception as e:
                    st.warning(f"Audio extraction/analysis failed: {e}")

            fused = (0.7 * conf) + (0.3 * audio_score)

            st.markdown(f"### Prediction: {label} (Video Conf: {conf:.2f}, AudioScore: {audio_score:.2f}, Fused: {fused:.2f})")

            # 🚨 Violence detection logic
            if fused >= FUSED_THRESHOLD and label == "Violence":
                alert_box = st.empty()
                alert_box.markdown("""
                    <div class='alert-box'>
                        🚨 <span class='alert-text'>VIOLENCE DETECTED!</span> ⚠
                    </div>
                    """, unsafe_allow_html=True)

                img_path, clip_path = save_detection_artifacts(sampled, image_prefix=cam_location.replace(' ', '_'))
                row = {
                    'timestamp': datetime.now().isoformat(),
                    'camera': cam_location,
                    'label': label,
                    'video_conf': float(conf),
                    'audio_score': float(audio_score),
                    'fused_score': float(fused),
                    'image_path': img_path,
                    'clip_path': clip_path
                }
                log_event_row(row)

                pdf_path = None
                if enable_pdf:
                    try:
                        pdf_path = create_pdf_report(
                            row, img_path,
                            os.path.join(REPORTS_DIR, os.path.basename(img_path).replace('.jpg', '.pdf'))
                        )
                    except Exception as e:
                        st.warning(f"PDF creation failed: {e}")

                # Notifications
                subject = f"ALERT: Violence Detected at {cam_location}"
                body = f"Time: {row['timestamp']}\nCamera: {cam_location}\nFused Score: {fused:.2f}"

                if use_email and smtp_user and smtp_password and email_recipient:
                    attachments = [p for p in [img_path, pdf_path] if p]
                    send_email_alert(
                        smtp_server, int(smtp_port), smtp_user, smtp_password,
                        email_recipient, subject, body, attachments
                    )

                if use_telegram and telegram_token and telegram_chat:
                    send_telegram_alert(telegram_token, telegram_chat, body, photo_path=img_path)

                if use_twilio and twilio_sid and twilio_token and twilio_from and twilio_to:
                    send_twilio_sms(twilio_sid, twilio_token, twilio_from, twilio_to, subject + " — " + body)

                # ✅ TTS: speak only when violence is detected
                if enable_tts:
                    tts_msg = f"Sir, I've detected unusual activity in {cam_location}."
                    speak_alert(tts_msg)

            else:
                # ✅ No violence detected — silent mode
                st.success(" No violence detected in the uploaded video.")

        try:
            os.remove(temp_video.name)
        except PermissionError:
            st.warning("⚠ Temporary file still in use. Skipping deletion for now.")
# ========================
# EVENT LOG UI & EXPORT
# ========================
st.markdown("---")
st.header("Event Log & Reports")
log_df = load_event_log()
st.dataframe(log_df.sort_values(by='timestamp', ascending=False).reset_index(drop=True))

# download CSV
if not log_df.empty:
    csv = log_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Event Log (CSV)", data=csv, file_name=EVENT_LOG, mime='text/csv')

# list generated PDF reports and allow download
pdfs = [os.path.join(REPORTS_DIR, f) for f in os.listdir(REPORTS_DIR) if f.lower().endswith('.pdf')]
if pdfs:
    st.write("Generated PDF Reports:")
    for p in sorted(pdfs, reverse=True):
        bname = os.path.basename(p)
        with open(p, 'rb') as f:
            st.download_button(label=f"Download {bname}", data=f.read(), file_name=bname, mime='application/pdf')

st.info("Tip: Supply notification credentials in the sidebar to enable Email/Telegram/SMS alerts. For production, store secrets in Streamlit secrets or environment variables, not typed into app.")

# End of app