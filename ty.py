import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading
import cv2
from deepface import DeepFace

class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frm = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 3)

        for x, y, w, h in faces:
            try:
                result = DeepFace.analyze(frm[y:y + h, x:x + w], actions=['emotion'])
                dominant_emotion = result[0]['dominant_emotion']

                emotion_text = f"Emotion: {dominant_emotion}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                shadow_offset = 2
                shadow_color = (0, 0, 0)
                cv2.putText(frm, emotion_text, (x + w + 10 + shadow_offset, y + h // 2 + shadow_offset), font, 0.8, shadow_color, 2, cv2.LINE_AA)
            except ValueError as e:
                error_message = "Face not detected :("
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frm, error_message, (x, y - 20), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frm, "Please check your position", (x, y + h + 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(frm, format='bgr24')

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Streamlit WebRTC Demo", page_icon="🤖")
task_list = ["Video Stream"]

with st.sidebar:
    st.title('Task Selection')
    task_name = st.selectbox("Select your tasks:", task_list)
st.title(task_name)

if task_name == task_list[0]:
    style_list = ['color', 'black and white']

    st.sidebar.header('Style Selection')
    style_selection = st.sidebar.selectbox("Choose your style:", style_list)

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model_lock = threading.Lock()
            self.style = style_list[0]

        def update_style(self, new_style):
            if self.style != new_style:
                with self.model_lock:
                    self.style = new_style

        def recv(self, frame):
            # img = frame.to_ndarray(format="bgr24")
            img = frame.to_image()
            if self.style == style_list[1]:
                img = img.convert("L")

            # return av.VideoFrame.from_ndarray(img, format="bgr24")
            return av.VideoFrame.from_image(img)

    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

    if ctx.video_processor:
        ctx.video_transformer.update_style(style_selection)
st.set_page_config(page_title="Streamlit WebRTC Emotion Detection", page_icon="🎭")

webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)
