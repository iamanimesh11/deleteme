from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
from deepface import DeepFace

class EmotionVideoProcessor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 3)

        for x, y, w, h in faces:
            try:
                result = DeepFace.analyze(frm[y:y + h, x:x + w], actions=['emotion'])
                dominant_emotion = result[0]['dominant_emotion']

                emotion_text = f"Emotion: {dominant_emotion}"
                font =  cv2.FONT_HERSHEY_SIMPLEX
                # Add text shadow
                shadow_offset = 2
                shadow_color = (0, 0, 0)  # Black shadow color
                cv2.putText(frm, emotion_text, (x + w + 10+shadow_offset, y + h // 2+shadow_offset), font, 0.8,shadow_color,  2, cv2.LINE_AA)
            except ValueError as e:
                error_message = "Face not detected :("
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frm, error_message, (x, y - 20), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frm, "Please check your position", (x, y + h + 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(frm, format='bgr24')

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionVideoProcessor,
    rtc_configuration=rtc_configuration,
    # Remove the AudioProcessor
    video_transformer_factory=None,
)
