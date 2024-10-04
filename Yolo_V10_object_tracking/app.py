import os
import base64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np
import faiss
import cv2
import streamlit as st
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
from draw_utils import draw_label, draw_rounded_rectangle, draw_text, get_box_details
import time
import datetime
import tempfile
import pandas as pd
import uuid
import threading
import torch

# Pastikan PyTorch mendeteksi GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Inisialisasi model YOLO dan perangkat
model = YOLO('yolov10n.pt')
deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'

# Inisialisasi DeepSort tracker
tracker = DeepSort(deep_sort_weights)

details = []
prev_details = {}
unique_track_ids = set()
frame_no = 0

# Buat folder untuk menyimpan indeks FAISS
faiss_index_folder = 'faiss_index'
os.makedirs(faiss_index_folder, exist_ok=True)

# Inisialisasi FAISS index
d = 2048  # Dimensi dari face embeddings
index = faiss.IndexFlatL2(d)
face_encodings = []
face_ids = []
person = {}

# Load FAISS index jika ada
faiss_index_file = os.path.join(faiss_index_folder, 'faiss_index.json')
if os.path.exists(faiss_index_file):
    with open(faiss_index_file, 'r') as f:
        data = json.load(f)
        face_encodings = data['encodings']
        face_ids = data['ids']
        
        for idx, ids in enumerate(face_ids):
            if ids not in person:
                person[ids] = {'encoding': [face_encodings[idx]]}
            else:
                person[ids]['encoding'].append(face_encodings[idx])
                
        if len(face_encodings) > 0 and len(face_encodings[0]) == d:
            index.add(np.array(face_encodings, dtype=np.float32))

def generate_random_features():
    return np.random.rand(d).astype(np.float32)

def smooth_ids(track_id, face_encoding):
    D, indices = index.search(np.array([face_encoding]), 1)
    if len(indices[0]) > 0 and D[0][0] < 0.5:
        return face_ids[indices[0][0]]
    else:
        face_encodings.append(face_encoding.tolist())
        face_ids.append(track_id)
        index.add(np.array([face_encoding], dtype=np.float32))
        
        if track_id not in person:
            person[track_id] = {'encoding': [face_encoding.tolist()]}
        else:
            person[track_id]['encoding'].append(face_encoding.tolist())

        with open(faiss_index_file, 'w') as f:
            json.dump({'encodings': face_encodings, 'ids': face_ids, 'person': person}, f)

        return track_id

def is_valid_face(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        return False
    face = image[y1:y2, x1:x2]
    return face.size != 0

def track_video(frame, model, detection_threshold, tracker, frame_no):
    og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = og_frame.copy()

    results = model(frame)
    bboxes_xywh, confs = [], []

    class_names = list(model.names.values())
    cls, xyxy, conf, xywh = get_box_details(results[0].boxes)

    for c, b, co in zip(cls, xywh, conf.cpu().numpy()):
        if class_names[int(c)] == "person" and co >= detection_threshold:
            bboxes_xywh.append(b.cpu().numpy())
            confs.append(co)

    bboxes_xywh = np.array(bboxes_xywh, dtype=float)
    object_counts = 0  # Initialize object count

    if len(bboxes_xywh) >= 1:
        tracks = tracker.update(bboxes_xywh, confs, og_frame)
        
        ids = []
        for track in tracker.tracker.tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_tlbr()
            w = x2 - x1
            h = y2 - y1

            if not is_valid_face(og_frame, (x1, y1, x2, y2)):
                continue

            face_encoding = generate_random_features()
            track_id = smooth_ids(track_id, face_encoding)

            blue_color = (0, 0, 255)
            red_color = (255, 0, 0)
            green_color = (0, 255, 0)

            color_id = track_id % 3
            if color_id == 0:
                color = red_color
                color_name = 'Red'
            elif color_id == 1:
                color = blue_color
                color_name = 'Blue'
            else:
                color = green_color
                color_name = 'Green'

            draw_rounded_rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 1, 15)

            text_color = (255, 255, 255)
            draw_label(og_frame, f"person-{track_id}", (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, text_color)
            
            if track_id not in prev_details:
                prev_details[track_id] = [time.time(), color_name]           

            unique_track_ids.add(track_id)
            ids.append(track_id)

        object_counts = len(ids)  # Update object count to the number of detected persons

        prev_ids = list(prev_details.keys())
        ids_done = set(prev_ids)^set(ids)

        for id in ids_done:
            details.append(["person", id, time.time() - prev_details[id][0], prev_details[id][1], frame_no-1])
            del prev_details[id]
                        
        og_frame = cv2.cvtColor(og_frame, cv2.COLOR_BGR2RGB)
        og_frame = cv2.resize(og_frame, (700, 600))

        font_color = (255, 255, 255)
        position = (10, 30)
        background_color = (0, 0, 0)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        text = f'Frame: {frame_no} | Time: {timestamp} | Count: {object_counts}'
        draw_text(og_frame, text, position, background_color, font_color)

        return og_frame, object_counts
    
    else:
        og_frame = cv2.cvtColor(og_frame, cv2.COLOR_BGR2RGB)
        return og_frame, 0 

st.title('Track your video here')

st.markdown(
    """
        <style>
            [data-testid="stSidebar"][area-expanded="true"] > div:first-child{
                width: 350px
            }
            [data-testid="stSidebar"][area-expanded="false"] > div:first-child{
                width: 350px
                margin-left: -350px
            }
        </style>

    """,
    unsafe_allow_html=True
)
  
st.sidebar.title("Tracker")
st.sidebar.subheader("Options")

@st.cache_data()
def frame_resize(frame, width=None, height=None, inter_=cv2.INTER_AREA):
    dim = None
    (h, w) = frame.shape[:2]
    
    if width is None and height is None:
        return frame
    
    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))

    resized_frame = cv2.resize(frame, dim, interpolation=inter_)

    return resized_frame

@st.cache_data()
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


app_mode = st.sidebar.selectbox('Please select', ['Run on local video', 'Run on live feed'])

detection_threshold = st.sidebar.slider('Select value for detection threshold', min_value=0.1, max_value=1.0, value=0.5)
max_iou_distance = st.sidebar.slider('Select value for max iou distance', min_value=0.1, max_value=1.0, value=0.5)
min_confidence = st.sidebar.slider('Select value for min confidence', min_value=0.1, max_value=1.0, value=0.3)
max_distance = st.sidebar.slider('Select value for max distance', min_value=0.1, max_value=1.0, value=0.2)

tracker = DeepSort(model_path=deep_sort_weights, max_age=70, n_init=5, max_iou_distance=0.8, min_confidence=min_confidence, 
                max_dist=max_distance)

if app_mode == 'Run on local video':
    st.markdown(
        """
            <style>
                [data-testid="stSidebar"][area-expanded="true"] > div:first-child{
                    width: 350px
                }
                [data-testid="stSidebar"][area-expanded="false"] > div:first-child{
                    width: 350px
                    margin-left: -350px
                }
            </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('Output')
    stframe = st.empty()

    st.sidebar.markdown('---')
    video_file = st.sidebar.file_uploader('Upload your video file here', type = ['mp4', 'mov', 'avi', 'm4v'])
    t_file = tempfile.NamedTemporaryFile(delete=False)

    record = st.sidebar.checkbox('Record Video')

    if record:
        st.checkbox('Reording', value=True)

    if not video_file:
        st.error('Video not found')
    else:
        prev_time = time.time()  # Initialize prev_time here
        frame_no = 0
        t_file.write(video_file.read())
        cap = cv2.VideoCapture(t_file.name)        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not cap.isOpened():
            st.error("Error: Could not open video source.")
            print("Error: Could not open video source.")
        else:
            st.sidebar.text('Uploaded Video')
            st.sidebar.video(t_file.name)

            kpi1, kpi2, kpi3 = st.columns(3)

            with kpi1:
                st.markdown('**Frame rate**')
                kpi1_text = st.markdown("0")

            with kpi2:
                st.markdown('**Object Counts**')
                kpi2_text = st.markdown("0")

            with kpi3:
                st.markdown('**Frame Width**')
                kpi3_text = st.markdown("0")
            
            st.markdown('<hr/>', unsafe_allow_html=True)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame, object_count = track_video(frame, model, detection_threshold, tracker, frame_no)

                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time

                kpi1_text.write(f"<h1 style=' color: red;'> {int(fps)} </h1>", unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style=' color: red;'> {int(object_count)} </h1>", unsafe_allow_html=True)
                kpi3_text.write(f"<h1 style=' color: red;'> {int(frame_width)} </h1>", unsafe_allow_html=True)

                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Reduce resolution
                frame = frame_resize(frame=frame, width=1080, height=720)
                stframe.image(frame, channels='BGR', use_column_width=True)

                frame_no += 1

            cap.release()

        st.text('Video Processed')
        st.markdown('---')
        
        details = pd.DataFrame(details, columns = ['Object', 'id', 'duration', 'color', 'frame_number'])
        st.dataframe(details)

        csv = convert_df(details)
        name = uuid.uuid1()  # Define 'name' here
        st.markdown(f'<a href="data:text/csv;charset=utf-8,{csv.decode()}" download="{name}.csv">Download CSV</a>', unsafe_allow_html=True)

        st.markdown('<script>document.querySelector("a").click();</script>', unsafe_allow_html=True)

        st.toast('Details file downloaded!!')

elif app_mode == 'Run on live feed':
    st.markdown(
        """
            <style>
                [data-testid="stSidebar"][area-expanded="true"] > div:first-child{
                    width: 350px
                }
                [data-testid="stSidebar"][area-expanded="false"] > div:first-child{
                    width: 350px
                    margin-left: -350px
                }
            </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('Output')
    stframe = st.empty()

    st.sidebar.markdown('---')
    video_file = st.sidebar.file_uploader('Upload your video file here', type = ['mp4', 'mov', 'avi', 'm4v'])
    t_file = tempfile.NamedTemporaryFile(delete=False)

    record = st.sidebar.checkbox('Record Video')

    if record:
        st.checkbox('Reording', value=True)

    video_device = st.selectbox('Select device', options=[0, 1])
    print(video_device)

    cap = cv2.VideoCapture(int(video_device))       

    if not cap.isOpened():
        st.error("Error: Could not open video source.")
        print("Error: Could not open video source.")
    else:
        prev_time = time.time()  # Initialize prev_time here
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown('**Frame rate**')
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown('**Object Counts**')
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown('**Frame Width**')
            kpi3_text = st.markdown("0")

        st.markdown('<hr/>', unsafe_allow_html=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, object_count = track_video(frame, model, detection_threshold, tracker, frame_no)

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            kpi1_text.write(f"<h1 style=' color: red;'> {int(fps)} </h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style=' color: red;'> {int(object_count)} </h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style=' color: red;'> {int(frame_width)} </h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Reduce resolution
            frame = frame_resize(frame=frame, width=1080, height=720)
            stframe.image(frame, channels='BGR', use_column_width=True)

            frame_no += 1

        cap.release()

    st.text('Video Processed')
    st.markdown('---')
    
    details = pd.DataFrame(details, columns = ['Object', 'id', 'duration', 'color', 'frame_number'])
    st.dataframe(details)

    csv = convert_df(details)
    name = uuid.uuid1()  # Define 'name' here
    st.markdown(f'<a href="data:text/csv;charset=utf-8,{csv.decode()}" download="{name}.csv">Download CSV</a>', unsafe_allow_html=True)

# Save FAISS data to a JSON file
faiss_index_path = 'faiss_index'
os.makedirs(faiss_index_path, exist_ok=True)

# Convert and save FAISS index to the desired format
converted_data = {i: {"encodings": [encoding]} for i, encoding in enumerate(face_encodings, start=1)}