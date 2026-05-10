import os, io, time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import cv2
import imutils
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (Callback, ModelCheckpoint,
                                        EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Flatten, Input, MaxPooling2D,
                                     ZeroPadding2D, Dropout)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, matthews_corrcoef,
                             balanced_accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from os import listdir
import warnings
warnings.filterwarnings('ignore')

YES_PATH       = 'yes'
NO_PATH        = 'no'
AUGMENTED_PATH = 'augmented_data/'
MODELS_DIR     = 'models/'
IMG_WIDTH      = 240
IMG_HEIGHT     = 240
IMG_SHAPE      = (IMG_WIDTH, IMG_HEIGHT, 3)

os.makedirs(AUGMENTED_PATH + 'yes', exist_ok=True)
os.makedirs(AUGMENTED_PATH + 'no',  exist_ok=True)
os.makedirs(MODELS_DIR,             exist_ok=True)

def apply_custom_css():
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif !important;
    }
    #MainMenu, footer { visibility: hidden; }
    .main, .stApp { background: #0d1117 !important; }
    [data-testid="stSidebar"] {
        background: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    h1 { color: #e6edf3 !important; font-size: 2rem !important; font-weight: 700 !important; }
    h2 { color: #c9d1d9 !important; font-size: 1.25rem !important; font-weight: 600 !important; }
    h3 { color: #c9d1d9 !important; font-size: 1rem !important; font-weight: 600 !important; }
    p, li { color: #8b949e !important; font-size: 0.9rem !important; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span { color: #8b949e !important; }

    .stat-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 1.1rem 1rem;
        text-align: center; border-top: 3px solid #1f6feb;
    }
    .stat-value { font-size: 1.9rem; font-weight: 700; color: #e6edf3; line-height: 1; }
    .stat-label { font-size: 0.72rem; text-transform: uppercase;
                  letter-spacing: 0.08em; color: #6e7681; margin-top: 0.35rem; }

    .box { border-radius: 8px; padding: 0.85rem 1rem;
           margin: 0.6rem 0; border-left: 4px solid; }
    .box p { margin: 0 !important; font-size: 0.88rem !important; }
    .box-info    { background:#0d2137; border-color:#1f6feb; }
    .box-info p  { color:#79c0ff !important; }
    .box-success { background:#0f2a1e; border-color:#238636; }
    .box-success p { color:#56d364 !important; }
    .box-warn    { background:#2a1e0f; border-color:#9e6a03; }
    .box-warn p  { color:#e3b341 !important; }
    .box-danger  { background:#2a0f0f; border-color:#da3633; }
    .box-danger p { color:#ff7b72 !important; }

    .result-card { border-radius: 12px; padding: 1.6rem; text-align: center; border: 2px solid; }
    .result-pos  { background:#0f2a1e; border-color:#238636; }
    .result-neg  { background:#2a0f0f; border-color:#da3633; }
    .result-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.3rem; }
    .result-pos .result-title { color: #56d364; }
    .result-neg .result-title { color: #ff7b72; }
    .result-conf  { font-size: 2.6rem; font-weight: 700; }
    .result-pos .result-conf { color: #3fb950; }
    .result-neg .result-conf { color: #f85149; }
    .result-sub   { font-size: 0.75rem; text-transform: uppercase;
                    letter-spacing: 0.1em; color: #6e7681; margin-top: 0.4rem; }

    .section-lbl {
        font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.14em; color: #6e7681;
        display: flex; align-items: center; gap: 0.6rem; margin: 1.4rem 0 0.8rem;
    }
    .section-lbl::before, .section-lbl::after {
        content: ''; flex: 1; height: 1px; background: #21262d;
    }

    .stButton > button {
        background: #1f6feb !important; color: #fff !important;
        border: none !important; border-radius: 6px !important;
        font-weight: 600 !important; font-size: 0.88rem !important;
        padding: 0.55rem 1.1rem !important; transition: background 0.2s !important;
    }
    .stButton > button:hover { background: #388bfd !important; }

    .stTabs [data-baseweb="tab-list"] {
        background: #161b22; border-radius: 8px; padding: 3px;
        gap: 3px; border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #8b949e; border-radius: 6px;
        padding: 0.45rem 1rem; font-size: 0.85rem; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #21262d !important; color: #e6edf3 !important;
    }

    .stProgress > div > div > div > div { background: #1f6feb !important; }

    .streamlit-expanderHeader {
        background: #161b22 !important; border: 1px solid #30363d !important;
        border-radius: 8px !important; color: #c9d1d9 !important; font-weight: 600 !important;
    }

    [data-testid="metric-container"] {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 0.9rem;
    }
    [data-testid="stMetricValue"]  { color: #e6edf3 !important; }
    [data-testid="stMetricLabel"] span {
        color: #6e7681 !important; font-size: 0.72rem !important;
        text-transform: uppercase; letter-spacing: 0.08em;
    }

    [data-testid="stFileUploader"] {
        background: #161b22 !important; border: 2px dashed #30363d !important;
        border-radius: 10px !important;
    }

    ::-webkit-scrollbar { width:5px; height:5px; }
    ::-webkit-scrollbar-track { background:#0d1117; }
    ::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }

    .stRadio label, .stSelectbox label, .stSlider label,
    .stCheckbox label, .stNumberInput label {
        color: #8b949e !important; font-size: 0.82rem !important; font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    plt.rcParams.update({
        'figure.facecolor': '#161b22', 'axes.facecolor': '#0d1117',
        'axes.edgecolor':   '#30363d', 'axes.labelcolor': '#8b949e',
        'text.color':       '#8b949e', 'xtick.color': '#6e7681',
        'ytick.color':      '#6e7681', 'grid.color': '#21262d',
        'grid.linewidth':   0.8,       'axes.titlecolor': '#e6edf3',
        'axes.titlesize':   11,        'axes.labelsize': 9,
        'figure.dpi':       110,
    })



def hms_string(sec):
    h=int(sec/3600); m=int((sec%3600)/60); s=sec%60
    return f"{h}h {m}m {round(s,1)}s"

def section(text):
    st.markdown(f'<div class="section-lbl">{text}</div>', unsafe_allow_html=True)

def box(icon, text, kind="info"):
    st.markdown(f'<div class="box box-{kind}"><p>{icon} {text}</p></div>', unsafe_allow_html=True)

def stat_card(col, value, label):
    col.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
    </div>""", unsafe_allow_html=True)


def crop_brain_contour(image):
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray   = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts   = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts   = imutils.grab_contours(cnts)
    if not cnts: return image
    c = max(cnts, key=cv2.contourArea)
    extLeft  = tuple(c[c[:,:,0].argmin()][0])
    extRight = tuple(c[c[:,:,0].argmax()][0])
    extTop   = tuple(c[c[:,:,1].argmin()][0])
    extBot   = tuple(c[c[:,:,1].argmax()][0])
    return image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

def augment_data(file_dir, n_generated_samples, save_to_dir, log_fn=print):
    data_gen = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, brightness_range=(0.3,1.0),
        horizontal_flip=True, vertical_flip=True, fill_mode='nearest'
    )
    files = [f for f in listdir(file_dir) if cv2.imread(os.path.join(file_dir,f)) is not None]
    for idx, filename in enumerate(files):
        image = cv2.imread(os.path.join(file_dir, filename))
        if image is None: continue
        image = image.reshape((1,)+image.shape)
        save_prefix = 'aug_'+os.path.splitext(filename)[0]
        i = 0
        for _ in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir,
                               save_prefix=save_prefix, save_format='jpg'):
            i += 1
            if i > n_generated_samples: break
        log_fn(f"  Augmented {idx+1}/{len(files)}: {filename}")

def load_data(dir_list, image_size, log_fn=print):
    X, y, filenames = [], [], []
    iw, ih = image_size
    for directory in dir_list:
        label = 1 if directory.rstrip('/\\').endswith('yes') else 0
        for filename in listdir(directory):
            img = cv2.imread(os.path.join(directory, filename))
            if img is None: continue
            try:    img = crop_brain_contour(img)
            except: continue
            img = cv2.resize(img, (iw,ih), interpolation=cv2.INTER_CUBIC)
            img = img / 255.0
            X.append(img); y.append([label]); filenames.append(filename)
    X = np.array(X); y = np.array(y)
    X, y, filenames = shuffle(X, y, filenames, random_state=42)
    log_fn(f"Loaded {len(X)} images.")
    return X, y, filenames

def split_data(X, y, test_size=0.3):
    X_train,X_tv,y_train,y_tv = train_test_split(X, y, test_size=test_size, random_state=42)
    X_test,X_val,y_test,y_val = train_test_split(X_tv, y_tv, test_size=0.5, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_model(input_shape, use_dropout=True):
    Xi = Input(input_shape)
    X  = ZeroPadding2D((2,2))(Xi)
    X  = Conv2D(32,(7,7), strides=(1,1), name='conv0')(X)
    X  = BatchNormalization(axis=3, name='bn0')(X); X = Activation('relu')(X)
    X  = MaxPooling2D((4,4), name='max_pool0')(X)
    X  = Conv2D(64,(5,5), padding='same', name='conv1')(X)
    X  = BatchNormalization(axis=3, name='bn1')(X); X = Activation('relu')(X)
    X  = MaxPooling2D((2,2), name='max_pool1')(X)
    X  = Conv2D(128,(3,3), padding='same', name='conv2')(X)
    X  = BatchNormalization(axis=3, name='bn2')(X); X = Activation('relu')(X)
    X  = MaxPooling2D((2,2), name='max_pool2')(X)
    X  = Flatten()(X)
    if use_dropout: X = Dropout(0.5)(X)
    X  = Dense(256, activation='relu', name='fc1')(X)
    if use_dropout: X = Dropout(0.3)(X)
    X  = Dense(1, activation='sigmoid', name='fc_out')(X)
    return Model(inputs=Xi, outputs=X, name='BrainTumorDetector')

def preprocess_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cropped    = crop_brain_contour(image)
    cropped_rgb= cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    resized    = cv2.resize(cropped, (IMG_WIDTH,IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    normalised = resized / 255.0
    return original, cropped_rgb, normalised

def analyze_image_quality(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    sharpness  = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    contrast   = gray.std()
    score = min(100, int(
        min(sharpness/500,1)*40 + min(brightness/128,1)*30 + min(contrast/60,1)*30
    ))
    grade = ('Excellent' if score>=80 else 'Good' if score>=60 else
             'Fair'      if score>=40 else 'Poor')
    return {'sharpness':round(sharpness,1),'brightness':round(brightness,1),
            'contrast':round(contrast,1),'score':score,'grade':grade}



def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2"):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(np.expand_dims(img_array, axis=0))
        class_channel   = preds[:, 0]
    grads   = tape.gradient(class_channel, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap,0) / (tf.math.reduce_max(heatmap)+1e-8)
    return heatmap.numpy()

def overlay_gradcam(img_rgb, heatmap, alpha=0.45):
    img = np.array(img_rgb).astype(np.float32)
    h   = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    h   = np.uint8(255*h)
    hc  = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    hc  = cv2.cvtColor(hc, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img.astype(np.uint8), alpha, hc, 1-alpha, 0)


class StreamlitCallback(Callback):
    def __init__(self, total_epochs, progress_bar, status_text, chart_ph):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = progress_bar
        self.status_text  = status_text
        self.chart_ph     = chart_ph
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.progress_bar.progress((epoch + 1) / self.total_epochs)
        
   
        ak = 'accuracy' if 'accuracy' in logs else 'acc'
        vak = 'val_accuracy' if 'val_accuracy' in logs else 'val_acc'
        
        # Store history
        self.history['loss'].append(logs.get('loss', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))
        self.history['accuracy'].append(logs.get(ak, 0))
        self.history['val_accuracy'].append(logs.get(vak, 0))

        # Update status text
        self.status_text.info(
            f"**Epoch {epoch+1}/{self.total_epochs}** | "
            f"Loss: {logs.get('loss', 0):.4f} | Acc: {logs.get(ak, 0):.4f} | "
            f"Val Loss: {logs.get('val_loss', 0):.4f} | Val Acc: {logs.get(vak, 0):.4f}"
        )
        
        # plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.history['loss'], color='#1f6feb', linewidth=2, label='Train Loss')
        axes[0].plot(self.history['val_loss'], color='#3fb950', linewidth=2, linestyle='--', label='Val Loss')
        axes[0].set_title('Loss', fontsize=12, color='#e6edf3')
        axes[0].set_xlabel('Epoch', color='#8b949e')
        axes[0].set_ylabel('Loss', color='#8b949e')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_facecolor('#0d1117')
        
        # Accuracy plot
        axes[1].plot(self.history['accuracy'], color='#1f6feb', linewidth=2, label='Train Acc')
        axes[1].plot(self.history['val_accuracy'], color='#3fb950', linewidth=2, linestyle='--', label='Val Acc')
        axes[1].set_title('Accuracy', fontsize=12, color='#e6edf3')
        axes[1].set_xlabel('Epoch', color='#8b949e')
        axes[1].set_ylabel('Accuracy', color='#8b949e')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_facecolor('#0d1117')
        
        plt.tight_layout()
        self.chart_ph.pyplot(fig)
        plt.close(fig)



def page_dashboard():
    st.markdown("## 🧠 Brain Tumor Detection System")
    box("ℹ️",
        "A Convolutional Neural Network for detecting brain tumors from MRI scans. "
        "Upload a scan, run inference, and get a confidence-scored result with Grad-CAM attention maps.",
        "info")

    # KPI row
    available_models = len([f for f in os.listdir(MODELS_DIR) if f.endswith('.keras')])
    dataset_yes = (len([f for f in listdir(YES_PATH) if f.endswith(('.jpg','.png','.jpeg'))])
                   if os.path.isdir(YES_PATH) else 0)
    dataset_no  = (len([f for f in listdir(NO_PATH)  if f.endswith(('.jpg','.png','.jpeg'))])
                   if os.path.isdir(NO_PATH)  else 0)
    total_imgs  = dataset_yes + dataset_no
    total_preds = len(st.session_state.get('prediction_history',[]))

    c1,c2,c3,c4,c5 = st.columns(5)
    stat_card(c1,"95%+","Target Accuracy")
    stat_card(c2,"<1s","Inference Time")
    stat_card(c3,str(available_models),"Trained Models")
    stat_card(c4,str(total_imgs),"Dataset Images")
    stat_card(c5,str(total_preds),"Predictions Made")

    # Dataset charts
    section("DATASET OVERVIEW")

    if total_imgs > 0:
        col_pie, col_bar, col_gauge = st.columns(3)

        with col_pie:
            fig_pie = go.Figure(go.Pie(
                labels=['Tumorous','Non-Tumorous'], values=[dataset_yes, dataset_no],
                hole=0.55, marker_colors=['#f85149','#3fb950'],
                textinfo='percent+label', textfont_size=11,
            ))
            fig_pie.update_layout(
                title=dict(text='Class Distribution', font=dict(color='#c9d1d9',size=12)),
                paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=40,b=10,l=10,r=10),
                legend=dict(font=dict(color='#8b949e',size=10)), height=260, showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            fig_bar = go.Figure(go.Bar(
                y=['Tumorous','Non-Tumorous'], x=[dataset_yes, dataset_no],
                orientation='h', marker_color=['#f85149','#3fb950'],
                text=[dataset_yes, dataset_no], textposition='outside',
                textfont=dict(color='#c9d1d9',size=11),
            ))
            fig_bar.update_layout(
                title=dict(text='Image Count', font=dict(color='#c9d1d9',size=12)),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True,gridcolor='#21262d',color='#6e7681'),
                yaxis=dict(showgrid=False,color='#8b949e'),
                margin=dict(t=40,b=20,l=10,r=50), height=260,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_gauge:
            balance = dataset_yes / max(total_imgs,1) * 100
            bar_color = '#f85149' if balance>60 else '#e3b341' if balance>40 else '#3fb950'
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=balance,
                title={'text':"Tumor Ratio (%)","font":{'color':'#8b949e','size':11}},
                gauge={
                    'axis':{'range':[0,100],'tickcolor':'#6e7681','tickfont':{'color':'#6e7681','size':9}},
                    'bar':{'color':bar_color},
                    'bgcolor':'#0d1117','bordercolor':'#30363d',
                    'steps':[
                        {'range':[0,40],  'color':'rgba(63,185,80,0.1)'},
                        {'range':[40,60], 'color':'rgba(227,179,65,0.1)'},
                        {'range':[60,100],'color':'rgba(248,81,73,0.1)'},
                    ],
                    'threshold':{'line':{'color':'#79c0ff','width':2},'thickness':0.75,'value':50},
                },
                number={'font':{'color':'#e6edf3','size':36},'suffix':'%'},
            ))
            fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=260,
                                margin=dict(t=30,b=10,l=20,r=20))
            st.plotly_chart(fig_g, use_container_width=True)

    else:
        box("📁","Dataset not found. Add <code>yes/</code> and <code>no/</code> folders to see visualizations.","warn")

    # Recent predictions
    section("RECENT PREDICTIONS")
    history = st.session_state.get('prediction_history',[])
    if history:
        recent = history[-8:][::-1]
        for item in recent:
            is_t = item['prediction'] == 'Tumor Detected'
            dot  = "🔴" if is_t else "🟢"
            st.markdown(
                f"{dot} &nbsp; **{item['prediction']}** &nbsp;·&nbsp; "
                f"<span style='color:#6e7681;font-size:0.82rem;'>{item['timestamp']}</span>"
                f" &nbsp;·&nbsp; <span style='color:#1f6feb;font-size:0.82rem;'>conf: {item['confidence']}</span>",
                unsafe_allow_html=True
            )
    else:
        box("📭","No predictions yet — head to <strong>Predict</strong> to analyse an MRI scan.","info")

    section("DISCLAIMER")
    box("⚠️",
        "<strong>For research & educational use only.</strong> "
        "Not a certified medical device. Consult a qualified radiologist for clinical diagnosis.",
        "warn")


def page_dataset_explorer():
    st.markdown("## 📊 Dataset Explorer")

    if not (os.path.isdir(YES_PATH) and os.path.isdir(NO_PATH)):
        box("❌","Dataset not found. Add <code>yes/</code> and <code>no/</code> folders.","danger"); return

    yes_files = [f for f in listdir(YES_PATH) if f.endswith(('.jpg','.png','.jpeg'))]
    no_files  = [f for f in listdir(NO_PATH)  if f.endswith(('.jpg','.png','.jpeg'))]
    total     = len(yes_files) + len(no_files)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Images", total)
    c2.metric("Tumorous",     len(yes_files))
    c3.metric("Non-Tumorous", len(no_files))
    c4.metric("Class Ratio",  f"{len(yes_files)/max(len(no_files),1):.2f}:1")

    tab1, tab2, tab3 = st.tabs(["📊 Distribution","🖼️ Gallery","🔧 Preprocessing"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            fig1 = go.Figure(go.Pie(
                labels=['Tumorous','Non-Tumorous'], values=[len(yes_files),len(no_files)],
                hole=0.5, marker_colors=['#f85149','#3fb950'],
                textinfo='percent+label', textfont_size=12,
            ))
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300,
                               margin=dict(t=30,b=10,l=10,r=10), showlegend=False,
                               title=dict(text='Class Split',font=dict(color='#c9d1d9',size=12)))
            st.plotly_chart(fig1, use_container_width=True)
        with col_b:
            fig2 = go.Figure(go.Bar(
                x=['Tumorous','Non-Tumorous'], y=[len(yes_files),len(no_files)],
                marker_color=['#f85149','#3fb950'],
                text=[len(yes_files),len(no_files)], textposition='outside',
                textfont=dict(color='#c9d1d9'),
            ))
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False,color='#8b949e'),
                yaxis=dict(gridcolor='#21262d',color='#6e7681'),
                height=300, margin=dict(t=30,b=20,l=20,r=20),
                title=dict(text='Image Counts',font=dict(color='#c9d1d9',size=12)),
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("#### Tumorous Samples")
        if yes_files:
            cols = st.columns(4)
            for i,f in enumerate(yes_files[:8]):
                img = cv2.imread(os.path.join(YES_PATH,f))
                if img is not None:
                    cols[i%4].image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),
                                    caption=f"T-{i+1}", use_container_width=True)
        st.markdown("#### Healthy Samples")
        if no_files:
            cols = st.columns(4)
            for i,f in enumerate(no_files[:8]):
                img = cv2.imread(os.path.join(NO_PATH,f))
                if img is not None:
                    cols[i%4].image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),
                                    caption=f"H-{i+1}", use_container_width=True)

    with tab3:
        if yes_files:
            img = cv2.imread(os.path.join(YES_PATH,yes_files[0]))
            if img is not None:
                orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cropped  = crop_brain_contour(img)
                crop_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                resized  = cv2.resize(cropped,(IMG_WIDTH,IMG_HEIGHT))
                c1,c2,c3 = st.columns(3)
                c1.image(orig_rgb, caption="① Original",         use_container_width=True)
                c2.image(crop_rgb, caption="② Brain Extracted",  use_container_width=True)
                c3.image(resized,  caption=f"③ Resized {IMG_WIDTH}×{IMG_HEIGHT}", use_container_width=True)
                box("ℹ️",
                    f"Pipeline: GaussianBlur → Threshold → Erode/Dilate → Contour Crop → "
                    f"Resize {IMG_WIDTH}×{IMG_HEIGHT} → Normalize [0,1]","info")


def page_train():
    st.markdown("## 🏋️ Train Model")

    data_ok = (os.path.isdir(YES_PATH) and os.path.isdir(NO_PATH) and
               len(listdir(YES_PATH))>0 and len(listdir(NO_PATH))>0)
    if not data_ok:
        box("❌","Dataset not found.","danger"); return

    with st.sidebar:
        st.markdown("### ⚙️ Hyperparameters")
        epochs       = st.slider("Epochs",       5, 50, 24)
        batch_size   = st.slider("Batch Size",   8,128, 32, step=8)
        test_size    = st.slider("Test Split %",10, 40, 30) / 100
        use_dropout  = st.checkbox("Dropout",      True)
        use_es       = st.checkbox("Early Stopping",True)
        use_lr_sched = st.checkbox("LR Scheduling", True)

    if st.button("🚀  Start Training", use_container_width=True):
        log_box = st.empty(); logs = []
        def log(m):
            logs.append(m)
            log_box.code("\n".join(logs[-25:]), language="text")

        with st.spinner("Augmenting data…"):
            done = (len(listdir(AUGMENTED_PATH+'yes'))>0 and
                    len(listdir(AUGMENTED_PATH+'no'))>0)
            if done:
                log("✓ Augmented data present — skipping")
            else:
                t0 = time.time()
                augment_data(YES_PATH, 6, AUGMENTED_PATH+'yes', log_fn=log)
                augment_data(NO_PATH,  9, AUGMENTED_PATH+'no',  log_fn=log)
                log(f"✓ Done in {hms_string(time.time()-t0)}")

        with st.spinner("Loading data…"):
            X,y,_ = load_data([AUGMENTED_PATH+'yes',AUGMENTED_PATH+'no'],
                              (IMG_WIDTH,IMG_HEIGHT), log_fn=log)

        n_pos = int(np.sum(y))
        c1,c2,c3 = st.columns(3)
        c1.metric("Total",len(y)); c2.metric("Tumorous",n_pos); c3.metric("Non-Tumorous",len(y)-n_pos)

        X_train,y_train,X_val,y_val,X_test,y_test = split_data(X,y,test_size)
        log(f"✓ Train:{len(X_train)} | Val:{len(X_val)} | Test:{len(X_test)}")

        section("TRAINING PROGRESS")
        prog   = st.progress(0.0)
        status = st.empty()
        charts = st.empty()

        model = build_model(IMG_SHAPE, use_dropout=use_dropout)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        with st.expander("📐 Model Architecture"):
            buf = io.StringIO()
            model.summary(print_fn=lambda x: buf.write(x+'\n'))
            st.code(buf.getvalue(), language="text")

        ckpt_path = os.path.join(MODELS_DIR, f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras')
        cbs = [ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=0),
               StreamlitCallback(epochs, prog, status, charts)]
        if use_es:       cbs.append(EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True))
        if use_lr_sched: cbs.append(ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,min_lr=1e-6))

        t0 = time.time()
        model.fit(X_train,y_train, batch_size=batch_size, epochs=epochs,
                  validation_data=(X_val,y_val), callbacks=cbs, verbose=0)
        log(f"✓ Training done in {hms_string(time.time()-t0)}")

        section("EVALUATION")
        best = load_model(ckpt_path)
        loss,acc = best.evaluate(X_test,y_test,verbose=0)
        y_prob   = best.predict(X_test)
        y_pred   = (y_prob>0.5).astype(int)
        f1_v     = f1_score(y_test,y_pred)
        mcc_v    = matthews_corrcoef(y_test.flatten(),y_pred.flatten())
        bal_acc  = balanced_accuracy_score(y_test.flatten(),y_pred.flatten())
        fpr,tpr,_= roc_curve(y_test.flatten(),y_prob.flatten())
        auc_v    = auc(fpr,tpr)

        cols = st.columns(5)
        for col,lbl,val in zip(cols,
            ["Accuracy","F1 Score","MCC","Balanced Acc","AUC-ROC"],
            [f"{acc*100:.1f}%",f"{f1_v:.4f}",f"{mcc_v:.4f}",f"{bal_acc*100:.1f}%",f"{auc_v:.4f}"]):
            col.metric(lbl,val)

        fig_ev, axes = plt.subplots(1,2, figsize=(12,5))
        cm = confusion_matrix(y_test.flatten(),y_pred.flatten())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['No Tumor','Tumor'], yticklabels=['No Tumor','Tumor'],
                    linewidths=0.5, annot_kws={'size':12,'weight':'bold'})
        axes[0].set_title('Confusion Matrix'); axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

        axes[1].plot(fpr,tpr,color='#1f6feb',linewidth=2,label=f'AUC = {auc_v:.3f}')
        axes[1].plot([0,1],[0,1],color='#6e7681',linestyle='--',linewidth=1)
        axes[1].fill_between(fpr,tpr,alpha=0.1,color='#1f6feb')
        axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve'); axes[1].legend(fontsize=9); axes[1].grid(True,alpha=0.4)
        plt.tight_layout(); st.pyplot(fig_ev); plt.close(fig_ev)

        with st.expander("📋 Classification Report"):
            rpt = classification_report(y_test.flatten(),y_pred.flatten(),
                                        target_names=['No Tumor','Tumor'],output_dict=True)
            st.dataframe(pd.DataFrame(rpt).transpose().round(4), use_container_width=True)

        st.session_state['model_path'] = ckpt_path
        box("✅","Training complete! Go to <strong>Predict</strong> to test on new scans.","success")


def page_predict():
    st.markdown("## 🔬 Predict")

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.keras')]
    if not model_files:
        box("⚠️","No trained model found. Train a model first.","warn"); return

    with st.sidebar:
        st.markdown("### 🎛️ Inference Settings")
        selected_model     = st.selectbox("Model", model_files)
        decision_threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)
        show_gradcam       = st.checkbox("Show Grad-CAM", True)
        show_quality       = st.checkbox("Image Quality Analysis", True)

    @st.cache_resource
    def get_model(path):
        return load_model(path)

    model = get_model(os.path.join(MODELS_DIR, selected_model))
    mode  = st.radio("Mode", ["Single Image","Batch Prediction"], horizontal=True)

    if mode == "Single Image":
        uploaded = st.file_uploader("Upload Brain MRI (JPG / PNG)", type=['jpg','jpeg','png'])

        if uploaded:
            try:
                original, cropped_rgb, normalised = preprocess_uploaded_image(uploaded)
                c1,c2 = st.columns(2)
                c1.image(original,    caption="Original Scan",        use_container_width=True)
                c2.image(cropped_rgb, caption="Extracted Brain Region",use_container_width=True)

                t0       = time.perf_counter()
                raw_prob = float(model.predict(np.expand_dims(normalised,0),verbose=0)[0][0])
                inf_ms   = (time.perf_counter()-t0)*1000
                is_tumor = raw_prob > decision_threshold
                label    = "Tumor Detected" if is_tumor else "No Tumor Detected"
                conf     = raw_prob if is_tumor else 1-raw_prob

                if show_gradcam:
                    try:
                        hm  = make_gradcam_heatmap(normalised, model)
                        gc  = overlay_gradcam(
                                cv2.resize(np.array(cropped_rgb),(IMG_WIDTH,IMG_HEIGHT)), hm)
                        section("GRAD-CAM ATTENTION MAP")
                        _,col_gc,_ = st.columns([1,2,1])
                        col_gc.image(gc, caption="Warmer colours = higher model attention",
                                     use_container_width=True)
                    except Exception as e:
                        box("ℹ️",f"Grad-CAM unavailable: {e}","info")

                if show_quality:
                    section("IMAGE QUALITY")
                    q = analyze_image_quality(original)
                    qc1,qc2,qc3,qc4 = st.columns(4)
                    qc1.metric("Grade",    q['grade'])
                    qc2.metric("Score",    f"{q['score']}/100")
                    qc3.metric("Sharpness",f"{q['sharpness']:.0f}")
                    qc4.metric("Contrast", f"{q['contrast']:.0f}")

                section("DIAGNOSIS RESULT")
                card_class = "result-pos" if not is_tumor else "result-neg"
                icon       = "✅" if not is_tumor else "⚠️"
                _,mid,_    = st.columns([1,2,1])
                mid.markdown(f"""
                <div class="result-card {card_class}">
                    <div style="font-size:2rem;margin-bottom:0.4rem;">{icon}</div>
                    <div class="result-title">{label}</div>
                    <div class="result-conf">{conf*100:.1f}%</div>
                    <div class="result-sub">Confidence · Threshold {decision_threshold:.2f} · {inf_ms:.1f} ms</div>
                </div>
                """, unsafe_allow_html=True)

                section("PROBABILITY GAUGE")
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=raw_prob*100,
                    title={'text':"Tumor Probability (%)","font":{'color':'#8b949e','size':12}},
                    gauge={
                        'axis':{'range':[0,100],'tickcolor':'#6e7681',
                                'tickfont':{'color':'#6e7681','size':9}},
                        'bar':{'color':'#f85149' if is_tumor else '#3fb950'},
                        'bgcolor':'#0d1117','bordercolor':'#30363d',
                        'steps':[
                            {'range':[0,decision_threshold*100],'color':'rgba(63,185,80,0.12)'},
                            {'range':[decision_threshold*100,100],'color':'rgba(248,81,73,0.12)'},
                        ],
                        'threshold':{'line':{'color':'#e3b341','width':2},
                                     'thickness':0.75,'value':decision_threshold*100},
                    },
                    number={'font':{'color':'#e6edf3','size':42},'suffix':'%'},
                ))
                fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=260,
                                    margin=dict(t=30,b=10,l=30,r=30))
                st.plotly_chart(fig_g, use_container_width=True)

                if 'prediction_history' not in st.session_state:
                    st.session_state.prediction_history = []
                st.session_state.prediction_history.append({
                    'timestamp':  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'prediction': label,
                    'confidence': f"{conf*100:.1f}%",
                })

            except Exception as e:
                box("❌",f"Error processing image: {e}","danger")

    else:
        uploaded_files = st.file_uploader("Upload Multiple MRI Scans",
                                          type=['jpg','jpeg','png'],
                                          accept_multiple_files=True)
        if uploaded_files and st.button("▶  Run Batch", use_container_width=True):
            results, prog = [], st.progress(0)
            for idx, f in enumerate(uploaded_files):
                try:
                    _,_,norm = preprocess_uploaded_image(f)
                    prob = float(model.predict(np.expand_dims(norm,0),verbose=0)[0][0])
                    is_t = prob > decision_threshold
                    conf = prob if is_t else 1-prob
                    results.append({'File':f.name,
                                    'Result':'🔴 Tumor' if is_t else '🟢 No Tumor',
                                    'Confidence':f"{conf*100:.1f}%",
                                    'Raw Prob':f"{prob:.4f}"})
                except:
                    results.append({'File':f.name,'Result':'⚠️ Error',
                                    'Confidence':'N/A','Raw Prob':'N/A'})
                prog.progress((idx+1)/len(uploaded_files))

            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            t_c = sum(1 for r in results if '🔴' in r['Result'])
            section("BATCH SUMMARY")
            fig_b = make_subplots(rows=1,cols=2, specs=[[{'type':'pie'},{'type':'bar'}]])
            fig_b.add_trace(go.Pie(
                labels=['Tumor','No Tumor'], values=[t_c,len(results)-t_c],
                hole=0.5, marker_colors=['#f85149','#3fb950'],
                textinfo='percent+label', textfont_size=11,
            ), row=1, col=1)
            fig_b.add_trace(go.Bar(
                x=['Tumor','No Tumor'], y=[t_c,len(results)-t_c],
                marker_color=['#f85149','#3fb950'],
                text=[t_c,len(results)-t_c], textposition='outside',
                textfont=dict(color='#c9d1d9'),
            ), row=1, col=2)
            fig_b.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font_color='#8b949e', showlegend=False,
                                height=300, margin=dict(t=20,b=20))
            fig_b.update_yaxes(gridcolor='#21262d',row=1,col=2)
            st.plotly_chart(fig_b, use_container_width=True)

            st.download_button("⬇  Download CSV",
                               df.to_csv(index=False),"predictions.csv","text/csv")

    if st.session_state.get('prediction_history'):
        with st.expander("📜 Prediction History"):
            st.dataframe(pd.DataFrame(st.session_state.prediction_history),
                         use_container_width=True)
            if st.button("🗑  Clear History"):
                st.session_state.prediction_history = []; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Brain Tumor Detection System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_css()

    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

    with st.sidebar:
        st.markdown("## 🧠 Brain Tumor AI")
        st.markdown("---")
        pages = {
            "🏠 Dashboard":        "dashboard",
            "📊 Dataset Explorer": "dataset",
            "🏋️ Train Model":      "train",
            "🔬 Predict":          "predict",
        }
        selection = st.radio("Navigation", list(pages.keys()))
        st.markdown("---")
        st.markdown("### 📋 System Info")
        tf_ver = tf.__version__
        gpu    = "GPU ✓" if tf.config.list_physical_devices('GPU') else "CPU only"
        st.markdown(f"""
- **Model:** Custom CNN
- **Input:** 240 × 240 × 3
- **Framework:** TensorFlow {tf_ver}
- **Compute:** {gpu}
- **Version:** 3.0
        """)
        st.markdown("---")
        st.caption("⚠️ For educational/research use only.")

    route = pages[selection]
    if   route == "dashboard": page_dashboard()
    elif route == "dataset":   page_dataset_explorer()
    elif route == "train":     page_train()
    elif route == "predict":   page_predict()


if __name__ == '__main__':
    main()
