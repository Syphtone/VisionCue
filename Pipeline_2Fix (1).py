import os
import numpy as np
import pandas as pd
import subprocess
import logging
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
import joblib
from pathlib import Path
from groq import Groq
from config import API_KEY

# Load model and scaler once
MODEL_PATH = r"D:\RP2\RP2_Projects\AI_Interview_Coach\Main\facelstm_WFV_v1.keras"
SCALER_PATH = r"D:\RP2\RP2_Projects\AI_Interview_Coach\Main\scaler.pkl"

# ⚠️ YOU MUST UPDATE THESE PATHS TO YOUR ACTUAL INSTALLATION
FFMPEG_PATH = r"D:\RP2\RP2_Projects\AI_Interview_Coach\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

# ⚠️ YOU MUST UPDATE THESE PATHS TO YOUR ACTUAL INSTALLATION
# OPENFACE_PATH = r"D:\RP2\AI_Interview_Coach\OpenFace\build\bin\Release\FeatureExtraction.exe"  # UPDATE THIS!
OPENSMILE_PATH = r"D:\RP2\RP2_Projects\AI_Interview_Coach\opensmile-3.0.2\bin\SMILExtract.exe"
OPENSMILE_CONFIG = r"D:\RP2\RP2_Projects\AI_Interview_Coach\opensmile-3.0.2\config\prosody\prosodyShs.conf"

def convert_webm_to_mp4(input_path, output_path="temp_converted.mp4"):
    """Convert WebM to MP4 using ffmpeg (must be installed)"""
    try:
        logger.info(f"Converting {input_path} to MP4...")
        subprocess.run([
            "ffmpeg", "-i", input_path,
            "-c:v", "libx264", "-c:a", "aac",
            "-y", output_path
        ], check=True, capture_output=True)
        logger.info("Conversion complete")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        # If ffmpeg fails, try to use original file anyway
        return input_path
    except FileNotFoundError:
        logger.warning("FFmpeg not found, using original file")
        return input_path

def extract_openface(video_path, output_dir="temp_openface_output"):
    """
    Extract facial features using OpenFace (via Docker)
    instead of local .exe dependency.
    """
    logger.info("Running OpenFace feature extraction using Docker...")

    try:
        os.makedirs(output_dir, exist_ok=True)
        video_path = Path(video_path).resolve()
        output_dir = Path(output_dir).resolve()

        # Docker command: mount temp directory, run OpenFace, output results
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{video_path.parent}:/data",
            "-v", f"{output_dir}:/output",
            "openface", # Docker image name
            "-f", f"/data/{video_path.name}",
            "-out_dir", "/output",
            "-no_face_detector_haar" # Disable Haar cascade face detector, use MTCNN instead
        ]

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300
        )

        logger.info("✅ OpenFace extraction complete (Docker)")

        # Find output CSV
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_csv = os.path.join(output_dir, f"{base_name}.csv")

        if not os.path.exists(out_csv):
            logger.error(f"Expected output not found: {out_csv}")
            logger.error(f"Files in output dir: {os.listdir(output_dir)}")
            raise FileNotFoundError(f"OpenFace output not created: {out_csv}")

        logger.info(f"OpenFace output found: {out_csv}")
        return pd.read_csv(out_csv)

    except subprocess.TimeoutExpired:
        logger.error("OpenFace timed out after 5 minutes")
        raise Exception("OpenFace (Docker) processing timed out")

    except subprocess.CalledProcessError as e:
        logger.error(f"OpenFace Docker failed: {e.stderr or str(e)}")
        raise Exception(f"OpenFace Docker extraction failed: {e}")

    except Exception as e:
        logger.error(f"OpenFace Docker error: {e}")
        raise


def convert_to_wav_ffmpeg(video_path, wav_path="temp_audio.wav"):
    """Convert video to WAV using ffmpeg with tqdm progress bar"""
    logger.info("🎧 Extracting audio from video using ffmpeg...")
    
    # Remove old file if it exists
    if os.path.exists(wav_path):
        os.remove(wav_path)

    # ffmpeg command
    cmd = [
        FFMPEG_PATH,
        "-y",  # overwrite
        "-i", video_path,
        "-vn",  # no video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",  # sample rate
        "-ac", "1",  # mono
        wav_path
    ]

    # Run ffmpeg with progress
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    for line in tqdm(process.stderr, desc="Converting to WAV", unit="lines"):
        pass  # tqdm just for visualization

    process.wait()

    if process.returncode != 0 or not os.path.exists(wav_path):
        logger.error("ffmpeg failed to extract audio.")
        raise Exception("Audio extraction failed using ffmpeg")

    logger.info(f"✅ Audio extracted successfully: {wav_path}")
    return wav_path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_opensmile(video_path, out_csv="processed\temp_opensmile_output.csv"):
    """Extract audio features using OpenSmile"""
    if not os.path.exists(OPENSMILE_PATH):
        logger.error(f"OpenSmile not found at {OPENSMILE_PATH}")
        raise FileNotFoundError(f"OpenSmile executable not found: {OPENSMILE_PATH}")
    
    logger.info("Running OpenSmile audio extraction...")
    audio_path = convert_to_wav_ffmpeg(video_path)
    try:
        result = subprocess.run([
            OPENSMILE_PATH,
            "-C", OPENSMILE_CONFIG,
            "-I", audio_path,
            "-csvoutput", out_csv
        ], check=True, capture_output=True, timeout=300)
        
        logger.info("OpenSmile extraction complete")
        
        if not os.path.exists(out_csv):
            raise FileNotFoundError(f"OpenSmile output not created: {out_csv}")
            
        return pd.read_csv(out_csv, sep=';', engine='python')
    
    except subprocess.TimeoutExpired:
        logger.error("OpenSmile timed out after 5 minutes")
        raise Exception("OpenSmile processing timed out")
    except subprocess.CalledProcessError as e:
        logger.error(f"OpenSmile failed: {e.stderr.decode() if e.stderr else str(e)}")
        raise Exception(f"OpenSmile extraction failed: {e}")

logger = logging.getLogger(__name__)

def align_features(openface_df, opensmile_df, opensmile_csv):
    """
    Align OpenFace (video) and OpenSMILE (audio) features for a single video.
    Uses nearest timestamp matching (merge_asof).
    """
    try:
        # --- Load OpenSMILE (audio) CSV ---
        opensmile_df = opensmile_df.loc[:, ~opensmile_df.columns.str.contains('^Unnamed')]
        opensmile_df.columns = [c.strip() for c in opensmile_df.columns]

        # Detect frameTime or time column
        if 'frameTime' in opensmile_df.columns:
            opensmile_df.rename(columns={'frameTime': 'timestamp'}, inplace=True)
        elif 'time' in opensmile_df.columns:
            opensmile_df.rename(columns={'time': 'timestamp'}, inplace=True)
        else:
            time_cols = [col for col in opensmile_df.columns if 'time' in col.lower()]
            if time_cols:
                opensmile_df.rename(columns={time_cols[0]: 'timestamp'}, inplace=True)
                logger.warning(f"Using '{time_cols[0]}' as timestamp column for OpenSMILE")
            else:
                raise KeyError(f"No timestamp column found in {opensmile_csv}")

        # Drop "name" if it exists
        if 'name' in opensmile_df.columns:
            opensmile_df.drop(columns=['name'], inplace=True)

        # --- Load OpenFace (video) CSV ---
        openface_df.columns = [c.strip() for c in openface_df.columns]

        # Detect timestamp column
        if 'timestamp' not in openface_df.columns:
            possible_cols = [c for c in openface_df.columns if 'timestamp' in c.lower() or 'frame' in c.lower()]
            if possible_cols:
                openface_df.rename(columns={possible_cols[0]: 'timestamp'}, inplace=True)
                logger.warning(f"Using '{possible_cols[0]}' as timestamp column for OpenFace")
            else:
                openface_df.rename(columns={openface_df.columns[0]: 'timestamp'}, inplace=True)

        # --- Clean and Convert ---
        opensmile_df['timestamp'] = pd.to_numeric(opensmile_df['timestamp'], errors='coerce')
        openface_df['timestamp'] = pd.to_numeric(openface_df['timestamp'], errors='coerce')
        opensmile_df.dropna(subset=['timestamp'], inplace=True)
        openface_df.dropna(subset=['timestamp'], inplace=True)

        if opensmile_df.empty or openface_df.empty:
            raise ValueError("Empty timestamps after cleaning")

        # Round timestamps to 3 decimals (millisecond precision)
        opensmile_df['timestamp'] = opensmile_df['timestamp'].round(3)
        openface_df['timestamp'] = openface_df['timestamp'].round(3)

        # Sort before merging
        opensmile_df.sort_values('timestamp', inplace=True)
        openface_df.sort_values('timestamp', inplace=True)

        # --- Align (nearest timestamp match) ---
        merged = pd.merge_asof(
            openface_df,
            opensmile_df,
            on='timestamp',
            direction='nearest',
            tolerance=0.1  # 100ms tolerance
        )

        logger.info(f"✅ Features aligned successfully. Shape: {merged.shape}")
        return merged

    except Exception as e:
        logger.error(f"❌ Error aligning features: {e}")
        raise

def preprocess_features(df):
    """Preprocess and scale features"""
    logger.info("Preprocessing features...")
    
    # Drop irrelevant columns
    drop_cols = ['frame', 'timestamp', 'face_id']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Select relevant columns
    selected_columns = [col for col in df.columns if col not in drop_cols]
    X = df[selected_columns].to_numpy()

    # Scale features
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)

    # Pad sequences
    MAX_SEQ_LENGTH = 1632
    X_padded = pad_sequences(
        [X_scaled], maxlen=MAX_SEQ_LENGTH, dtype="float32", padding="post", truncating="post"
    )
    
    logger.info(f"Preprocessed shape: {X_padded.shape}")
    return X_padded

def signals_to_behavioral_features(df, confidence):
    """Converts merged OpenFace and OpenSMILE signals into a compact behavioral JSON for LLM feedback."""
    logger.info("Converting signals to behavioral features...")
    
    n = len(df)
    thirds = {
        'start': df.iloc[:n//3],
        'middle': df.iloc[n//3:2*n//3],
        'end': df.iloc[2*n//3:]
    }

    #------------------
    # Facial Behaviors
    #------------------
    def eye_contact(chunk):
        gaze = chunk[["gaze_angle_x", "gaze_angle_y"]].abs().mean().mean()
        return "strong" if gaze < 0.15 else "moderate" if gaze < 0.3 else "weak"
    
    def head_stability(chunk):
        motion = chunk[["pose_Rx", "pose_Ry", "pose_Rz"]].std().mean()
        return "stable" if motion < 0.4 else "moderate" if motion < 0.8 else "unstable"
    
    def stress(chunk):
        stress_val = chunk[["AU04_r", "AU15_r"]].mean().mean()
        return "low" if stress_val < 0.8 else "moderate" if stress_val < 1.5 else "high"
    
    def friendliness(chunk):
        smile = chunk[["AU12_r", "AU06_r"]].mean().mean()
        return "high" if smile > 1.5 else "moderate" if smile > 0.7 else "low"
        
    #------------------
    # Audio Behaviors
    #------------------

    #-----loudness------
    loud_mean = df["pcm_loudness_sma"].mean()
    loud_std = df["pcm_loudness_sma"].std()

    if loud_mean < 0.3:
        loudness = "soft"
    elif loud_mean < 0.7:
        loudness = "moderate"
    else:
        loudness = "loud"

    loudness_stability = "stable" if loud_std < 0.2 else "variable"

    #------Speech Fluency------
    voicing_ratio = df["voicingFinalUnclipped_sma"].mean()
    if voicing_ratio > 0.85:
        fluency = "consistent"
    elif voicing_ratio > 0.6:
        fluency = "moderate"
    else:
        fluency = "hesitant"

    #---------------------------
    # Normalize confidence score
    #---------------------------
    if confidence <= 1.0:
        confidence_score = int(confidence*100)
    else:
        confidence_score = int(confidence)
    logger.info(f"Normalized confidence score: {confidence_score}")

    if confidence_score >= 75:
        confidence_level = "high"
    elif confidence_score >= 50:
        confidence_level = "moderate"
    else:
        confidence_level = "low"

    #--------------------
    # Final JSON Output
    #--------------------
    behavioral_features = {
        "visual": {
            "eye_contact": {
                "overall": eye_contact(df),
                "timeline": {k: eye_contact(v) for k, v in thirds.items()}
            },
            "head_stability": {
                "overall": head_stability(df),
            },
            "facial_expressions": {
                "friendliness": friendliness(df),
                "stress": {
                    "overall": stress(df),
                    "timeline": {k: stress(v) for k, v in thirds.items()}
                }
            }
        },
        "audio": {
            "loudness": {
                "level": loudness,
                "stability": loudness_stability
            },
            "speech_fluency": fluency
        },
        "confidence": {
            "score": confidence_score,
            "level": confidence_level
        }
    }
    logger.info("Behavioral features extraction complete")
    return behavioral_features

#-----------------------------
# Feedback Generation Pipeline
#-----------------------------  
client = Groq(api_key=API_KEY)

def generate_feedback(behavioral_features):
    """Generate feedback using Groq LLM based on behavioral features"""
    logger.info("Generating feedback using Groq LLM...")
    
    prompt = f"""
    You are an expert interview coach. Based on the following behavioral features extracted from a candidate's video interview, provide constructive feedback. \n
    Also comment on the overall confidence level.

    Behavioral Features:
    {behavioral_features} \n

    Provide specific suggestions for improvement where applicable.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a professional interview coach."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.6
    )

    feedback = response.choices[0].message.content
    logger.info("Feedback generation complete")
    return feedback

#--------------
# Main Pipeline
#--------------
def predict_from_video(video_path):
    """Main pipeline: extract features and predict confidence"""
    logger.info(f"Starting prediction pipeline for {video_path}")
    
    # Initialize temp file paths
    converted_path = None
    openface_csv = "temp_openface_output.csv"
    opensmile_csv = "temp_opensmile_output.csv"
    
    try:
        # 0. Convert video format if needed
        if video_path.endswith('.webm'):
            converted_path = convert_webm_to_mp4(video_path)
            video_path = converted_path
        
        # 1. Extract features
        openface_df = extract_openface(video_path, openface_csv)
        opensmile_df = extract_opensmile(video_path, opensmile_csv)

        # 2. Merge
        merged = align_features(openface_df, opensmile_df, opensmile_csv)
        
        if len(merged) == 0:
            raise Exception("No aligned features found - video may be too short or corrupted")

        # 3. Preprocess
        X = preprocess_features(merged)

        # 4. Load model and predict
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        
        logger.info("Loading model and predicting...")
        model = load_model(MODEL_PATH)
        pred = model.predict(X, verbose=0)
        confidence = float(pred[0][0])
        
        logger.info(f"Prediction complete: {confidence}")
    
        # 5. Convert signals to behavioral features
        behavioral_features = signals_to_behavioral_features(merged, confidence)

        # 6. Generate AI feedback
        feedback = generate_feedback(behavioral_features)
        return {"confidence": confidence, "feedback": feedback}

    finally:
        # Cleanup temp files
        for temp_file in [converted_path, openface_csv, opensmile_csv]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.info(f"Cleaned up: {temp_file}")
                except Exception as e:
                    logger.warning(f"Could not remove {temp_file}: {e}")
