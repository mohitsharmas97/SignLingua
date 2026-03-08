from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import threading
import os
import time
from datetime import datetime
import asyncio
import uvicorn
from models import UnifiedSignLanguageDetector

app = FastAPI(title="Unified Sign Language Detection API", version="3.0.0")

# Add CORS middleware for React app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DETECTED_TEXT_FILE = "detected_text.txt"
DETECTION_LOG_FILE = "sign_language_detections.txt"

# Global variables
detection_active = False
detection_thread = None
detector = UnifiedSignLanguageDetector()
detection_status = {
    "active": False,
    "language": None,
    "word_buffer": "",
    "sentence_buffer": "",
    "last_detected_char": "?",
    "confidence": 0.0,
    "session_id": None,
    "completed": False,
    "final_sentence": "",
    "detection_progress": 0.0,
    "auto_detection_enabled": True
}

# Request models
class DetectionRequest(BaseModel):
    language: str = "ASL"  # ASL, ISL

def log_detection(detection_data):
    """Log detection to text file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log to text file
    with open(DETECTION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {detection_data}\n")

def process_camera_opencv(language="ASL"):
    """Unified camera processing for ASL, ISL, and Tamil"""
    global detection_active, detection_status, detector
    
    # Initialize detector for specified language
    if not detector.initialize(language):
        print(f"Failed to load {language} model")
        return False
    
    detector.cap = cv2.VideoCapture(0)
    if not detector.cap.isOpened():
        print("Cannot access the camera")
        return False
    
    # Set frame dimensions based on language
    frame_width = 1280 if language in ["ISL", "TSL", "TAMIL"] else 1920
    frame_height = 720 if language in ["ISL", "TSL", "TAMIL"] else 1080
    detector.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    detector.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    window_name = f'{language} Character Detection - Unified API'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, frame_width, frame_height)
    
    # Get language-specific instructions
    instructions = detector.get_instructions()
    
    word_buffer = ""
    sentence_buffer = ""
    detector.session_start_time = datetime.now().isoformat()
    detection_active = True
    
    # Initialize log file
    with open(DETECTION_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== {language} Character Detection Session Started: {detector.session_start_time} ===\n")
    
    # Reset detection status
    detection_status.update({
        "active": True,
        "language": language,
        "word_buffer": "",
        "sentence_buffer": "",
        "last_detected_char": "?",
        "confidence": 0.0,
        "session_id": detector.session_start_time,
        "completed": False,
        "final_sentence": "",
        "detection_progress": 0.0,
        "auto_detection_enabled": True  # All languages now use auto-detection
    })
    
    while detection_active:
        ret, frame = detector.cap.read()
        if not ret:
            print("Failed to read from camera")
            break
        
        # Mirror the frame for more intuitive interaction (except ASL)
        if language in ["ISL", "TSL", "TAMIL"]:
            frame = cv2.flip(frame, 1)
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Add guidance box
        box_size = int(min(width, height) * 0.7)
        center_x = width // 2
        center_y = height // 2
        cv2.rectangle(frame, 
                      (center_x - box_size//2, center_y - box_size//2),
                      (center_x + box_size//2, center_y + box_size//2), 
                      (0, 255, 0), 3)
        
        # Add overlay for status info based on language
        if language in ["ISL", "TSL", "TAMIL"]:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            font_scale = 0.6
            line_spacing = 25
            start_y = 30
        else:
            # ASL style overlay
            overlay = np.zeros((200, width, 3), dtype=np.uint8)
            frame[0:200, 0:width] = cv2.addWeighted(overlay, 0.5, frame[0:200, 0:width], 0.5, 0)
            font_scale = 0.8
            line_spacing = 40
            start_y = 40
        
        # Instructions overlay
        for idx, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, start_y + idx * line_spacing), 
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2)
        
        # Process frame for hand detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.current_detector.hands.process(frame_rgb)
        
        # Get detection results
        detected_char, confidence, extra_info = detector.process_frame(frame)
        
        # Draw hand landmarks
        detector.draw_landmarks(frame, results)
        
        # Display detected character and confidence
        if language in ["ISL", "TSL", "TAMIL"]:
            cv2.putText(frame, f"Detected: {detected_char} ({confidence:.1f}%)", 
                        (10, height - 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, 
                        (0, 255, 0) if detected_char != '?' else (100, 100, 100), 2)
        else:
            if detected_char != '?':
                cv2.putText(frame, f"Detected: {detected_char}", (10, 250), 
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        
        # Handle auto-detection progress bar for all languages
        if extra_info and extra_info.get("detection_progress", 0) > 0:
            progress = extra_info["detection_progress"]
            bar_width = 200
            bar_height = 20
            bar_x = width - 220
            bar_y = height - 100
            
            # Draw progress bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
            
            # Show progress text
            progress_text = f"Detecting '{detected_char}': {int(progress * 100)}%"
            cv2.putText(frame, progress_text, (bar_x - 50, bar_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display current word and sentence
        word_y = height - 50 if language in ["ISL", "TSL", "TAMIL"] else 300
        sentence_y = height - 20 if language in ["ISL", "TSL", "TAMIL"] else 350
        font_size = 0.7 if language in ["ISL", "TSL", "TAMIL"] else 1
        
        cv2.putText(frame, f"Word: {word_buffer}", (10, word_y), 
                    cv2.FONT_HERSHEY_DUPLEX, font_size, (255, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {sentence_buffer}", (10, sentence_y), 
                    cv2.FONT_HERSHEY_DUPLEX, font_size, (0, 255, 255), 2)
        
        # Handle automatic detection for all languages
        if extra_info and extra_info.get("should_auto_detect", False):
            word_buffer += detected_char
            log_detection(f"Auto-added '{detected_char}' to word. Current word: '{word_buffer}'")
            
            # Flash effect
            flash = np.ones_like(frame) * 255
            cv2.imshow(window_name, flash)
            cv2.waitKey(50)
        
        # Update global status for API access
        detection_status.update({
            "active": True,
            "language": language,
            "word_buffer": word_buffer,
            "sentence_buffer": sentence_buffer,
            "last_detected_char": detected_char,
            "confidence": confidence,
            "session_id": detector.session_start_time,
            "completed": False,
            "final_sentence": "",
            "detection_progress": extra_info.get("detection_progress", 0.0) if extra_info else 0.0,
            "auto_detection_enabled": True
        })
        
        cv2.imshow(window_name, frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # ENTER - Add word to sentence
            if word_buffer.strip():
                if sentence_buffer:
                    sentence_buffer += ' ' + word_buffer.strip()
                else:
                    sentence_buffer = word_buffer.strip()
                
                log_detection(f"Added word '{word_buffer}' to sentence. Current sentence: '{sentence_buffer}'")
                word_buffer = ""
                
        elif key == ord('q'):  # Q - Finish sentence detection
            # Add the last word if any
            if word_buffer.strip():
                if sentence_buffer:
                    sentence_buffer += ' ' + word_buffer.strip()
                else:
                    sentence_buffer = word_buffer.strip()
                word_buffer = ""
            
            # Save sentence to text file
            with open(DETECTED_TEXT_FILE, 'w', encoding='utf-8') as f:
                f.write(sentence_buffer)
            
            log_detection(f"Session completed. Final sentence: '{sentence_buffer}'")
            
            # Update final status
            detection_status.update({
                "active": False,
                "language": language,
                "word_buffer": word_buffer,
                "sentence_buffer": sentence_buffer,
                "last_detected_char": detected_char,
                "confidence": confidence,
                "completed": True,
                "final_sentence": sentence_buffer,
                "detection_progress": 0.0
            })
            
            detection_active = False
            break
    
    # Cleanup
    detector.cleanup()
    
    # Log session end
    with open(DETECTION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"=== Session Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    return True

@app.post("/start_detection")
async def start_detection(request: DetectionRequest, background_tasks: BackgroundTasks):
    """Start sign language character detection with OpenCV popup window"""
    global detection_thread, detection_active
    
    # Validate language
    language = request.language.upper()
    if language not in ["ASL", "ISL", "TSL", "TAMIL"]:
        raise HTTPException(status_code=400, detail="Language must be 'ASL', 'ISL', 'TSL', or 'TAMIL'")
    
    if detection_active:
        return JSONResponse(
            status_code=400,
            content={"message": f"{language} detection is already running"}
        )
    
    # Start detection in background thread
    detection_thread = threading.Thread(
        target=process_camera_opencv,
        args=(language,),
        daemon=True
    )
    detection_thread.start()
    
    # Wait a moment for detection to start
    await asyncio.sleep(1)
    
    # Get detector info
    detector_info = detector.get_detector_info()
    
    return JSONResponse(
        content={
            "message": f"{language} character detection started successfully",
            "language": language,
            "instructions": detector.get_instructions(),
            "session_id": detector.session_start_time,
            "status_url": "/detection_status",
            "detector_info": detector_info
        }
    )

@app.post("/stop_detection")
async def stop_detection():
    """Stop sign language character detection"""
    global detection_active, detection_thread
    
    if not detection_active:
        return JSONResponse(
            status_code=400,
            content={"message": "Detection is not running"}
        )
    
    detection_active = False
    
    # Wait for thread to finish
    if detection_thread and detection_thread.is_alive():
        detection_thread.join(timeout=5)
    
    return JSONResponse(content={"message": "Character detection stopped successfully"})

@app.get("/detection_status")
async def get_detection_status():
    """Get current character detection status and text output"""
    global detection_status
    
    return JSONResponse(content=detection_status)

@app.get("/get_detected_text")
async def get_detected_text():
    """Get the final detected sentence as text"""
    try:
        if os.path.exists(DETECTED_TEXT_FILE):
            with open(DETECTED_TEXT_FILE, 'r', encoding='utf-8') as f:
                detected_sentence = f.read().strip()
            
            return JSONResponse(content={
                "text": detected_sentence,
                "available": True,
                "language": detection_status.get("language", "Unknown"),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return JSONResponse(content={
                "text": "",
                "available": False,
                "language": detection_status.get("language", "Unknown"),
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error reading detected text: {str(e)}"}
        )

@app.get("/detection_log")
async def get_detection_log():
    """Get sign language character detection log"""
    try:
        with open(DETECTION_LOG_FILE, "r", encoding="utf-8") as f:
            log_content = f.read()
        return JSONResponse(content={"log": log_content})
    except FileNotFoundError:
        return JSONResponse(content={"log": ""})

@app.delete("/clear_session")
async def clear_session():
    """Clear current session data"""
    global detection_status
    
    try:
        # Clear files
        if os.path.exists(DETECTED_TEXT_FILE):
            os.remove(DETECTED_TEXT_FILE)
        
        # Reset status
        detection_status = {
            "active": False,
            "language": None,
            "word_buffer": "",
            "sentence_buffer": "",
            "last_detected_char": "?",
            "confidence": 0.0,
            "session_id": None,
            "completed": False,
            "final_sentence": "",
            "detection_progress": 0.0,
            "auto_detection_enabled": True
        }
        
        return JSONResponse(content={"message": "Session cleared successfully"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error clearing session: {str(e)}"}
        )

@app.get("/model_info/{language}")
async def get_model_info(language: str):
    """Get information about the specified language model"""
    language = language.upper()
    
    if language not in ["ASL", "ISL", "TSL", "TAMIL"]:
        raise HTTPException(status_code=400, detail="Language must be 'ASL', 'ISL', 'TSL', or 'TAMIL'")
    
    # Temporarily initialize detector to get info
    temp_detector = UnifiedSignLanguageDetector()
    if temp_detector.initialize(language):
        info = temp_detector.get_detector_info()
        temp_detector.cleanup()
        return JSONResponse(content=info)
    else:
        return JSONResponse(
            status_code=500,
            content={"message": f"Could not load {language} model information"}
        )

@app.get("/supported_languages")
async def get_supported_languages():
    """Get list of supported sign languages"""
    return JSONResponse(content={
        "supported_languages": ["ASL", "ISL", "TSL", "TAMIL"],
        "ASL": {
            "name": "American Sign Language",
            "model_type": "Random Forest Classifier",
            "capture_method": "Automatic (hold for 1.5-2 seconds)",
            "hands_required": "Single hand"
        },
        "ISL": {
            "name": "Indian Sign Language", 
            "model_type": "LSTM Neural Network",
            "capture_method": "Automatic (hold for 2-3 seconds)",
            "hands_required": "Both hands supported"
        },
        "TSL": {
            "name": "Tamil Sign Language",
            "model_type": "LSTM Neural Network", 
            "capture_method": "Automatic (hold for 2-3 seconds)",
            "hands_required": "Both hands supported"
        },
        "TAMIL": {
            "name": "Tamil Sign Language (alias for TSL)",
            "model_type": "LSTM Neural Network",
            "capture_method": "Automatic (hold for 2-3 seconds)", 
            "hands_required": "Both hands supported"
        }
    })

@app.get("/word_formation/{language}")
async def get_word_formation_exercises(language: str):
    """Get available word formation exercises for the specified language"""
    language = language.upper()
    
    if language not in ["ASL", "ISL", "TSL", "TAMIL"]:
        raise HTTPException(status_code=400, detail="Language must be 'ASL', 'ISL', 'TSL', or 'TAMIL'")
    
    # Common words that can be formed (you can expand this based on your models)
    common_words = {
        "ASL": {
            "HELLO": ["H", "E", "L", "L", "O"],
            "WORLD": ["W", "O", "R", "L", "D"],
            "LOVE": ["L", "O", "V", "E"],
            "PEACE": ["P", "E", "A", "C", "E"]
        },
        "ISL": {
            "HELLO": ["H", "E", "L", "L", "O"],
            "NAMASTE": ["N", "A", "M", "A", "S", "T", "E"],
            "INDIA": ["I", "N", "D", "I", "A"]
        },
        "TSL": {
            "VANAKKAM": ["வ", "ண", "க", "க", "ம"],
            "NANDRI": ["ந", "ன", "ற", "ி"],
            "TAMIL": ["த", "மி", "ழ"]
        },
        "TAMIL": {
            "VANAKKAM": ["வ", "ண", "க", "க", "ம"],
            "NANDRI": ["ந", "ன", "ற", "ி"],
            "TAMIL": ["த", "மி", "ழ"]
        }
    }
    
    return JSONResponse(content={
        "language": language,
        "available_words": common_words.get(language, {}),
        "instructions": f"Practice forming words in {language} by holding each character sign for the required duration"
    })

@app.get("/statistics")
async def get_detection_statistics():
    """Get detection statistics from the current session"""
    global detection_status
    
    stats = {
        "current_session": {
            "active": detection_status["active"],
            "language": detection_status.get("language"),
            "characters_detected": len(detection_status.get("word_buffer", "") + detection_status.get("sentence_buffer", "")),
            "words_formed": len(detection_status.get("sentence_buffer", "").split()) if detection_status.get("sentence_buffer") else 0,
            "session_duration": None
        },
        "system_info": {
            "supported_languages": ["ASL", "ISL", "TSL", "TAMIL"],
            "detection_method": "Real-time camera with MediaPipe hand tracking",
            "api_version": "3.0.0"
        }
    }
    
    # Calculate session duration if active
    if detection_status["active"] and detection_status.get("session_id"):
        try:
            session_start = datetime.fromisoformat(detection_status["session_id"])
            duration = datetime.now() - session_start
            stats["current_session"]["session_duration"] = str(duration).split('.')[0]  # Remove microseconds
        except:
            pass
    
    return JSONResponse(content=stats)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Unified Sign Language Detection API",
        "version": "3.0.0",
        "description": "Real-time ASL, ISL, and Tamil Sign Language character detection with OpenCV popup window",
        "supported_languages": ["ASL", "ISL", "TSL", "TAMIL"],
        "endpoints": {
            "start_detection": "POST /start_detection - Start camera detection (specify language in body)",
            "stop_detection": "POST /stop_detection - Stop detection",
            "detection_status": "GET /detection_status - Get real-time status",
            "get_detected_text": "GET /get_detected_text - Get final sentence text",
            "detection_log": "GET /detection_log - Get detection log",
            "clear_session": "DELETE /clear_session - Clear session data",
            "model_info": "GET /model_info/{language} - Get model information",
            "supported_languages": "GET /supported_languages - Get supported languages info",
            "word_formation": "GET /word_formation/{language} - Get word formation exercises",
            "statistics": "GET /statistics - Get detection statistics"
        },
        "usage": {
            "All_Languages": {
                "controls": {
                    "ENTER": "Add current word to sentence",
                    "Q": "Complete sentence and save as text"
                },
                "method": "Automatic detection - hold sign for required duration"
            }
        },
        "features": {
            "unified_api": "Single API for ASL, ISL, and Tamil Sign Language",
            "language_switching": "Switch between different sign languages",
            "real_time_status": "Live detection status and progress",
            "text_output": "Detected sentences saved as text files",
            "automatic_detection": "All languages use automatic character detection",
            "progress_tracking": "Visual progress bars for character detection",
            "word_formation": "Support for word formation exercises"
        },
        "auto_detection_info": {
            "ASL": "Hold sign for 1.5-2 seconds for automatic detection",
            "ISL": "Hold sign for 2-3 seconds for automatic detection", 
            "TSL/TAMIL": "Hold sign for 2-3 seconds for automatic detection"
        }
    }

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(os.path.dirname(os.path.abspath(DETECTION_LOG_FILE)), exist_ok=True)
    
    print("Starting Unified Sign Language Detection API...")
    print("Supported Languages: ASL (American), ISL (Indian), and TSL/Tamil")
    print("All languages now use automatic detection - no manual key presses needed!")
    print("The API will open OpenCV popup windows for camera detection")
    print("Text output will be available via /get_detected_text endpoint")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )