# Sign Kit - Multi-Language Sign Language Detection Platform

A full-stack web application for real-time sign language detection, learning, and practice. Sign Kit supports three sign language systems — American Sign Language (ASL), Indian Sign Language (ISL), and Tamil Sign Language (TSL) — using machine learning models powered by MediaPipe hand tracking.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Machine Learning Models](#machine-learning-models)
- [API Reference](#api-reference)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup (Node.js)](#backend-setup-nodejs)
  - [ML Service Setup (Python)](#ml-service-setup-python)
  - [Frontend Setup (React)](#frontend-setup-react)
- [Usage](#usage)
- [Application Pages](#application-pages)

---

## Project Overview

Sign Kit is designed to bridge the communication gap for the hearing-impaired community. It provides a platform where users can:

- Perform real-time sign language character detection via webcam
- Learn sign language through guided video content
- Practice signing with interactive games
- Convert detected signs into readable text sentences
- Share and create sign language learning videos

The system uses computer vision and machine learning to recognize hand gestures and map them to alphabets and characters in multiple sign language systems.

---

## Features

- Real-time hand landmark detection using MediaPipe
- Multi-language support: ASL, ISL, and Tamil Sign Language (TSL)
- Automatic character detection with configurable hold-time thresholds
- Word and sentence building from detected characters
- Sign language learning videos (browse, create, and share)
- Interactive sign language learning games
- User authentication with signup and login
- Feedback submission system
- Text conversion from detected sign sequences
- Session-based detection logging

---

## Tech Stack

| Layer        | Technology                                      |
|--------------|-------------------------------------------------|
| Frontend     | React.js, React Router, CSS                     |
| Auth Backend | Node.js, Express.js, MongoDB, Mongoose          |
| ML Service   | Python, FastAPI, Uvicorn                        |
| ML / Vision  | MediaPipe, OpenCV, NumPy, scikit-learn, Keras   |
| Database     | MongoDB (via Mongoose ODM)                      |
| Models       | Random Forest (ASL), LSTM Neural Network (ISL/TSL) |

---

## Architecture

The application is composed of three independently running services:

```
User (Browser)
      |
      | HTTP (React Router)
      v
React Frontend  :3001 (or CRA default)
      |
      |---> Node.js Auth Backend  :3000   (User registration, login)
      |         |
      |         v
      |       MongoDB              (User data storage)
      |
      |---> Python FastAPI ML Service  :8000  (Sign detection)
                  |
                  v
              Webcam (OpenCV) + MediaPipe
                  |
                  v
              ML Model Inference
              (Random Forest / LSTM)
```

**Data Flow for Sign Detection:**

1. User opens the Detect page and selects a sign language (ASL, ISL, or TSL).
2. The frontend calls `POST /start_detection` on the FastAPI service with the selected language.
3. The ML service opens an OpenCV window, activates the webcam, and initializes the appropriate model.
4. MediaPipe processes each video frame to extract 21 hand landmark coordinates (63 features per hand).
5. The extracted features are fed into the ML model for character prediction.
6. When a character is held consistently for 1.5–3 seconds, it is automatically added to the current word buffer.
7. The user presses ENTER to commit the word to a sentence, and Q to finalize the session.
8. The frontend polls `GET /detection_status` to display live character, word, and sentence output.
9. The final sentence is saved to a text file and retrievable from `GET /get_detected_text`.

---

## Project Structure

```
C976/
├── frontend/                   # React.js web application
│   ├── public/                 # Static public assets
│   └── src/
│       ├── App.js              # Root router and layout
│       ├── Pages/              # Application page components
│       │   ├── Login.js        # User login page
│       │   ├── SignUp.js       # User registration page
│       │   ├── Home.js         # Landing/home page
│       │   ├── Detect.js       # Sign detection interface
│       │   ├── Sign.js         # Sign language viewer
│       │   ├── Convert.js      # Sign-to-text conversion
│       │   ├── LearnSign.js    # Learning module
│       │   ├── Games.js        # Interactive games
│       │   ├── Videos.js       # Video gallery
│       │   ├── Video.js        # Single video player
│       │   ├── CreateVideo.js  # Video upload/creation
│       │   └── Feedback.js     # User feedback form
│       ├── Components/         # Shared UI components (Navbar, Footer)
│       ├── Animations/         # Animation assets
│       ├── Config/             # Frontend configuration
│       └── Models/             # Frontend data models/constants
│
├── backend/                    # Node.js authentication server
│   ├── server.js               # Express server entry point
│   ├── config/
│   │   └── db.js               # MongoDB connection setup
│   ├── models/
│   │   └── User.js             # Mongoose User schema
│   ├── controllers/
│   │   └── UserController.js   # Signup and login logic
│   └── routes/
│       └── UserRoutes.js       # Auth route definitions
│
├── ml/                         # Python ML service
│   ├── api.py                  # FastAPI server (main entry point)
│   ├── models.py               # Unified sign language detector classes
│   ├── model.p                 # Pickled ASL Random Forest model
│   ├── random_forest_isl_model.pkl       # ISL Random Forest model (legacy)
│   ├── best_lstm_model.keras             # Best LSTM model for ISL
│   ├── final_lstm_hand_model.keras       # Final LSTM model for hand detection
│   ├── tamil_labels.json                 # Label mapping for Tamil signs
│   ├── sign_language_detections.txt      # Session detection log output
│   ├── requirements.txt                  # Python dependencies
│   ├── ml_model.ipynb                    # ISL/ASL model training notebook
│   ├── ISL.ipynb                         # ISL-specific experiments
│   └── tamil-sign-to-text.ipynb         # Tamil sign language training notebook
│
└── dataset/                    # Training datasets (not version-controlled)
```

---

## Machine Learning Models

### ASL (American Sign Language)
- **Model:** Random Forest Classifier (`model.p`)
- **Framework:** scikit-learn
- **Input:** 63 normalized hand landmark coordinates (21 keypoints x 3 values) from MediaPipe single-hand detection
- **Output:** A-Z character classification
- **Auto-detection Threshold:** Character must be held consistently for 1.5 to 2 seconds

### ISL (Indian Sign Language)
- **Model:** LSTM Neural Network (`best_lstm_model.keras`)
- **Framework:** TensorFlow / Keras
- **Input:** Sequence of hand landmark coordinates from MediaPipe (supports two-hand detection)
- **Output:** ISL character/letter classification
- **Auto-detection Threshold:** Character must be held consistently for 2 to 3 seconds

### TSL / Tamil Sign Language
- **Model:** LSTM Neural Network (`final_lstm_hand_model.keras`)
- **Framework:** TensorFlow / Keras
- **Label Mapping:** `tamil_labels.json`
- **Input:** Sequence of hand landmark coordinates from MediaPipe (supports two-hand detection)
- **Output:** Tamil character classification
- **Auto-detection Threshold:** Character must be held consistently for 2 to 3 seconds

---

## API Reference

The ML service runs on `http://localhost:8000`. All endpoints return JSON.

| Method | Endpoint                      | Description                                              |
|--------|-------------------------------|----------------------------------------------------------|
| GET    | `/`                           | API overview and usage guide                             |
| POST   | `/start_detection`            | Start webcam-based detection for a given language        |
| POST   | `/stop_detection`             | Stop the active detection session                        |
| GET    | `/detection_status`           | Poll for live detection state (char, word, sentence)     |
| GET    | `/get_detected_text`          | Retrieve the finalized sentence from the last session    |
| GET    | `/detection_log`              | Retrieve the full detection log for the current session  |
| DELETE | `/clear_session`              | Reset session state and remove saved text files          |
| GET    | `/model_info/{language}`      | Get model metadata for a given language                  |
| GET    | `/supported_languages`        | List all supported sign languages and their properties   |
| GET    | `/word_formation/{language}`  | Get word formation practice exercises                    |
| GET    | `/statistics`                 | Get session-level detection statistics                   |

**Start Detection Request Body:**
```json
{
  "language": "ASL"
}
```
Valid values for `language`: `ASL`, `ISL`, `TSL`, `TAMIL`

---

## Getting Started

### Prerequisites

- Node.js v18 or later
- npm v9 or later
- Python 3.9 or later
- MongoDB (local instance or MongoDB Atlas connection string)
- A webcam connected to the machine running the ML service

---

### Backend Setup (Node.js)

```bash
cd backend
npm install
```

Create a `.env` file or update `config/db.js` with your MongoDB connection string.

```bash
npm start
```

The auth server will start at `http://localhost:3000`.

---

### ML Service Setup (Python)

```bash
cd ml
pip install -r requirements.txt
python api.py
```

The FastAPI ML service will start at `http://localhost:8000`.

> Note: The ML service opens an OpenCV window on the machine where it is running. It requires a physical webcam and a display environment.

---

### Frontend Setup (React)

```bash
cd frontend
npm install
npm start
```

The React development server will start at `http://localhost:3001` (or the next available port).

---

## Usage

1. Open the application in a browser and register a new account or log in.
2. Navigate to the Detect page from the navigation bar.
3. Select a sign language: ASL, ISL, or Tamil.
4. An OpenCV window will open on the ML server machine, activating the webcam.
5. Position your hand inside the green bounding box shown on the video feed.
6. Hold a sign gesture steady. A progress bar will indicate detection confidence.
7. When confidence reaches 100%, the character is automatically added to the word buffer.
8. Press **ENTER** to confirm the current word and add it to the sentence.
9. Press **Q** to finalize the session. The detected sentence is saved and available in the application.

---

## Application Pages

| Route                   | Page            | Description                                        |
|-------------------------|-----------------|----------------------------------------------------|
| `/`                     | Login           | User login screen                                  |
| `/signup`               | Sign Up         | New user registration                              |
| `/sign-kit/home`        | Home            | Main landing page after login                      |
| `/sign-kit/detect`      | Detect          | Real-time sign language detection interface        |
| `/sign-kit/convert`     | Convert         | Sign sequence to text conversion                   |
| `/sign-kit/learn-sign`  | Learn Sign      | Guided sign language learning module               |
| `/sign-kit/games`       | Games           | Interactive games for sign language practice       |
| `/sign-kit/all-videos`  | Videos          | Browse community sign language videos              |
| `/sign-kit/video/:id`   | Video Player    | Watch an individual video                          |
| `/sign-kit/create-video`| Create Video    | Upload or record a new video                       |
| `/sign-kit/feedback`    | Feedback        | Submit user feedback and suggestions               |
