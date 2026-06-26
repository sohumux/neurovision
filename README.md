# 🧠 NeuroVision — Intelligent Face Recognition System

A web-based AI face recognition application built with Python (Flask) and the `face_recognition` library.

## Features
- 🎥 Live webcam face capture
- 🖼️ Upload image for identification
- 👤 Register persons with name, age, gender, course
- 🔍 Match faces against database (with confidence score)
- 📊 Classify as **Adult** (18+) or **Teen** (<18)
- 🗃️ SQLite database — no extra DB server needed

---

## Quick Start

### 1. Install Dependencies

**Linux/macOS:**
```bash
sudo apt-get install cmake build-essential libopenblas-dev liblapack-dev
pip install flask face_recognition Pillow numpy
```

**Windows:**
1. Install [CMake](https://cmake.org/download/) and add to PATH
2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
3. `pip install flask face_recognition Pillow numpy`

### 2. Run the App
```bash
python app.py
```
Visit: **http://localhost:5000**

---

## Uploading Data to the Database

### Option A — Web UI
Navigate to the **Register** tab and fill in the form.

### Option B — Batch Upload (CSV)
Create `data.csv`:
```
name,age,gender,course,photo_path
Arjun Sharma,20,Male,B.Tech CSE,photos/arjun.jpg
Priya Nair,17,Female,11th Science,photos/priya.jpg
```
Then run:
```bash
python batch_upload.py csv data.csv
```

### Option C — Batch Upload (Folder)
```bash
python batch_upload.py folder ./photos
```
Each filename becomes the person's name (e.g. `Arjun_Sharma.jpg` → "Arjun Sharma").

---

## Project Structure
```
neurovision/
├── app.py              ← Flask backend
├── batch_upload.py     ← Bulk upload helper
├── requirements.txt    ← Dependencies
├── database/
│   └── neurovision.db  ← SQLite database (auto-created)
└── templates/
    └── index.html      ← Frontend UI
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/register` | Register a new person |
| POST | `/api/identify` | Identify a person from image |
| GET  | `/api/persons`  | List all persons |
| DELETE | `/api/persons/<id>` | Delete a person |
| GET  | `/api/stats`    | Get database statistics |
"# neurovision" 
