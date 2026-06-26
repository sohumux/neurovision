from flask import Flask, render_template, request, jsonify
import sqlite3, base64, os, json, io, traceback, tempfile
import numpy as np
from PIL import Image, ImageOps

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
os.makedirs('database', exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  UNIVERSAL IMAGE DECODER  — Pure Pillow, zero OpenCV
#  Converts ANY image format/mode into a clean JPEG temp file on disk.
#  DeepFace reads the JPEG file directly — no numpy array handed to it at all.
# ═══════════════════════════════════════════════════════════════════════════════

def base64_to_jpeg_tempfile(image_data: str) -> str:
    """
    Converts any base64 image (JPEG/PNG/WEBP/BMP/TIFF/GIF/ICO/PPM …)
    into a temporary JPEG file on disk.
    Returns the temp file path (caller must delete it after use).
    """
    # ── 1. strip data-URL prefix and decode ──────────────────────────────────
    s = image_data
    if ',' in s:
        s = s.split(',', 1)[1]
    s = s.strip().replace('\n', '').replace('\r', '').replace(' ', '')
    s += '=' * (-len(s) % 4)
    raw = base64.b64decode(s)

    # ── 2. open with Pillow ───────────────────────────────────────────────────
    img = Image.open(io.BytesIO(raw))
    img.load()

    # ── 3. fix EXIF rotation ──────────────────────────────────────────────────
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # ── 4. normalise 16-bit / 32-bit → 8-bit ─────────────────────────────────
    if img.mode == 'F':                          # 32-bit float
        a = np.array(img, dtype=np.float32)
        lo, hi = a.min(), a.max()
        a = ((a - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
        img = Image.fromarray(a, 'L')

    elif img.mode in ('I', 'I;16', 'I;16B',
                      'I;16L', 'I;16S', 'I;32'):  # 16/32-bit int
        try:
            img = img.convert('I')
        except Exception:
            pass
        a = np.array(img, dtype=np.float32)
        lo, hi = a.min(), a.max()
        a = ((a - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
        img = Image.fromarray(a, 'L')

    # ── 5. flatten transparency ───────────────────────────────────────────────
    if img.mode in ('RGBA', 'PA'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img.convert('RGB'), mask=img.split()[-1])
        img = bg
    elif img.mode == 'LA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img.convert('L').convert('RGB'), mask=img.split()[1])
        img = bg
    elif img.mode == 'P':
        img = img.convert('RGBA')
        bg  = Image.new('RGB', img.size, (255, 255, 255))
        try:
            bg.paste(img.convert('RGB'), mask=img.split()[3])
        except Exception:
            bg.paste(img.convert('RGB'))
        img = bg

    # ── 6. final → RGB ────────────────────────────────────────────────────────
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # ── 7. save as JPEG temp file ─────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(tmp, format='JPEG', quality=95)
    tmp.close()
    return tmp.name          # e.g. /tmp/tmpXXXXXX.jpg


def pil_from_b64(image_data: str) -> Image.Image:
    """Same decode pipeline but returns PIL image (for JPEG storage)."""
    path = base64_to_jpeg_tempfile(image_data)
    try:
        img = Image.open(path).copy()
    finally:
        os.unlink(path)
    return img


# ═══════════════════════════════════════════════════════════════════════════════
#  FACE EMBEDDING  — DeepFace + Facenet512
#  We pass the JPEG *file path* to DeepFace.represent(), never a numpy array.
#  Detector chain: fastmtcnn → mtcnn → retinaface  (all pure Python/TF)
# ═══════════════════════════════════════════════════════════════════════════════

MODEL = 'Facenet512'
DETECTORS = ['retinaface', 'mtcnn']
THRESHOLD = 0.72   # cosine similarity threshold (0–1); raise = stricter


def get_embedding(jpeg_path: str) -> np.ndarray:
    """
    jpeg_path : path to a valid 8-bit RGB JPEG file
    Returns   : float32 numpy vector (512-d)
    Raises    : ValueError if no face found
    """
    from deepface import DeepFace
    last_err = None

    for detector in DETECTORS:
        try:
            result = DeepFace.represent(
                img_path          = jpeg_path,   # ← file path, not array
                model_name        = MODEL,
                enforce_detection = True,
                detector_backend  = detector,
                align             = True,
            )
            if result:
                return np.array(result[0]['embedding'], dtype=np.float32)
        except Exception as e:
            last_err = e
            continue

    raise ValueError(
        f'No face detected in the image. '
        f'Please use a clear, well-lit, frontal photo. '
        f'(tried: {", ".join(DETECTORS)}, last error: {last_err})'
    )


def cosine_sim(a, b) -> float:
    a, b = np.array(a, np.float32), np.array(b, np.float32)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect('database/neurovision.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT    NOT NULL,
            age            INTEGER NOT NULL,
            gender         TEXT    NOT NULL,
            course         TEXT    NOT NULL,
            face_embedding TEXT    NOT NULL,
            photo          BLOB,
            registered_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
    conn.commit(); conn.close()

init_db()

def get_db():
    c = sqlite3.connect('database/neurovision.db')
    c.row_factory = sqlite3.Row
    return c

def classify_age(age): return 'Adult' if int(age) >= 18 else 'Teen'

def photo_b64(blob):
    if not blob: return None
    return 'data:image/jpeg;base64,' + base64.b64encode(bytes(blob)).decode()


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/register', methods=['POST'])
def register_person():
    tmp_path = None
    try:
        data   = request.get_json(force=True)
        name   = data.get('name',   '').strip()
        age    = int(data.get('age', 0))
        gender = data.get('gender', '').strip()
        course = data.get('course', '').strip()
        img_b64 = data.get('image', '')

        if not all([name, age, gender, course, img_b64]):
            return jsonify({'success': False, 'error': 'All fields are required'}), 400

        # convert to JPEG temp file
        try:
            tmp_path = base64_to_jpeg_tempfile(img_b64)
        except Exception as e:
            return jsonify({'success': False,
                'error': f'Image could not be decoded: {e}. '
                         'Supported: JPEG, PNG, WEBP, BMP, TIFF, GIF, ICO.'}), 400

        # get embedding from file path
        try:
            embedding = get_embedding(tmp_path)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400

        # read JPEG bytes for DB storage
        with open(tmp_path, 'rb') as f:
            jpeg_bytes = f.read()

        conn = get_db()
        conn.execute(
            'INSERT INTO persons(name,age,gender,course,face_embedding,photo)'
            ' VALUES(?,?,?,?,?,?)',
            (name, age, gender, course, json.dumps(embedding.tolist()), jpeg_bytes)
        )
        conn.commit(); conn.close()
        return jsonify({'success': True, 'message': f'{name} registered successfully!'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route('/api/identify', methods=['POST'])
def identify_person():
    tmp_path = None
    try:
        data    = request.get_json(force=True)
        img_b64 = data.get('image', '')
        if not img_b64:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        try:
            tmp_path = base64_to_jpeg_tempfile(img_b64)
        except Exception as e:
            return jsonify({'success': False,
                'error': f'Image could not be decoded: {e}'}), 400

        try:
            query_emb = get_embedding(tmp_path)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400

        conn    = get_db()
        persons = conn.execute(
            'SELECT id,name,age,gender,course,face_embedding,photo FROM persons'
        ).fetchall()
        conn.close()

        if not persons:
            return jsonify({'success': False,
                'error': 'No persons registered yet. Please register someone first.'}), 404

        best, best_sim = None, -1.0
        for p in persons:
            stored = np.array(json.loads(p['face_embedding']), np.float32)
            sim    = cosine_sim(query_emb, stored)
            if sim > best_sim:
                best_sim, best = sim, p

        if best and best_sim >= THRESHOLD:
            age = best['age']
            return jsonify({
                'success': True, 'matched': True,
                'confidence': round(best_sim * 100, 1),
                'person': {
                    'id':             best['id'],
                    'name':           best['name'],
                    'age':            age,
                    'gender':         best['gender'],
                    'course':         best['course'],
                    'classification': classify_age(age),
                    'photo':          photo_b64(best['photo'])
                }
            })
        return jsonify({'success': True, 'matched': False,
                        'message': 'No matching person found in the database.'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route('/api/persons', methods=['GET'])
def list_persons():
    conn = get_db()
    rows = conn.execute(
        'SELECT id,name,age,gender,course,registered_at,photo'
        ' FROM persons ORDER BY registered_at DESC'
    ).fetchall()
    conn.close()
    return jsonify({'success': True, 'total': len(rows), 'persons': [
        {'id': p['id'], 'name': p['name'], 'age': p['age'],
         'gender': p['gender'], 'course': p['course'],
         'classification': classify_age(p['age']),
         'registered_at': p['registered_at'],
         'photo': photo_b64(p['photo'])}
        for p in rows
    ]})


@app.route('/api/persons/<int:pid>', methods=['DELETE'])
def delete_person(pid):
    conn = get_db()
    conn.execute('DELETE FROM persons WHERE id=?', (pid,))
    conn.commit(); conn.close()
    return jsonify({'success': True, 'message': 'Person deleted'})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    conn   = get_db()
    total  = conn.execute('SELECT COUNT(*) FROM persons').fetchone()[0]
    adults = conn.execute('SELECT COUNT(*) FROM persons WHERE age>=18').fetchone()[0]
    teens  = conn.execute('SELECT COUNT(*) FROM persons WHERE age<18').fetchone()[0]
    conn.close()
    return jsonify({'total': total, 'adults': adults, 'teens': teens})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context='adhoc')