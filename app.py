from flask import Flask, render_template, request, jsonify
import sqlite3, base64, os, json, io, traceback, time
import numpy as np
from PIL import Image, ImageOps

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
os.makedirs('database', exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL CACHE — loaded ONCE at startup, reused on every request
# ═══════════════════════════════════════════════════════════════════════════════
_MODEL_LOADED = False
_DB_CACHE     = {}      # { person_id : np.array(512,) }
_DB_CACHE_TS  = 0.0

def load_model_once():
    """Pre-warm the Facenet512 model so first request is fast."""
    global _MODEL_LOADED
    if _MODEL_LOADED:
        return
    print("[NeuroVision] Warming up Facenet512 model...")
    t0 = time.time()
    try:
        from deepface import DeepFace
        dummy = np.ones((160, 160, 3), dtype=np.uint8) * 128
        DeepFace.represent(
            img_path          = dummy,
            model_name        = 'Facenet',
            enforce_detection = False,
            detector_backend  = 'skip',
        )
        _MODEL_LOADED = True
        print(f"[NeuroVision] Model warm in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"[NeuroVision] Warmup error (non-fatal): {e}")
        _MODEL_LOADED = True   # continue anyway

def refresh_db_cache():
    """Load all face embeddings from SQLite into RAM numpy arrays."""
    global _DB_CACHE, _DB_CACHE_TS
    conn = get_db()
    rows = conn.execute('SELECT id, face_embedding FROM persons').fetchall()
    conn.close()
    _DB_CACHE = {
        r['id']: np.array(json.loads(r['face_embedding']), dtype=np.float32)
        for r in rows
    }
    _DB_CACHE_TS = time.time()
    print(f"[NeuroVision] DB cache: {len(_DB_CACHE)} persons")

# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE DECODER — Pure Pillow, no OpenCV needed
# ═══════════════════════════════════════════════════════════════════════════════
def decode_image(image_data) -> Image.Image:
    if isinstance(image_data, (bytes, bytearray)):
        raw = bytes(image_data)
    else:
        s = image_data
        if ',' in s:
            s = s.split(',', 1)[1]
        s = s.strip().replace('\n','').replace('\r','').replace(' ','')
        s += '=' * (-len(s) % 4)
        raw = base64.b64decode(s)

    img = Image.open(io.BytesIO(raw))
    img.load()

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode in ('I', 'F'):
        arr = np.array(img, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        arr = ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
        img = Image.fromarray(arr, 'L')

    if img.mode in ('RGBA', 'LA'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img.convert('RGB'), mask=img.split()[-1])
        img = bg
    elif img.mode == 'P':
        img = img.convert('RGBA')
        bg  = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img.convert('RGB'), mask=img.split()[-1])
        img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    return img


def resize_for_speed(img: Image.Image, max_w: int = 480) -> Image.Image:
    """Shrink large images — smaller = faster detection."""
    if img.width > max_w:
        ratio = max_w / img.width
        return img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
    return img


def pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    img = resize_for_speed(img, 640)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=88)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
#  FAST EMBEDDING
#  Detector priority: opencv (~0.05s) → ssd (~0.1s) → mtcnn (~0.3s)
#  Pass numpy array directly — no temp file I/O
# ═══════════════════════════════════════════════════════════════════════════════
DETECTORS  = ['opencv', ]
MODEL_NAME = 'Facenet512'
THRESHOLD  = 0.70

def get_embedding(img_pil: Image.Image) -> np.ndarray:
    from deepface import DeepFace

    # Resize to 480px wide for faster detector pass
    small = resize_for_speed(img_pil, 480)
    img_arr = np.array(small, dtype=np.uint8)

    last_err = None
    for detector in DETECTORS:
        try:
            t0 = time.time()
            result = DeepFace.represent(
                img_path          = img_arr,
                model_name        = MODEL_NAME,
                enforce_detection = True,
                detector_backend  = detector,
                align             = True,
            )
            print(f"[NeuroVision] {detector}: {time.time()-t0:.2f}s")
            if result:
                return np.array(result[0]['embedding'], dtype=np.float32)
        except Exception as e:
            last_err = e
            print(f"[NeuroVision] {detector} failed: {e}")
            continue

    raise ValueError(
        f'No face detected. Use a clear, well-lit, frontal photo. '
        f'(Tried: {DETECTORS}, last error: {last_err})'
    )


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def fast_search(query_emb: np.ndarray):
    """
    Vectorised 1:N cosine search against RAM-cached embeddings.
    Returns (person_id, similarity_score).
    ~0.001s for 1000 persons.
    """
    # Refresh cache if stale (>30s) or empty
    if not _DB_CACHE or (time.time() - _DB_CACHE_TS > 30):
        refresh_db_cache()

    if not _DB_CACHE:
        return None, -1.0

    ids  = list(_DB_CACHE.keys())
    mat  = np.stack(list(_DB_CACHE.values()))        # (N, 512)
    q    = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    sims  = (mat / norms).dot(q)                     # vectorised dot product
    best  = int(np.argmax(sims))
    return ids[best], float(sims[best])


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
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_db():
    c = sqlite3.connect('database/neurovision.db')
    c.row_factory = sqlite3.Row
    return c

def classify_age(age):
    return 'Adult' if int(age) >= 18 else 'Teen'

def photo_b64(blob):
    if not blob:
        return None
    return 'data:image/jpeg;base64,' + base64.b64encode(bytes(blob)).decode()


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/warmup', methods=['GET'])
def warmup():
    """Call once after starting server to pre-load model."""
    load_model_once()
    refresh_db_cache()
    return jsonify({
        'success': True,
        'model_loaded': _MODEL_LOADED,
        'persons_cached': len(_DB_CACHE),
    })


@app.route('/api/ping', methods=['GET'])
def ping():
    return jsonify({
        'status': 'online',
        'model_loaded': _MODEL_LOADED,
        'persons_in_cache': len(_DB_CACHE),
    })


@app.route('/api/register', methods=['POST'])
def register_person():
    t0 = time.time()
    try:
        data   = request.get_json(force=True)
        name   = data.get('name',   '').strip()
        age    = int(data.get('age', 0))
        gender = data.get('gender', '').strip()
        course = data.get('course', '').strip()
        img_b64= data.get('image',  '')

        if not all([name, age, gender, course, img_b64]):
            return jsonify({'success': False, 'error': 'All fields are required'}), 400

        try:
            img_pil = decode_image(img_b64)
        except Exception as e:
            return jsonify({'success': False,
                'error': f'Image decode failed: {e}'}), 400

        try:
            embedding = get_embedding(img_pil)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400

        jpeg_bytes = pil_to_jpeg_bytes(img_pil)

        conn = get_db()
        conn.execute(
            'INSERT INTO persons(name,age,gender,course,face_embedding,photo)'
            ' VALUES(?,?,?,?,?,?)',
            (name, age, gender, course, json.dumps(embedding.tolist()), jpeg_bytes)
        )
        conn.commit()
        conn.close()

        # Invalidate RAM cache so next identify picks up the new person
        _DB_CACHE.clear()

        return jsonify({
            'success': True,
            'message': f'{name} registered!',
            'time_seconds': round(time.time() - t0, 2),
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500


@app.route('/api/identify', methods=['POST'])
def identify_person():
    t0 = time.time()
    try:
        img_b64 = request.get_json(force=True).get('image', '')
        if not img_b64:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        try:
            img_pil = decode_image(img_b64)
        except Exception as e:
            return jsonify({'success': False,
                'error': f'Image decode failed: {e}'}), 400

        try:
            query_emb = get_embedding(img_pil)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400

        best_id, best_sim = fast_search(query_emb)
        total = round(time.time() - t0, 2)
        print(f"[NeuroVision] Total identify time: {total}s")

        if best_id is None:
            return jsonify({'success': False,
                'error': 'No persons registered in database.'}), 404

        if best_sim >= THRESHOLD:
            conn = get_db()
            p = conn.execute('SELECT * FROM persons WHERE id=?',
                             (best_id,)).fetchone()
            conn.close()
            age = p['age']
            return jsonify({
                'success':      True,
                'matched':      True,
                'confidence':   round(best_sim * 100, 1),
                'time_seconds': total,
                'person': {
                    'id':             p['id'],
                    'name':           p['name'],
                    'age':            age,
                    'gender':         p['gender'],
                    'course':         p['course'],
                    'classification': classify_age(age),
                    'photo':          photo_b64(p['photo']),
                }
            })

        return jsonify({
            'success':      True,
            'matched':      False,
            'message':      'No matching person found.',
            'time_seconds': total,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500


@app.route('/api/persons', methods=['GET'])
def list_persons():
    conn = get_db()
    rows = conn.execute(
        'SELECT id,name,age,gender,course,registered_at,photo'
        ' FROM persons ORDER BY registered_at DESC'
    ).fetchall()
    conn.close()
    return jsonify({
        'success': True,
        'total':   len(rows),
        'persons': [{
            'id':             r['id'],
            'name':           r['name'],
            'age':            r['age'],
            'gender':         r['gender'],
            'course':         r['course'],
            'classification': classify_age(r['age']),
            'registered_at':  r['registered_at'],
            'photo':          photo_b64(r['photo']),
        } for r in rows]
    })


@app.route('/api/persons/<int:pid>', methods=['DELETE'])
def delete_person(pid):
    conn = get_db()
    conn.execute('DELETE FROM persons WHERE id=?', (pid,))
    conn.commit()
    conn.close()
    _DB_CACHE.pop(pid, None)
    return jsonify({'success': True, 'message': 'Person deleted'})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    conn = get_db()
    total  = conn.execute('SELECT COUNT(*) FROM persons').fetchone()[0]
    adults = conn.execute('SELECT COUNT(*) FROM persons WHERE age>=18').fetchone()[0]
    teens  = conn.execute('SELECT COUNT(*) FROM persons WHERE age<18').fetchone()[0]
    conn.close()
    return jsonify({'total': total, 'adults': adults, 'teens': teens})


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(
        debug     = False,
        host      = '0.0.0.0',
        port      = 5000,
        ssl_context = 'adhoc',
        threaded  = True,
    )
