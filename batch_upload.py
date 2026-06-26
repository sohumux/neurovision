"""
NeuroVision — Batch Upload Script
Use this to bulk-register persons from a folder of photos.

Folder structure expected:
  photos/
    Arjun_Sharma.jpg
    Priya_Nair.jpg
    ...

Or use a CSV: data.csv with columns: name,age,gender,course,photo_path
"""

import requests
import base64
import csv
import os

BASE_URL = "http://localhost:5000"

def register_person(name, age, gender, course, image_path):
    """Register one person via the API."""
    if not os.path.exists(image_path):
        print(f"[SKIP] Image not found: {image_path}")
        return False

    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()

    # Detect image type
    ext = image_path.lower().split('.')[-1]
    media_type = 'image/jpeg' if ext in ('jpg', 'jpeg') else 'image/png'

    response = requests.post(f"{BASE_URL}/api/register", json={
        "name": name,
        "age": int(age),
        "gender": gender,
        "course": course,
        "image": f"data:{media_type};base64,{b64}"
    })

    result = response.json()
    if result.get('success'):
        print(f"[OK]  Registered: {name}")
        return True
    else:
        print(f"[ERR] Failed {name}: {result.get('error')}")
        return False


def bulk_upload_from_csv(csv_path):
    """Upload from a CSV file."""
    print(f"Loading from: {csv_path}")
    success, fail = 0, 0
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ok = register_person(
                name=row['name'],
                age=row['age'],
                gender=row['gender'],
                course=row['course'],
                image_path=row['photo_path']
            )
            if ok: success += 1
            else: fail += 1
    print(f"\nDone! ✓ {success} registered, ✗ {fail} failed")


def bulk_upload_from_folder(folder_path, age=20, gender="Unknown", course="Unknown"):
    """Upload all images from a folder. Filename (without ext) becomes the name."""
    print(f"Scanning folder: {folder_path}")
    success, fail = 0, 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ')
            path = os.path.join(folder_path, filename)
            ok = register_person(name, age, gender, course, path)
            if ok: success += 1
            else: fail += 1
    print(f"\nDone! ✓ {success} registered, ✗ {fail} failed")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python batch_upload.py csv data.csv")
        print("  python batch_upload.py folder ./photos")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == 'csv' and len(sys.argv) >= 3:
        bulk_upload_from_csv(sys.argv[2])
    elif mode == 'folder' and len(sys.argv) >= 3:
        bulk_upload_from_folder(sys.argv[2])
    else:
        print("Invalid arguments.")
