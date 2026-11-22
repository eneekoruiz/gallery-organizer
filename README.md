# Gallery Organizer

Local photo and video management application with EXIF metadata parsing, face detection, and embedding-based clustering.

## Tech Stack

- **Backend:** Python 3.10+, FastAPI, Uvicorn, Flask (legacy modules)
- **Computer Vision:** OpenCV (`opencv-python-headless`), YuNet face detector, SFace face recognizer
- **Image Processing:** Pillow, `pillow-heif` (HEIC/HEIF Apple formats), NumPy
- **Validation:** Pydantic

## Architecture & Key Decisions

- **Local-First Indexing:** Scans local filesystem directories, extracts EXIF/XMP timestamps and GPS coordinates, and caches metadata in SQLite.
- **Face Clustering:** Uses OpenCV YuNet for face detection combined with SFace 128-D embedding vectors to group recurring individuals without sending data to external APIs.
- **Format Compatibility:** Translates HEIC/HEIF images and video containers into browser-compatible previews on demand.

## Local Setup

### Prerequisites

- Python 3.10+
- `libheif` support (installed automatically via `pillow-heif`)

### Installation & Run

```bash
# Clone and enter directory
git clone https://github.com/eneekoruiz/gallery-organizer.git
cd gallery-organizer

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn app_fastapi:app --host 127.0.0.1 --port 8000 --reload
```
