import os
import requests
import time
import uuid
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Config
HARUKI_API_BASE = "https://public-api.haruki.seiunx.com/alias/v1/music"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/embeddings"
SILICONFLOW_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "song_aliases"
VECTOR_SIZE = 1024
MUSICS_JSON_URL = "https://raw.githubusercontent.com/Team-Haruki/haruki-sekai-master/main/master/musics.json"

# Environment Variables
SILICONFLOW_KEY = os.getenv("SILICONFLOW_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")

# Fix Port Parsing: Handle empty string or missing var
port_str = os.getenv("QDRANT_PORT", "")
if not port_str:
    QDRANT_PORT = None # Let client decide (likely default or inferred from URL)
else:
    try:
        QDRANT_PORT = int(port_str)
    except ValueError:
        print(f"Warning: Invalid QDRANT_PORT '{port_str}', defaulting to None")
        QDRANT_PORT = None

if not SILICONFLOW_KEY or not QDRANT_HOST:
    print("Error: Missing env vars SILICONFLOW_API_KEY or QDRANT_HOST")
    exit(1)

# Initialize Qdrant
# Handle URL scheme to ensure HTTPS if needed for Zeabur/Cloud
url = QDRANT_HOST
if "zeabur.app" in url and not url.startswith("http"):
    url = "https://" + url

client = QdrantClient(url=url, port=QDRANT_PORT, api_key=QDRANT_KEY, https=None) # https=None lets it infer from URL scheme

# Ensure Collection
if not client.collection_exists(COLLECTION_NAME):
    print(f"Creating collection {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )

def get_embedding(text):
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": SILICONFLOW_MODEL,
        "input": text
    }
    resp = requests.post(SILICONFLOW_API_URL, json=data, headers=headers)
    if resp.status_code != 200:
        print(f"Embedding error: {resp.text}")
        return None
    
    json_resp = resp.json()
    if not json_resp.get("data"):
        return None
    return json_resp["data"][0]["embedding"]

def generate_uuid(mid, alias):
    raw = f"{mid}-{alias}"
    return str(uuid.UUID(hex=hashlib.md5(raw.encode('utf-8')).hexdigest()))

def index_song(mid):
    url = f"{HARUKI_API_BASE}/{mid}"
    resp = requests.get(url)
    
    if resp.status_code == 404:
        return 0
    if resp.status_code != 200:
        print(f"API Error for {mid}: {resp.status_code}")
        return 0

    data = resp.json()
    aliases = data.get("aliases", [])
    if not aliases:
        return 0

    points = []
    for alias in aliases:
        vec = get_embedding(alias)
        if not vec:
            continue
        
        point_id = generate_uuid(mid, alias)
        
        points.append(models.PointStruct(
            id=point_id,
            vector=vec,
            payload={
                "mid": mid,
                "alias": alias,
                "source": "haruki"
            }
        ))

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
    
    return len(points)

def fetch_music_ids():
    print(f"Fetching music list from {MUSICS_JSON_URL}...")
    resp = requests.get(MUSICS_JSON_URL)
    if resp.status_code != 200:
        print(f"Failed to fetch music list: {resp.status_code}")
        return []
    
    try:
        musics = resp.json()
        ids = [item['id'] for item in musics if 'id' in item]
        ids.sort()
        return ids
    except Exception as e:
        print(f"Error parsing music list: {e}")
        return []

def main():
    ids = fetch_music_ids()
    print(f"Found {len(ids)} songs.")
    
    total = 0
    for i in ids:
        try:
            count = index_song(i)
            if count > 0:
                print(f"Indexed song {i}: {count} aliases")
                total += count
            # time.sleep(0.1) # Rate limit if needed
        except Exception as e:
            print(f"Error indexing {i}: {e}")
            
    print(f"Complete. Total indexed: {total}")

if __name__ == "__main__":
    main()
