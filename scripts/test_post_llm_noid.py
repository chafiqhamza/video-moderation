import requests, json, sys
url = 'http://127.0.0.1:8000/api/recommendations/llm'
payload = {
    'video_title': 'test video',
    'retrieved_docs': [
        {'frame': 100, 'timestamp': 15.0, 'category': 'violence', 'blip': 'Someone punches', 'confidence': 0.92}
    ],
    'compliance_rate': 0.2
}
try:
    r = requests.post(url, json=payload, timeout=30)
except Exception as e:
    print('ERROR_POST:', e)
    sys.exit(2)
print('STATUS', r.status_code)
try:
    data = r.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    # If server saved a debug file, print its path
    if isinstance(data, dict) and data.get('llm_debug_file'):
        print('\nLLM_DEBUG_FILE:', data.get('llm_debug_file'))
except Exception:
    print('NONJSON_RESPONSE:\n', r.text[:4000])
