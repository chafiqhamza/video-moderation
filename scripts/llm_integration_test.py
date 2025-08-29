"""Integration test: POST a representative RAG payload to /api/recommendations/llm
and assert that the response returns an 'ok' status with a recommendations array
where each recommendation has required fields.

Run: python scripts/llm_integration_test.py
"""
import sys
import json
import time

try:
    import requests
except Exception:
    print('requests library required: pip install requests')
    sys.exit(2)

API = 'http://127.0.0.1:8000/api/recommendations/llm'

sample = {
    'video_id': None,
    'video_title': 'Integration Test Video',
    'retrieved_docs': [
        {
            'frame': 142,
            'timestamp': 3.7,
            'category': 'copyright',
            'blip': 'Short clip from CopyRightFix',
            'ocr': '',
            'policy': {'description': 'Copyrighted material', 'action_required': 'Remove'},
            'personalized_reason': 'Identified a copyrighted clip from a known source.'
        }
    ]
}

print('Posting test payload to', API)
try:
    r = requests.post(API, json=sample, timeout=180)
except Exception as e:
    print('Request failed:', e)
    sys.exit(3)

try:
    data = r.json()
except Exception:
    print('Non-JSON response status:', r.status_code)
    print(r.text[:2000])
    sys.exit(4)

print('Response status:', data.get('status'))

# If unparsable or validation failed, print raw and exit non-zero
if data.get('status') in ('llm_unparsable', 'llm_validation_failed'):
    print('LLM returned unparsable/invalid output. Status:', data.get('status'))
    print('Message:', data.get('message'))
    if data.get('llm_raw'):
        print('LLM raw (truncated):')
        print(data.get('llm_raw')[:2000])
    sys.exit(5)

if data.get('status') != 'ok':
    print('Unexpected status:', data.get('status'))
    print(json.dumps(data, indent=2)[:4000])
    sys.exit(6)

recs = data.get('recommendations')
if not isinstance(recs, list) or len(recs) == 0:
    print('No recommendations returned')
    sys.exit(7)

required_fields = ['category', 'description', 'suggested_action']
errors = 0
for i, rec in enumerate(recs):
    if not isinstance(rec, dict):
        print(f'Recommendation {i} is not an object: {rec}')
        errors += 1
        continue
    for f in required_fields:
        if f not in rec:
            print(f'Recommendation {i} missing required field: {f}')
            errors += 1
        else:
            if f == 'category' or f == 'description' or f == 'suggested_action':
                if not isinstance(rec[f], str):
                    print(f'Recommendation {i} field {f} has wrong type: {type(rec[f])}')
                    errors += 1

if errors:
    print('Validation failed with', errors, 'errors')
    print('\nFull response (truncated):')
    print(json.dumps(data, indent=2)[:8000])
    sys.exit(8)

print('Integration test passed: recommendations present and schema looks valid')
print('\nExample recommendation:\n')
print(json.dumps(recs[0], indent=2)[:4000])
sys.exit(0)
