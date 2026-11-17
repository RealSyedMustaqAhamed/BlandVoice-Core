import urllib.request
import json
import time

prompts = [
    "Hi John, this is Sarah from Bland AI. Did you have a quick minute to chat about your enterprise phone agents?",
    "[filler:um] Let me check your account balance... [filler:uh_huh] Yes, it was processed this morning.",
    "[emotion:empathetic] I completely understand your frustration with the delayed shipment, and I will personally resolve this for you.",
    "[emotion:urgent] Security alert: A new login was attempted from an unrecognized device. Please verify your identity."
]

print("Testing live deployment at https://blandvoice-core.onrender.com...")

for p in prompts:
    url = "https://blandvoice-core.onrender.com/api/synthesize"
    payload = json.dumps({"text": p, "emotion": "neutral", "chunk_frames": 4}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=20) as resp:
            elapsed = time.time() - t0
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[OK 200] ({elapsed:.2f}s) '{p[:35]}...' -> Audio: {len(data['audio_base64'])} bytes, Duration: {data['duration_sec']}s")
    except Exception as e:
        print(f"[FAIL] '{p[:35]}...' -> Error: {e}")
