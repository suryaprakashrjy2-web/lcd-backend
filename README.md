# LCD Backend Server

This is the "brain" of your LCD personal voice assistant. It runs for free,
24/7, in the cloud — no laptop needs to stay on.

## What it does
- Receives text (already converted from your voice on the phone) at `/chat`
- Looks up your saved contacts/notes for context
- Asks Gemini (free tier) for a reply + a structured action
- Returns `{ "reply": "...", "action": {...} }` for the Android app to speak
  and execute

## Files
- `app.py` — Flask server, all routes
- `rag.py` — simple JSON-file store for contacts/notes (no external DB)
- `llm.py` — Gemini API wrapper
- `requirements.txt` — Python dependencies

## Step 1 — Get a free Gemini API key
1. Go to https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click "Create API key" → copy it. This is free tier, generous limits for
   personal use.

## Step 2 — Deploy for free on Render.com
1. Create a free account at https://render.com (GitHub login is easiest)
2. Push this `lcd-backend` folder to a new GitHub repo (see Step 3 below if
   you're not sure how)
3. On Render: **New +** → **Web Service** → connect your GitHub repo
4. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free
5. Under **Environment Variables**, add:
   - `GEMINI_API_KEY` = *(the key you copied in Step 1)*
6. Click **Create Web Service**. Wait ~2 minutes for the first deploy.
7. Render gives you a public URL like `https://lcd-backend-xxxx.onrender.com`
   — this is what the Android app will call instead of a local IP.

**Free tier note:** Render's free web services "spin down" after 15 minutes
of no traffic, and take ~30-50 seconds to "wake up" on the next request.
Since you use LCD occasionally, this just means the very first command after
a while might have a short delay before LCD replies — completely fine for
personal use. If that delay ever bothers you, a free tool like
[UptimeRobot](https://uptimerobot.com) can ping your `/health` endpoint every
few minutes to keep it always warm, still free.

## Step 3 — Push this folder to GitHub (if you haven't before)
```bash
cd lcd-backend
git init
git add .
git commit -m "LCD backend"
# create a new empty repo on github.com first, then:
git remote add origin https://github.com/<your-username>/lcd-backend.git
git branch -M main
git push -u origin main
```

## Step 4 — Test it's live
```bash
curl https://<your-render-url>/health
# should return {"status":"ok"}

curl -X POST https://<your-render-url>/contacts \
  -H "Content-Type: application/json" \
  -d '{"name":"Amma","phone":"+91XXXXXXXXXX","notes":"mother"}'

curl -X POST https://<your-render-url>/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Amma ki call cheyyi"}'
```

Once this responds correctly, the backend is done — next step is the
Android app, which will call this same `/chat` URL.

## Local testing (optional, before deploying)
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
python app.py
# server runs at http://127.0.0.1:5000
```
