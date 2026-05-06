# Sirius Backend - Deploy Guide for Render

## Quick Deploy

### 1. Prerequisites
- GitHub account
- Render account (render.com)

### 2. Upload to GitHub

Create a new repository called `sirius-backend` and upload the `backend/` folder.

### 3. Deploy on Render

1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Select your GitHub repository
4. Configure:
   - **Name**: sirius-backend
   - **Branch**: main
   - **Build Command**: (leave empty - uses requirements.txt)
   - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free (or select paid)

5. Click "Create Web Service"

### 4. Set Environment Variables

In your Render dashboard, go to "Environment" and add:

```
MONGO_URL=your_mongodb_atlas_connection_string
DB_NAME=sirius_production
GOOGLE_GEMINI_API_KEY=your_api_key (optional)
JWT_SECRET=generate_a_random_string
FRONTEND_URL=https://your-frontend.onrender.com
ENVIRONMENT=production
```

### 5. Get Your Backend URL

After deploy, you'll get a URL like:
`https://sirius-backend.onrender.com`

Copy this URL!

---

## Frontend Configuration

After backend is deployed, update your frontend:

1. Open `frontend/.env`
2. Add: `REACT_APP_BACKEND_URL=https://sirius-backend.onrender.com`
3. Set `OFFLINE_MODE=false` in `frontend/src/lib/offline-mode.js`
4. Rebuild and deploy frontend to Render or Netlify

---

## MongoDB Atlas Setup

1. Go to https://www.mongodb.com/cloud/atlas
2. Create free account → Free Cluster
3. Create database user (username/password)
4. Get connection string: "Connect" → "Connect your application"
5. Replace `<password>` in MONGO_URL

---

## Troubleshooting

### 500 Error on first run
- Check environment variables are set
- Check MongoDB connection string is correct

### Slow cold starts
- This is normal on free tier (first request after 15min may take 10-15s)

### WebSocket issues
- Render free tier doesn't support websockets well
- Consider using paid plan for real-time features
