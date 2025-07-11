# Mood-Beats

## Local Hosting

### Backend (Flask)
1. Install Python dependencies:
   ```bash
   pip install flask flask-cors joblib
   ```
2. Run the backend server:
   ```bash
   python app.py
   ```
   By default, it runs at http://127.0.0.1:8080

### Frontend (Static)
1. Open `frontend/index.html` directly in your browser for local static testing.
2. For full API functionality, deploy the backend to a public URL and update the `API_URL` in `frontend/index.html`.

---

## Deploying to Netlify

1. Place all frontend files (HTML, static assets) in the `frontend/` directory.
2. In `frontend/index.html`, set `API_URL` in the script to your deployed backend URL (e.g., Render, Railway, Heroku).
3. Add the provided `_redirects` file in `frontend/` if you want to proxy API calls (edit `YOUR_BACKEND_URL`).
4. Drag and drop the `frontend/` folder in the Netlify dashboard or use the Netlify CLI:
   ```bash
   netlify deploy --dir=frontend
   ```

---

## Notes
- The backend (Flask) cannot be hosted on Netlify. Use a service like Render, Railway, or Heroku.
- The frontend (static) is Netlify-ready.
- For mobile responsiveness and a modern look, the UI is already optimized.
- Update the API endpoint in the frontend as needed.
