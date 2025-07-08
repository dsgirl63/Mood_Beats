from flask import Flask , render_template, request , url_for, jsonify
from flask_cors import CORS
import joblib
import random 
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
#Load the trained model and label encoder
model = joblib.load("mood_model.pkl")
le = joblib.load("label_encoder.pkl")

#predefined song database

song_db = {
    'Happy':[
        {'name':'Dil Dhadakne do', 'img':'happy1.jpg','file':'happy1.mp3'},
        {'name':'Hawa Hawai 2.0', 'img':'happy2.jpg','file':'happy2.mp3'},
        {'name':'Gallan Goodiyaan ', 'img':'happy3.jpg','file':'happy3.mp3'},
        {'name':'Lutt Putt Gaya', 'img':'happy4.jpg','file':'happy4.mp3'},
        {'name':'Haan ke Haan', 'img':'happy5.jpg','file':'happy5.mp3'},
        {'name':'Love You Zindagi', 'img':'happy6.jpg','file':'happy6.mp3'}


    ],

    'Sad':[
        {'name':'Baarish', 'img':'sad1.jpg','file':'sad1.mp3'},
        {'name':'Ranjha', 'img':'sad2.jpg','file':'sad2.mp3'},
        {'name':'Khairiyat', 'img':'sad3.jpg','file':'sad3.mp3'},
        {'name':'Bolna Mahi Bolna', 'img':'sad4.jpg','file':'sad4.mp3'},
        {'name':'Bin Tere', 'img':'sad5.jpg','file':'sad5.mp3'},
        {'name':'Teri Kher Mangdi', 'img':'sad6.jpg','file':'sad6.mp3'}
    ],
    'Motivational':[
        {'name':'Get Ready To Fight', 'img':'mot1.jpg','file':'mot1.mp3'},
        {'name':'Ghamand Kar', 'img':'mot2.jpg','file':'mot2.mp3'},
        {'name':'Chak Lein De', 'img':'mot3.jpg','file':'mot3.mp3'},
        {'name':'Zinda', 'img':'mot4.jpg','file':'mot4.mp3'},
        {'name':'Badal Pe Paon Hain', 'img':'mot5.jpg','file':'mot5.mp3'},
        {'name':'Jeete Hain Chal', 'img':'mot6.jpg','file':'mot6.mp3'}
    ],
    'Party':[
        {'name':'Ankh Marey', 'img':'party1.jpg','file':'party1.mp3'},
        {'name':'Tauba Tauba', 'img':'party2.jpg','file':'party2.mp3'},
        {'name':'Hookah Bar', 'img':'party3.jpg','file':'party13.mp3'},
        {'name':'Akhiyaan Gulaab ', 'img':'party4.jpg','file':'party4.mp3'},
        {'name':'Hauli Hauli', 'img':'party5.jpg','file':'party5.mp3'},
        {'name':'Morni Banke', 'img':'party6.jpg','file':'party6.mp3'}
    ]
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            energy = float(request.form['energy'])
            dance = float(request.form['danceability'])
            valence = float(request.form['valence'])

            prediction = model.predict([[energy, dance, valence]])[0]
            mood = le.inverse_transform([prediction])[0]  # Convert label index to label name

            songs = random.sample(song_db[mood], min(6, len(song_db[mood])))
            
            # Process songs to include proper URLs for images and audio files
            for song in songs:
                song['image_url'] = url_for('static', filename=f'images/{song["img"]}')
                song['audio_url'] = url_for('static', filename=f'audio/{song["file"]}')
                
            return render_template("index.html", mood=mood, songs=songs)
        except Exception as e:
            # Return JSON error for API use
            return jsonify({'error': str(e)}), 400

    return render_template("index.html", mood=None)

if __name__ == "__main__":
    app.run(debug=True,port=8080)