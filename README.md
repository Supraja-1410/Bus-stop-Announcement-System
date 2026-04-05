Neuro-Fuzzy Bus Stop Announcement System

A smart bus-stop announcement and travel-time prediction system built using Neuro-Fuzzy techniques (ANFIS).
This project simulates a real-world public transport system with multi-language audio announcements and intelligent ETA prediction based on dynamic conditions like weather, traffic, crowd, and time.

📌 Features
🔊 Automated Voice Announcements
Supports English, Hindi, and Telugu
Announces:
Upcoming stop (approach)
Arrival at stop
Door closing + next stop ETA
🧠 Neuro-Fuzzy Model (ANFIS)
Predicts travel time using:
Distance
Speed
Stop time
Weather conditions
Crowd level
Time of day
Weekday/Weekend
🌦️ Real-Time Weather Integration
Fetches live weather data
Adjusts speed and ETA accordingly
👥 Crowd-Aware System
Dynamically adjusts:
Bus speed
Stop waiting time
⏱️ Fuzzy Logic ETA Adjustment
Uses fuzzy rules for:
Rush hours
Weather conditions
Passenger density
🔁 Self-Learning System
Retrains model after each simulation run
📂 Project Structure
.
├── announcements11.py        # Main simulation script
├── bus_fuzzy_model.pth      # Trained Neuro-Fuzzy model
├── stops_1.csv              # Bus stop data (lat, long, sequence)
├── audio/                   # Generated announcement audio files
└── README.md
⚙️ Technologies Used
Python
PyTorch – Neuro-Fuzzy model
scikit-fuzzy – Fuzzy logic system
gTTS – Text-to-Speech
pygame – Audio playback
geopy – Distance calculation
pandas / numpy
🚀 How It Works
Load Bus Stops
Reads stop details from stops_1.csv
Initialize Model
Loads pre-trained model (.pth)
If not found → trains a new model
Simulate Bus Movement
Moves from one stop to another
Calculates:
Distance
Speed
Crowd impact
Weather impact
Predict Travel Time
Neuro-Fuzzy model predicts base time
Fuzzy logic adjusts ETA
Trigger Announcements
At 300m → approach announcement
At stop → arrival announcement
After stop → door closing + next ETA
Retraining
Model retrains after simulation for better accuracy
📊 Input Data Format (stops_1.csv)
stop_name,latitude,longitude,sequence
Stop A,17.3850,78.4867,1
Stop B,17.4000,78.5000,2
...
▶️ How to Run
1. Install Dependencies
pip install pandas numpy torch scikit-fuzzy gtts pygame geopy requests
2. Run the Simulation
python announcements11.py
🔈 Output
Real-time console logs:
Distance, speed, ETA, weather, crowd

Audio announcements generated in:

announcements_<session_id>/
🧠 Neuro-Fuzzy Model Details
Type: Sugeno-type ANFIS
Inputs:
Distance
Speed
Stop time
Weather
Weekend flag
Time-of-day factor
Output:
Travel time (seconds)
Uses:
Gaussian membership functions
Rule-based weighted output
📈 Fuzzy Logic Rules (Examples)
IF rush hour OR high crowd OR bad weather → ETA is slower
IF moderate conditions → ETA is normal
IF off-peak AND low crowd AND good weather → ETA is faster
🌍 Real-World Applications
Smart public transport systems
Metro/train announcement systems
Autonomous vehicle navigation
Intelligent traffic management
⚠️ Notes
Internet connection required for weather data
Audio playback depends on system audio drivers
Model retraining may take a few seconds
💡 Future Enhancements
GPS-based real-time tracking
Mobile app integration
Multi-route simulation
Deep learning hybrid models
Cloud-based deployment
