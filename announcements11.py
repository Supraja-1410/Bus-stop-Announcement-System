#!/usr/bin/env python3
"""
Neuro-Fuzzy Bus Simulation (Sugeno-type ANFIS)
- Telugu + Hindi + English announcements
- Approach announcement (300 m fuzzy-based trigger)
- Arrival and Doors-closing + ETA announcement
- Travel time predicted using trained neuro-fuzzy model
- Weekday vs Weekend, time-of-day, crowd, and weather integrated
"""

import pandas as pd
import time, os, uuid
from gtts import gTTS
from geopy.distance import geodesic
from datetime import datetime
import pygame
import numpy as np
import torch
import torch.nn as nn
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------------- CONFIG ----------------
CITY = "Hyderabad"
ANN_DIR_PREFIX = "announcements_"
MODEL_FILE = "bus_fuzzy_model.pth"
ANNOUNCEMENT_DISTANCE_KM = 0.3  # 300 meters

WEATHER_SPEED_FACTOR = {
    "Clear": 1.0, "Clouds": 0.95, "Drizzle": 0.85, "Rain": 0.80,
    "Thunderstorm": 0.60, "Snow": 0.70, "Mist": 0.75, "Fog": 0.70,
    "Haze": 0.85, "Default": 0.90
}
WEATHER_MAP = {w: i for i,w in enumerate(WEATHER_SPEED_FACTOR.keys())}

# ---------------- HELPERS ----------------
def is_weekend():
    return datetime.now().weekday() >= 5

def fetch_current_weather():
    try:
        import requests
        url = f"https://wttr.in/{CITY}?format=j1"
        resp = requests.get(url, timeout=5)
        j = resp.json()
        weather_desc = j["current_condition"][0]["weatherDesc"][0]["value"]
        main = "Default"
        desc = weather_desc.lower()
        if "rain" in desc: main = "Rain"
        elif "drizzle" in desc: main = "Drizzle"
        elif "thunder" in desc: main = "Thunderstorm"
        elif "fog" in desc: main = "Fog"
        elif "mist" in desc: main = "Mist"
        elif "snow" in desc: main = "Snow"
        elif "cloud" in desc: main = "Clouds"
        elif "clear" in desc: main = "Clear"
        elif "haze" in desc: main = "Haze"
        return main, weather_desc
    except:
        return "Default", "Unavailable"

def weather_to_fuzzy_value(weather_main):
    mapping = {
        "Clear": 10,
        "Clouds": 40,
        "Drizzle": 60,
        "Rain": 80,
        "Thunderstorm": 90,
        "Mist": 70,
        "Fog": 90,
        "Haze": 50,
        "Snow": 85,
        "Default": 30
    }
    return mapping.get(weather_main, 30)

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

def get_time_of_day_factor():
    hour = datetime.now().hour
    if is_weekend():
        if 8 <= hour <= 10 or 18 <= hour <= 21: return 0.9
        else: return 1.0
    else:
        if 7 <= hour <= 9 or 17 <= hour <= 20: return 0.65
        elif 10 <= hour <= 16: return 0.85
        else: return 1.05

def get_crowd_level():
    now = datetime.now()
    hour = now.hour
    weekend_flag = is_weekend()
    if weekend_flag:
        if 10 <= hour <= 13 or 17 <= hour <= 21:
            return "Medium"
        else:
            return "Low"
    else:
        if 7 <= hour <= 9 or 17 <= hour <= 20:
            return "High"
        elif 10 <= hour <= 16:
            return "Medium"
        else:
            return "Low"

def get_stop_time_based_on_crowd(stop_name, crowd_level):
    base = 30 if "station" in stop_name.lower() else 8
    if crowd_level == "High":
        return int(base * 2.5)
    elif crowd_level == "Medium":
        return int(base * 1.6)
    else:
        return base

def get_stop_time(stop_name):
    crowd = get_crowd_level()
    return get_stop_time_based_on_crowd(stop_name, crowd)

def adjust_speed_by_crowd(speed_kmh, crowd_level):
    if crowd_level == "High":
        return speed_kmh * 0.7
    elif crowd_level == "Medium":
        return speed_kmh * 0.85
    else:
        return speed_kmh

# ---------------- NEURO-FUZZY MODEL ----------------
class SugenoFuzzyModel(nn.Module):
    def __init__(self, n_rules=5, input_size=6):
        super().__init__()
        self.n_rules = n_rules
        self.input_size = input_size
        self.weights = nn.Parameter(torch.randn(n_rules, input_size))
        self.bias = nn.Parameter(torch.randn(n_rules, 1))
        self.centers = nn.Parameter(torch.randn(n_rules, input_size))
        self.widths = nn.Parameter(torch.rand(n_rules, input_size)+0.5)

    def forward(self, x):
        x_exp = x.unsqueeze(1).repeat(1, self.n_rules, 1)
        diff = x_exp - self.centers
        gauss = torch.exp(-0.5 * (diff/self.widths)**2)
        firing_strength = torch.prod(gauss, dim=2)
        rule_output = torch.matmul(x, self.weights.T) + self.bias.T
        output = (firing_strength * rule_output).sum(dim=1) / (firing_strength.sum(dim=1)+1e-6)
        return output.unsqueeze(1)

# ---------------- ANNOUNCEMENTS ----------------
class Announcer:
    def __init__(self, session_id):
        self.ann_dir = ANN_DIR_PREFIX + session_id
        os.makedirs(self.ann_dir, exist_ok=True)
        pygame.mixer.init()

    def text_to_speech(self, text, lang='en', prefix="ann"):
        safe_name = f"{prefix}_{lang}_{int(time.time()*1000)}.mp3"
        path = os.path.join(self.ann_dir, safe_name)
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(path)
            return path if os.path.exists(path) else None
        except:
            return None

    def play_audio(self, path):
        if not path or not os.path.exists(path): return
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)

    def announce_approach(self, stop_name, distance_remaining):
        en = f"Next stop: {stop_name}. Prepare to exit. ({distance_remaining:.2f} km left)"
        hi = f"अगला स्टॉप: {stop_name}. उतरने की तैयारी करें। ({distance_remaining:.2f} km बचा)"
        te = f"తదుపరి స్టాప్: {stop_name}. దిగడానికి సిద్ధం అవ్వండి. ({distance_remaining:.2f} km మిగిలింది)"
        for lang, msg in [('en', en), ('hi', hi), ('te', te)]:
            self.play_audio(self.text_to_speech(msg, lang, "approach"))

    def announce_arrival(self, stop_name):
        en = f"Now arriving at {stop_name}. Doors opening on the right."
        hi = f"अब {stop_name} पहुंच रहे हैं. दरवाजे दाईं ओर खुल रहे हैं।"
        te = f"{stop_name} వద్దకు చేరుకుంటున్నాము. కుడి వైపు తలుపులు తెరుచుకుంటాయి."
        for lang, msg in [('en', en), ('hi', hi), ('te', te)]:
            self.play_audio(self.text_to_speech(msg, lang, "arrival"))

    def announce_door_close(self, next_stop_name, eta_min):
        en = f"Doors closing. ETA to {next_stop_name}: {eta_min:.1f} minutes."
        hi = f"दरवाजे बंद हो रहे हैं. {next_stop_name} तक ETA {eta_min:.1f} मिनट है।"
        te = f"తలుపులు మూసుకుంటున్నాయి. {next_stop_name} చేరుకునే అంచనా సమయం {eta_min:.1f} నిమిషాలు."
        for lang, msg in [('en', en), ('hi', hi), ('te', te)]:
            self.play_audio(self.text_to_speech(msg, lang, "doorclose"))

# ---------------- FUZZY ETA SYSTEM ----------------
time_of_day = ctrl.Antecedent(np.arange(0,24,1), 'time_of_day')
weekend = ctrl.Antecedent(np.arange(0,2,1), 'weekend')
crowd = ctrl.Antecedent(np.arange(0,101,1), 'crowd')
weather = ctrl.Antecedent(np.arange(0,101,1), 'weather')
eta_factor = ctrl.Consequent(np.arange(0.5, 2.0, 0.01), 'eta_factor')

time_of_day['off_peak'] = fuzz.trimf(time_of_day.universe, [0,0,10])
time_of_day['moderate'] = fuzz.trimf(time_of_day.universe, [8,12,16])
time_of_day['rush'] = fuzz.trimf(time_of_day.universe, [15,18,23])

weekend['weekday'] = fuzz.trimf(weekend.universe, [0,0,0.5])
weekend['weekend'] = fuzz.trimf(weekend.universe, [0.5,1,1])

crowd['low'] = fuzz.trimf(crowd.universe, [0,0,30])
crowd['medium'] = fuzz.trimf(crowd.universe, [20,50,80])
crowd['high'] = fuzz.trimf(crowd.universe, [70,100,100])

weather['good'] = fuzz.trimf(weather.universe, [0,0,30])
weather['moderate'] = fuzz.trimf(weather.universe, [20,50,80])
weather['bad'] = fuzz.trimf(weather.universe, [70,100,100])

eta_factor['faster'] = fuzz.trimf(eta_factor.universe, [0.5,0.7,1.0])
eta_factor['normal'] = fuzz.trimf(eta_factor.universe, [0.9,1.0,1.1])
eta_factor['slower'] = fuzz.trimf(eta_factor.universe, [1.0,1.3,2.0])

rule1 = ctrl.Rule(time_of_day['rush'] | crowd['high'] | weather['bad'], eta_factor['slower'])
rule2 = ctrl.Rule(time_of_day['moderate'] | crowd['medium'] | weather['moderate'], eta_factor['normal'])
rule3 = ctrl.Rule(time_of_day['off_peak'] & crowd['low'] & weather['good'], eta_factor['faster'])
rule4 = ctrl.Rule(weekend['weekend'] & crowd['medium'], eta_factor['normal'])

eta_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
eta_sim = ctrl.ControlSystemSimulation(eta_ctrl)

# ---------------- BUS SIMULATION ----------------
class NeuroFuzzyBusSimulation:
    def __init__(self, stops_file="stops_1.csv"):
        self.stops_df = pd.read_csv(stops_file)
        required_cols = {"stop_name", "latitude", "longitude", "sequence"}
        if not required_cols.issubset(self.stops_df.columns):
            raise RuntimeError(f"CSV must contain {required_cols}")

        self.base_speed = 50.0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.announcer = Announcer(self.session_id)
        self.model = SugenoFuzzyModel()

        if os.path.exists(MODEL_FILE):
            try:
                self.model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
                self.model.eval()
                print("✅ Neuro-fuzzy model loaded successfully.")
            except Exception as e:
                print(f"⚠️ Could not load model ({e}). Training new model...")
                self.train_model()
        else:
            print("⚠️ Model not found. Training new model...")
            self.train_model()

    def train_model(self):
        X, y = [], []
        for i in range(len(self.stops_df)-1):
            f, t = self.stops_df.iloc[i], self.stops_df.iloc[i+1]
            distance = calculate_distance(f['latitude'], f['longitude'], t['latitude'], t['longitude'])
            stop_time = get_stop_time(t['stop_name'])
            weekend_flag = 1.0 if is_weekend() else 0.0
            features = [distance, self.base_speed, stop_time, WEATHER_MAP["Default"], weekend_flag, get_time_of_day_factor()]
            X.append(features)
            y.append(distance/self.base_speed*3600 + stop_time)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1,1)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        for epoch in range(300):
            optimizer.zero_grad()
            pred = self.model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        torch.save(self.model.state_dict(), MODEL_FILE)
        self.model.eval()
        print("✅ Neuro-fuzzy model trained and saved.")

    def predict_travel_time(self, distance_km, speed_kmh, stop_time_s, weekend_flag, crowd_level, weather_main):
        x = torch.tensor([[distance_km, speed_kmh, stop_time_s, WEATHER_MAP["Default"], weekend_flag, get_time_of_day_factor()]], dtype=torch.float32)
        with torch.no_grad():
            pred = float(self.model(x).item())

        hour = datetime.now().hour
        crowd_val = {"Low": 15, "Medium": 50, "High": 85}.get(crowd_level, 50)
        weather_val = weather_to_fuzzy_value(weather_main)

        eta_sim.input['time_of_day'] = hour
        eta_sim.input['weekend'] = weekend_flag
        eta_sim.input['crowd'] = crowd_val
        eta_sim.input['weather'] = weather_val
        eta_sim.compute()
        factor = eta_sim.output['eta_factor']

        pred *= factor
        pred = max(pred, distance_km / max(speed_kmh,5)*3600 + stop_time_s)
        return pred

    def simulate_bus_movement(self, from_stop, to_stop, next_stop=None):
        f_lat, f_lon = float(from_stop['latitude']), float(from_stop['longitude'])
        t_lat, t_lon = float(to_stop['latitude']), float(to_stop['longitude'])
        weather_main, weather_desc = fetch_current_weather()
        weekend_flag = 1.0 if is_weekend() else 0.0

        distance_km = calculate_distance(f_lat, f_lon, t_lat, t_lon)
        crowd_level = get_crowd_level()
        stop_time = get_stop_time_based_on_crowd(to_stop['stop_name'], crowd_level)
        base_speed = self.base_speed * WEATHER_SPEED_FACTOR.get(weather_main, 0.9) * get_time_of_day_factor()
        speed_kmh = adjust_speed_by_crowd(base_speed, crowd_level)
        travel_secs = self.predict_travel_time(distance_km, speed_kmh, stop_time, weekend_flag, crowd_level, weather_main)

        elapsed = 0
        approach_announced = False
        print(f"\n🚌 {from_stop['stop_name']} → {to_stop['stop_name']}")
        print(f"🌦️ {weather_main} ({weather_desc}) | {distance_km:.2f} km | speed {speed_kmh:.1f} km/h | crowd {crowd_level} | stop {stop_time}s | ETA {travel_secs/60:.2f} min")

        while elapsed < travel_secs:
            rem_distance = distance_km * (1 - elapsed / travel_secs) if travel_secs > 0 else 0
            if not approach_announced and rem_distance <= ANNOUNCEMENT_DISTANCE_KM:
                self.announcer.announce_approach(to_stop['stop_name'], rem_distance)
                approach_announced = True
            time.sleep(1)
            elapsed += 1

        self.announcer.announce_arrival(to_stop['stop_name'])
        time.sleep(stop_time)

        if next_stop is not None:
            n_lat, n_lon = float(next_stop['latitude']), float(next_stop['longitude'])
            next_distance = calculate_distance(t_lat, t_lon, n_lat, n_lon)
            next_crowd = get_crowd_level()
            next_stop_time = get_stop_time_based_on_crowd(next_stop['stop_name'], next_crowd)
            next_speed = adjust_speed_by_crowd(self.base_speed * WEATHER_SPEED_FACTOR.get(weather_main, 0.9) * get_time_of_day_factor(), next_crowd)
            next_eta = self.predict_travel_time(next_distance, next_speed, next_stop_time, weekend_flag, next_crowd, weather_main)/60.0
            self.announcer.announce_door_close(next_stop['stop_name'], next_eta)

    def start_simulation(self):
        n = len(self.stops_df)
        for i in range(n-1):
            from_stop = self.stops_df.iloc[i]
            to_stop = self.stops_df.iloc[i+1]
            next_stop = self.stops_df.iloc[i+2] if i+2<n else None
            self.simulate_bus_movement(from_stop, to_stop, next_stop)

        print("\n✅ Simulation complete!")
        print("🔁 Re-training model based on this trip’s conditions...\n")
        self.train_model()   # <-- ✅ Auto-retrain after simulation


# ---------------- MAIN ----------------
if __name__ == "__main__":
    sim = NeuroFuzzyBusSimulation("stops_1.csv")
    sim.start_simulation()
