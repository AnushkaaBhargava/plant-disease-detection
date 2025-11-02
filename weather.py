import requests
import statistics

# -----------------------
# Config
# -----------------------
API_KEY = "31d0b53cae3d1255b65b2a43597eccd4"  # OpenWeather API Key
LAT, LON = 28.6139, 77.2090    # Example: Delhi coordinates
URL = f"http://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

# -----------------------
# Fetch Weather Data
# -----------------------
response = requests.get(URL)
data = response.json()

if response.status_code != 200:
    print("Error:", data.get("message", "Failed to fetch data"))
    exit()

temps, rainfall, humidity, wind_speeds = [], [], [], []

for entry in data["list"]:
    temps.append(entry["main"]["temp"])
    humidity.append(entry["main"]["humidity"])
    wind_speeds.append(entry["wind"]["speed"])
    rain = entry.get("rain", {}).get("3h", 0)
    rainfall.append(rain)

avg_temp = statistics.mean(temps)
avg_rainfall = statistics.mean(rainfall)
avg_humidity = statistics.mean(humidity)
avg_wind = statistics.mean(wind_speeds)

# -----------------------
# Mock Soil Nutrient Data (replace with real dataset/API)
# -----------------------
# Example: soil fertility at LAT, LON
soil_data = {
    "nitrogen": 45,    # kg/ha
    "phosphorus": 12,  # kg/ha
    "potassium": 110   # kg/ha
}

# -----------------------
# Display
# -----------------------
print(f"Weather & Soil Summary at ({LAT}, {LON}):")
print(f"- Avg Temperature: {avg_temp:.2f} °C")
print(f"- Avg Rainfall: {avg_rainfall:.2f} mm (per 3h window)")
print(f"- Avg Humidity: {avg_humidity:.2f} %")
print(f"- Avg Wind Speed: {avg_wind:.2f} m/s")
print(f"- Soil Nitrogen: {soil_data['nitrogen']} kg/ha")
print(f"- Soil Phosphorus: {soil_data['phosphorus']} kg/ha")
print(f"- Soil Potassium: {soil_data['potassium']} kg/ha")

# -----------------------
# Crop Suggestion (example logic)
# -----------------------
if soil_data["nitrogen"] < 20:
    print("🌱 Soil is nitrogen-deficient → consider legumes (e.g., Soybean, Groundnut) to fix N.")
elif soil_data["phosphorus"] < 15:
    print("🌱 Low phosphorus → consider crops like Lentils, Peas, or apply P fertilizer.")
elif soil_data["potassium"] < 50:
    print("🌱 Low potassium → crops like Potatoes, Sugarcane, or add potash fertilizers.")
else:
    if avg_rainfall > 5:
        print("🌱 Suggestion: Water-loving crops like Rice, Jute, Sugarcane.")
    elif avg_temp > 30:
        print("🌱 Suggestion: Heat-tolerant crops like Millet, Sorghum, Cotton.")
    elif avg_temp < 20:
        print("🌱 Suggestion: Cool-season crops like Wheat, Barley, Mustard.")
    else:
        print("🌱 Suggestion: Balanced crops like Maize, Pulses, Groundnut.")
