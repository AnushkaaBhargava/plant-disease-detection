#!/usr/bin/env python3
"""
Production-ready crop recommender that:
 - Loads XGBoost JSON model and LabelEncoder (pickle)
 - Predicts top-K crops for a given input (N,P,K,temperature,humidity,ph,rainfall)
 - Fetches OpenWeather OneCall (current + daily) for given lat/lon
 - Computes weather averages and monthly rainfall estimate
 - Scores each predicted crop vs crop_requirements.json (temp, humidity, monthly_rain, pH, wind)
 - Returns top-2 crops ranked by combined score

Usage examples:
  python crop_recommender_prod.py --model model_xgb.json --encoder label_encoder.pkl \
    --crop-req crop_requirements.json --api-key 31d0b53cae3d1255b65b2a43597eccd4 \
    --lat 12.97 --lon 77.59 --features-file sample.csv --feature-index 0

  OR

  python crop_recommender_prod.py --model model_xgb.json --encoder label_encoder.pkl \
    --crop-req crop_requirements.json --api-key 31d0b53cae3d1255b65b2a43597eccd4 \
    --lat 12.97 --lon 77.59 --feature-values "90,42,43,26,80,6.5,200"

  OR (Quick run with defaults - just provide your files and coordinates):
  
  python crop_recommender_prod.py --api-key YOUR_API_KEY --lat 12.97 --lon 77.59 \
    --feature-values "90,42,43,26,80,6.5,200"
"""
import argparse
import json
import logging
import os
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import pickle

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -----------------------
# Helper functions
# -----------------------
def load_xgb_model(path: str) -> xgb.XGBClassifier:
    """Load XGBoost model from JSON or pickle file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    if path.endswith('.pkl') or path.endswith('.pickle'):
        # Load from pickle file
        logging.info("Loading XGBoost model from pickle file")
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    elif path.endswith('.json'):
        # Load from JSON file
        logging.info("Loading XGBoost model from JSON file")
        model = xgb.XGBClassifier()
        model.load_model(path)
        return model
    else:
        # Try to determine format by attempting to load
        try:
            # First try pickle
            logging.info("Attempting to load as pickle file")
            with open(path, 'rb') as f:
                model = pickle.load(f)
            return model
        except:
            try:
                # Then try JSON
                logging.info("Attempting to load as JSON file")
                model = xgb.XGBClassifier()
                model.load_model(path)
                return model
            except Exception as e:
                raise RuntimeError(f"Could not load model from {path}. Tried both pickle and JSON formats. Error: {e}")


def load_label_encoder(path: str):
    """Load label encoder from pickle file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label encoder file not found: {path}")
    with open(path, "rb") as f:
        le = pickle.load(f)
    return le


def load_feature_row_from_csv(path: str, index: int = 0) -> np.ndarray:
    """Load feature row from CSV file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features CSV file not found: {path}")
    df = pd.read_csv(path)
    if index >= len(df):
        raise IndexError(f"Feature index {index} is out of range. CSV has {len(df)} rows.")
    row = df.iloc[index].values.astype(float)
    return row.reshape(1, -1)


def parse_feature_values(csv_str: str) -> np.ndarray:
    """Parse comma-separated feature values"""
    try:
        vals = [float(x.strip()) for x in csv_str.split(",")]
        if len(vals) != 7:
            raise ValueError(f"Expected 7 feature values (N,P,K,temperature,humidity,ph,rainfall), got {len(vals)}")
        return np.array(vals).reshape(1, -1)
    except ValueError as e:
        raise ValueError(f"Error parsing feature values '{csv_str}': {e}")


# -----------------------
# Weather fetch & summarise
# -----------------------
def fetch_openweather_onecall(api_key: str, lat: float, lon: float, days: int = 7) -> Dict[str, Any]:
    """Fetch weather data from OpenWeather OneCall API"""
    url = "https://api.openweathermap.org/data/2.5/onecall"
    params = {
        "lat": lat,
        "lon": lon,
        "exclude": "minutely,alerts",
        "appid": api_key,
        "units": "metric"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch weather data: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse weather data: {e}")

    temps, hums, winds, daily_precip_mm = [], [], [], []

    # current weather
    cur = data.get("current", {})
    if cur:
        if cur.get("temp") is not None:
            temps.append(cur["temp"])
        if cur.get("humidity") is not None:
            hums.append(cur["humidity"])
        if cur.get("wind_speed") is not None:
            winds.append(cur["wind_speed"])
        rain_h = cur.get("rain", {}).get("1h", 0.0) if isinstance(cur.get("rain"), dict) else 0.0
        snow_h = cur.get("snow", {}).get("1h", 0.0) if isinstance(cur.get("snow"), dict) else 0.0
        daily_precip_mm.append(rain_h * 24)

    # daily forecasts
    daily_list = data.get("daily", [])[:days]
    for d in daily_list:
        t = None
        if isinstance(d.get("temp"), dict):
            t = d["temp"].get("day")
        else:
            t = d.get("temp")
        if t is not None:
            temps.append(t)
        if d.get("humidity") is not None:
            hums.append(d.get("humidity"))
        if d.get("wind_speed") is not None:
            winds.append(d.get("wind_speed"))
        rain = d.get("rain", 0.0) or 0.0
        snow = d.get("snow", 0.0) or 0.0
        daily_precip_mm.append(rain + snow)

    avg_temp = float(np.mean(temps)) if temps else None
    avg_humidity = float(np.mean(hums)) if hums else None
    avg_wind = float(np.mean(winds)) if winds else None
    avg_daily_precip_mm = float(np.mean(daily_precip_mm)) if daily_precip_mm else 0.0

    monthly_precip_mm = avg_daily_precip_mm * 30.0
    annual_precip_mm = avg_daily_precip_mm * 365.0

    summary = {
        "avg_temp_c": avg_temp,
        "avg_humidity_pct": avg_humidity,
        "avg_wind_m_s": avg_wind,
        "avg_daily_precip_mm": avg_daily_precip_mm,
        "monthly_precip_mm_est": monthly_precip_mm,
        "annual_precip_mm_est": annual_precip_mm,
        "raw": data
    }
    return summary


# -----------------------
# Crop scoring
# -----------------------
def score_crop_requirements(req: Dict[str, Any], weather_summary: Dict[str, Any], soil_row: Dict[str, float]) -> Tuple[float, List[str]]:
    """Score how well a crop matches current weather and soil conditions"""
    reasons = []
    weights = req.get("weights", {"temp": 0.35, "rain": 0.35, "humidity": 0.15, "ph": 0.15})

    # Temp
    temp_score = 0.5
    if weather_summary.get("avg_temp_c") is None or req.get("temp_min") is None:
        reasons.append("Temp data or requirement missing -> neutral")
        temp_score = 0.6
    else:
        t = weather_summary["avg_temp_c"]
        tmin, tmax = req["temp_min"], req["temp_max"]
        if tmin <= t <= tmax:
            temp_score = 1.0
            reasons.append(f"Temp {t:.1f}°C within [{tmin},{tmax}]")
        else:
            dist = min(abs(t - tmin), abs(t - tmax))
            temp_score = max(0.0, 1.0 - (dist / 10.0))
            reasons.append(f"Temp {t:.1f}°C outside [{tmin},{tmax}], penalty applied")

    # Rainfall
    rain_score = 0.5
    monthly = weather_summary.get("monthly_precip_mm_est")
    if monthly is None or req.get("monthly_rain_min") is None:
        reasons.append("Rainfall data or requirement missing -> neutral")
        rain_score = 0.6
    else:
        rmin, rmax = req["monthly_rain_min"], req["monthly_rain_max"]
        if rmin <= monthly <= rmax:
            rain_score = 1.0
            reasons.append(f"Monthly precip ~{monthly:.0f}mm within [{rmin},{rmax}]")
        else:
            dist = min(abs(monthly - rmin), abs(monthly - rmax))
            denom = max(1.0, (rmax - rmin) if (rmax - rmin) > 0 else 100.0)
            rain_score = max(0.0, 1.0 - (dist / denom))
            reasons.append(f"Monthly precip ~{monthly:.0f}mm outside [{rmin},{rmax}], penalty applied")

    # Humidity
    hum_score = 0.6
    if weather_summary.get("avg_humidity_pct") is None or req.get("humidity_min") is None:
        reasons.append("Humidity data or requirement missing -> neutral")
        hum_score = 0.6
    else:
        h = weather_summary["avg_humidity_pct"]
        hmin, hmax = req["humidity_min"], req["humidity_max"]
        if hmin <= h <= hmax:
            hum_score = 1.0
            reasons.append(f"Humidity {h:.0f}% within [{hmin},{hmax}]")
        else:
            dist = min(abs(h - hmin), abs(h - hmax))
            hum_score = max(0.0, 1.0 - (dist / 50.0))
            reasons.append(f"Humidity {h:.0f}% outside [{hmin},{hmax}], penalty applied")

    # pH
    ph_score = 0.7
    soil_ph = soil_row.get("ph")
    if soil_ph is None or req.get("ph_min") is None:
        reasons.append("Soil pH or requirement missing -> neutral")
        ph_score = 0.6
    else:
        phmin, phmax = req["ph_min"], req["ph_max"]
        if phmin <= soil_ph <= phmax:
            ph_score = 1.0
            reasons.append(f"Soil pH {soil_ph:.2f} within [{phmin},{phmax}]")
        else:
            dist = min(abs(soil_ph - phmin), abs(soil_ph - phmax))
            ph_score = max(0.0, 1.0 - (dist / 3.0))
            reasons.append(f"Soil pH {soil_ph:.2f} outside [{phmin},{phmax}], penalty applied")

    # Weighted sum
    wtemp = weights.get("temp", 0.35)
    wpre = weights.get("rain", 0.35)
    whum = weights.get("humidity", 0.15)
    wph = weights.get("ph", 0.15)
    final = (temp_score * wtemp + rain_score * wpre + hum_score * whum + ph_score * wph) * 100.0
    final = round(final, 2)

    return final, reasons


# -----------------------
# Utilities
# -----------------------
def top_k_predictions(model, X_row: np.ndarray, label_encoder, k: int = 4) -> List[Tuple[str, float]]:
    """Get top-k crop predictions with probabilities"""
    probs = model.predict_proba(X_row)[0]
    idx = np.argsort(probs)[::-1][:k]
    top = []
    
    for i in idx:
        try:
            # Try to get class name from model classes
            if hasattr(model, 'classes_') and len(model.classes_) > i:
                if hasattr(model.classes_[i], 'item'):
                    # Handle numpy types
                    class_idx = model.classes_[i].item() if hasattr(model.classes_[i], 'item') else model.classes_[i]
                else:
                    class_idx = model.classes_[i]
                
                # Convert using label encoder
                crop_name = label_encoder.inverse_transform([class_idx])[0]
            else:
                # Fallback: use index directly
                crop_name = label_encoder.inverse_transform([i])[0]
                
        except Exception as e:
            logging.warning(f"Could not decode class {i}: {e}, using raw class")
            crop_name = f"crop_{i}"
            
        top.append((crop_name, float(probs[i])))
    
    return top


def get_default_files():
    """Get default file paths if they exist in current directory"""
    defaults = {
        'model': None,
        'encoder': None,
        'crop_req': None
    }
    
    # Common model file names
    model_names = ['model_xgb.json', 'model.json', 'xgb_model.json', 'XGBoost-final-crop.pkl', 'model.pkl', 'xgb_model.pkl']
    for name in model_names:
        if os.path.exists(name):
            defaults['model'] = name
            break
    
    # Common encoder file names  
    encoder_names = ['label_encoder.pkl', 'encoder.pkl', 'le.pkl']
    for name in encoder_names:
        if os.path.exists(name):
            defaults['encoder'] = name
            break
    
    # Common crop requirements file names
    crop_names = ['crop_requirements.json', 'crop_reqs.json', 'requirements.json']
    for name in crop_names:
        if os.path.exists(name):
            defaults['crop_req'] = name
            break
    
    return defaults


# -----------------------
# Main CLI
# -----------------------
def main():
    # Get default file paths
    defaults = get_default_files()
    
    p = argparse.ArgumentParser(
        description="Crop recommendation system using ML model and weather data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with feature values
  python %(prog)s --api-key YOUR_KEY --lat 12.97 --lon 77.59 --feature-values "90,42,43,26,80,6.5,200"
  
  # Using CSV file for features
  python %(prog)s --api-key YOUR_KEY --lat 12.97 --lon 77.59 --features-file sample.csv --feature-index 0
  
  # Full command with all parameters
  python %(prog)s --model model_xgb.json --encoder label_encoder.pkl --crop-req crop_requirements.json --api-key YOUR_KEY --lat 12.97 --lon 77.59 --feature-values "90,42,43,26,80,6.5,200"
  
Feature values format: N,P,K,temperature,humidity,ph,rainfall
Example: "90,42,43,26,80,6.5,200" means N=90, P=42, K=43, temp=26°C, humidity=80%, pH=6.5, rainfall=200mm
        """
    )
    
    # Required parameters
    p.add_argument("--api-key", required=True, 
                   help="OpenWeather API key")
    p.add_argument("--lat", type=float, required=True,
                   help="Latitude coordinate")
    p.add_argument("--lon", type=float, required=True,
                   help="Longitude coordinate")
    
    # Model files (with defaults)
    p.add_argument("--model", default=defaults['model'],
                   help=f"Path to XGBoost model file (.json or .pkl) (default: auto-detect, found: {defaults['model'] or 'None'})")
    p.add_argument("--encoder", default=defaults['encoder'],
                   help=f"Path to label encoder pickle (default: auto-detect, found: {defaults['encoder'] or 'None'})")
    p.add_argument("--crop-req", default=defaults['crop_req'],
                   help=f"Path to crop_requirements.json (default: auto-detect, found: {defaults['crop_req'] or 'None'})")
    
    # Input features (mutually exclusive)
    feature_group = p.add_mutually_exclusive_group(required=True)
    feature_group.add_argument("--features-file", 
                               help="CSV file with feature rows (N,P,K,temperature,humidity,ph,rainfall)")
    feature_group.add_argument("--feature-values", 
                               help='Comma-separated features: "N,P,K,temperature,humidity,ph,rainfall"')
    
    # Optional parameters
    p.add_argument("--feature-index", type=int, default=0,
                   help="Row index to use from features CSV file (default: 0)")
    p.add_argument("--top-k", type=int, default=4,
                   help="Number of top predictions to consider (default: 4)")
    p.add_argument("--forecast-days", type=int, default=7, 
                   help="Days to average weather over (default: 7)")
    p.add_argument("--skip-weather", action="store_true",
                   help="Skip weather fetching (for testing or if API key unavailable)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable verbose logging")
    
    args = p.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate required files
    required_files = [
        (args.model, "model"),
        (args.encoder, "encoder"), 
        (args.crop_req, "crop requirements")
    ]
    
    missing_files = []
    for file_path, file_type in required_files:
        if not file_path:
            missing_files.append(f"{file_type} file not specified and not found automatically")
        elif not os.path.exists(file_path):
            missing_files.append(f"{file_type} file not found: {file_path}")
    
    if missing_files:
        print("Error: Missing required files:")
        for msg in missing_files:
            print(f"  - {msg}")
        print("\nPlease ensure all required files are present or specify their paths.")
        return 1

    try:
        # Load model + encoder
        logging.info("Loading model from: %s", args.model)
        model = load_xgb_model(args.model)
        
        logging.info("Loading encoder from: %s", args.encoder)
        encoder = load_label_encoder(args.encoder)

        # Load crop requirements
        logging.info("Loading crop requirements from: %s", args.crop_req)
        with open(args.crop_req, "r") as f:
            crop_reqs = json.load(f)

        # Load input features
        if args.features_file:
            logging.info("Loading features from CSV: %s (row %d)", args.features_file, args.feature_index)
            X_row = load_feature_row_from_csv(args.features_file, args.feature_index)
        else:
            logging.info("Parsing feature values: %s", args.feature_values)
            X_row = parse_feature_values(args.feature_values)

        # Create soil dict
        soil_vals = X_row.flatten().tolist()
        soil_keys = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        soil_row = {k: float(v) for k, v in zip(soil_keys, soil_vals)}
        
        logging.info("Input features: %s", soil_row)

        # Predict top K
        logging.info("Predicting top-%d crops using model", args.top_k)
        topk = top_k_predictions(model, X_row, encoder, k=args.top_k)
        logging.info("Top predictions: %s", [(crop, f"{prob:.3f}") for crop, prob in topk])

        # Fetch weather summary (or use defaults if skipped)
        if args.skip_weather:
            logging.info("Skipping weather fetch - using neutral weather values")
            weather = {
                "avg_temp_c": 25.0,  # Neutral temperature
                "avg_humidity_pct": 60.0,  # Neutral humidity
                "avg_wind_m_s": 2.0,
                "avg_daily_precip_mm": 5.0,
                "monthly_precip_mm_est": 150.0,  # Neutral rainfall
                "annual_precip_mm_est": 1800.0
            }
        else:
            logging.info("Fetching weather for lat=%.3f lon=%.3f (avg of %d days)", 
                         args.lat, args.lon, args.forecast_days)
            weather = fetch_openweather_onecall(args.api_key, args.lat, args.lon, days=args.forecast_days)
            logging.info("Weather summary: temp=%.1f°C humidity=%.0f%% monthly_precip~%.0fmm",
                         weather.get("avg_temp_c") or 0.0, 
                         weather.get("avg_humidity_pct") or 0.0, 
                         weather.get("monthly_precip_mm_est") or 0.0)

        # Score each predicted crop
        scored = []
        for crop_name, prob in topk:
            req = crop_reqs.get(crop_name.lower()) or crop_reqs.get(crop_name) or {}
            score, reasons = score_crop_requirements(req, weather, soil_row)
            alpha = 0.7  # Weight for suitability score
            beta = 0.3   # Weight for model confidence
            final_rank = round(alpha * score + beta * (prob * 100.0), 2)
            scored.append({
                "crop": crop_name,
                "model_prob": round(prob, 4),
                "suitability_score": score,
                "final_rank_score": final_rank,
                "reasons": reasons,
                "requirements_used": req
            })

        # Sort by final ranking score
        scored_sorted = sorted(scored, key=lambda x: x["final_rank_score"], reverse=True)
        top2 = scored_sorted[:2]

        # Prepare output
        output = {
            "input_features": soil_row,
            "location": {"lat": args.lat, "lon": args.lon},
            "weather_summary": {k: v for k, v in weather.items() if k != "raw"},
            "top_predictions": scored_sorted,
            "recommended_crops": top2,
            "model_info": {
                "model_file": args.model,
                "encoder_file": args.encoder,
                "crop_requirements_file": args.crop_req,
                "top_k_considered": args.top_k
            }
        }

        # Print results
        print(json.dumps(output, indent=2, ensure_ascii=False))
        
        # Print summary to stderr so it doesn't interfere with JSON output
        print(f"\n=== CROP RECOMMENDATION SUMMARY ===", file=sys.stderr)
        print(f"Location: {args.lat:.3f}, {args.lon:.3f}", file=sys.stderr)
        print(f"Weather: {weather.get('avg_temp_c', 0):.1f}°C, {weather.get('avg_humidity_pct', 0):.0f}% humidity", file=sys.stderr)
        print(f"Top 2 Recommended Crops:", file=sys.stderr)
        for i, crop in enumerate(top2, 1):
            print(f"  {i}. {crop['crop']} (Score: {crop['final_rank_score']:.1f}/100)", file=sys.stderr)
        
        return 0

    except Exception as e:
        logging.error("Error: %s", str(e))
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())