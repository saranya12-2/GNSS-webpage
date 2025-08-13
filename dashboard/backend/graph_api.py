from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import json
import glob
import logging
from datetime import datetime, timedelta

graph_api = Flask(__name__)
CORS(graph_api)

# Configure logging
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.join(os.getcwd(), "output")


def get_predictions_for_day(date_str, model_id):
    """
    Fetch all predicted + original TEC data for a given date and model.
    """
    try:
        model_dir = os.path.join(BASE_DIR, model_id)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model folder not found for ID {model_id}")

        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        date_fmt = date_obj.strftime("%d%m%Y")  # like "02042024"

        all_rows = []

        # Match files like 020420240.json (original), 020420241.json ... 020420246.json (predicted)
        pattern = os.path.join(model_dir, f"{date_fmt}[0-6].json")
        matching_files = glob.glob(pattern)

        for file_path in matching_files:
            suffix = os.path.splitext(file_path)[0][-1]  # last character before .json
            source = "original" if suffix == "0" else "predicted"

            with open(file_path) as f:
                day_data = json.load(f)
                for row in day_data:
                    full_time = f"{date_obj.strftime('%Y-%m-%d')} {row['Time']}"
                    all_rows.append({
                        "time": full_time,
                        "tec": row["TEC"],
                        "source": source
                    })

        # Sort by full time
        all_rows.sort(key=lambda x: datetime.strptime(x["time"], "%Y-%m-%d %H:%M:%S"))
        return all_rows

    except Exception as e:
        logging.exception("Error in get_predictions_for_day")
        raise e


@graph_api.route("/api/model-predict-date")
def predict_single_date():
    model = request.args.get("model")
    date = request.args.get("date")

    if not model or not date:
        return jsonify({"error": "Missing model or date parameter"}), 400

    try:
        data = get_predictions_for_day(date, model)
        if not data:
            return jsonify({"error": "No data found for the selected date"}), 404
        return jsonify({"data": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@graph_api.route("/api/model-predict-range")
def predict_range():
    model_id = request.args.get("model")
    start_date = request.args.get("start")
    end_date = request.args.get("end")

    if not all([model_id, start_date, end_date]):
        return jsonify({"error": "Missing model/start/end parameter"}), 400

    model_dir = os.path.join(BASE_DIR, model_id)
    if not os.path.exists(model_dir):
        return jsonify({"error": "Model folder not found"}), 404

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        delta = (end - start).days + 1

        all_rows = []

        for i in range(delta):
            date_obj = start + timedelta(days=i)
            date_str = date_obj.strftime("%d%m%Y")  # e.g., 02042024

            # Match files like 020420240.json, 020420241.json, ..., 020420246.json
            pattern = os.path.join(model_dir, f"{date_str}[0-6].json")
            matching_files = glob.glob(pattern)

            for file_path in matching_files:
                suffix = os.path.splitext(file_path)[0][-1]
                source = "original" if suffix == "0" else "predicted"

                with open(file_path) as f:
                    day_data = json.load(f)
                    for row in day_data:
                        full_time = f"{date_obj.strftime('%Y-%m-%d')} {row['Time']}"
                        all_rows.append({
                            "time": full_time,
                            "tec": row["TEC"],
                            "source": source
                        })

        if not all_rows:
            return jsonify({"error": "No data found in the date range"}), 404

        all_rows.sort(key=lambda x: datetime.strptime(x['time'], "%Y-%m-%d %H:%M:%S"))

        return jsonify({
            "model": model_id,
            "start_date": start_date,
            "end_date": end_date,
            "data": all_rows
        })

    except Exception as e:
        logging.exception("Exception occurred during prediction range fetch.")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    graph_api.run(host='0.0.0.0', port=5000, debug=True)
