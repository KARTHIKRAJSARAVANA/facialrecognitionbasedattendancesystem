from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import csv
import json
import bcrypt

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend-backend communication

USER_DATA_FILE = 'users.json'
ATTENDANCE_FOLDER = "Attendance"

# Ensure the user data file exists
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, 'w') as file:
        json.dump({}, file)

# Ensure the attendance folder exists
if not os.path.exists(ATTENDANCE_FOLDER):
    os.makedirs(ATTENDANCE_FOLDER)


def read_user_data():
    with open(USER_DATA_FILE, 'r') as file:
        return json.load(file)


def write_user_data(data):
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(data, file, indent=4)






@app.route('/attendance/<date>', methods=['GET'])
def get_attendance_by_date(date):
    """
    Fetch attendance data for a specific date.
    :param date: Date in the format DD-MM-YYYY.
    """
    file_path = os.path.join(ATTENDANCE_FOLDER, f"Attendance_{date}.csv")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                reader = csv.DictReader(file)
                attendance_data = list(reader)
            return jsonify({"success": True, "data": attendance_data}), 200
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500
    else:
        return jsonify({"success": False, "message": "Attendance file not found for the given date."}), 404


@app.route('/attendance/all', methods=['GET'])
def get_all_attendance():
    """
    Fetch attendance data from all files.
    """
    attendance_records = []
    try:
        for filename in os.listdir(ATTENDANCE_FOLDER):
            if filename.endswith(".csv"):
                with open(os.path.join(ATTENDANCE_FOLDER, filename), 'r') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        row['file'] = filename  # Add filename for reference
                        attendance_records.append(row)
        return jsonify({"success": True, "data": attendance_records}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/create-account', methods=['POST'])
def create_account():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not (name and email and password):
        return jsonify({"success": False, "message": "All fields are required!"}), 400

    users = read_user_data()

    if email in users:
        return jsonify({"success": False, "message": "Email already registered!"}), 400

    # Hash the password before saving
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[email] = {"name": name, "password": hashed_password}
    write_user_data(users)

    return jsonify({"success": True, "message": "Account created successfully!"}), 201


@app.route('/register-face', methods=['POST'])
def register_face():
    data = request.get_json()
    name = data.get('name')

    if not name:
        return jsonify({"success": False, "message": "Name is required for face registration!"}), 400

    try:
        subprocess.Popen(["python", "add_faces.py"], stdin=subprocess.PIPE).communicate(input=name.encode())
        return jsonify({"success": True, "message": "Face registration started!"}), 200
    except FileNotFoundError:
        return jsonify({"success": False, "message": "Script 'add_faces.py' not found!"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/signin', methods=['POST'])
def signin():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    users = read_user_data()

    if email in users and bcrypt.checkpw(password.encode(), users[email]["password"].encode()):
        return jsonify({"success": True, "message": "Sign-in successful!"}), 200
    return jsonify({"success": False, "message": "Invalid credentials!"}), 401


@app.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    try:
        result = subprocess.run(['python', 'test.py'], capture_output=True, text=True)
        if result.returncode == 0:
            return jsonify({"success": True, "message": "Attendance marked successfully."}), 200
        return jsonify({"success": False, "message": "Failed to mark attendance. Check logs."}), 500
    except FileNotFoundError:
        return jsonify({"success": False, "message": "Script 'test.py' not found!"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500


if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000)
