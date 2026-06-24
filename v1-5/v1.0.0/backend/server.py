from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from frontend


@app.route('/api/run-script', methods=['POST'])
def run_script():
    data = request.json  # Get data from frontend

    # Here you would add your actual Python processing logic
    # For now, we'll just echo the data back

    result = {
        "message": "Python script executed successfully!",
        "input": data
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
