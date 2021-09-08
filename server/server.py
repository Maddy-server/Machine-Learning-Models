from flask import Flask, request, jsonify
import util

app = Flask(__name__)


@app.route('/get_locations_names', methods=['GET'])
def get_locations_names():
    response = jsonify({
        'locations': util.get_locations_names()
    })

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.route('/predict_home_price', methods=['GET', 'POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    response = jsonify({
        'estimated_price': util.get_estimated_price(location, total_sqft, bhk, bath)
    })

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    print('Starting python flask server for home price predic')
    util.load_saved_artifacts()
    app.run()
