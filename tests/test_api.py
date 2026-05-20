# tests/test_api.py
"""
API integration tests — covers all routes and error paths.
Uses TestClient fixtures from conftest.py. No running server needed.
"""

VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20,
}


class TestHealthEndpoints:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_health_returns_200_when_model_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

    def test_health_returns_503_when_model_not_loaded(self, client_unloaded):
        response = client_unloaded.get("/health")
        assert response.status_code == 503


class TestPredictEndpoint:
    def test_valid_input_returns_200(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_response_contains_required_keys(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "churn_prediction" in body
        assert "churn_probability" in body

    def test_churn_probability_in_valid_range(self, client):
        prob = client.post("/predict", json=VALID_PAYLOAD).json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_invalid_contract_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "Contract": "Weekly"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_field_returns_error_response_shape(self, client):
        payload = {**VALID_PAYLOAD, "Contract": "Weekly"}
        body = client.post("/predict", json=payload).json()
        assert "error" in body
        assert "detail" in body

    def test_missing_required_field_returns_422(self, client):
        response = client.post("/predict", json={"gender": "Female"})
        assert response.status_code == 422

    def test_missing_field_returns_error_response_shape(self, client):
        body = client.post("/predict", json={"gender": "Female"}).json()
        assert "error" in body
        assert "detail" in body

    def test_negative_tenure_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "tenure": -1}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422