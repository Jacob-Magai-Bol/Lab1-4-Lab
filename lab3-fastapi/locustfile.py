from locust import HttpUser, task, between

class IrisAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        self.client.post(
            "/predict",
            json={"features": [5.1, 3.5, 1.4, 0.2]},
            name="/predict"
        )
