# ☀️ Automated Solar Panel Quality Assessment (MLOps Project)

This project automates the process of detecting **dust and damage on solar panels** using deep learning and an **end-to-end MLOps pipeline**.  
It integrates **DVC**, **GitHub Actions**, **Docker**, and **AWS EC2** to ensure **reproducibility, automation, and continuous deployment**.

---

## Project Overview
Solar panels often accumulate dust or suffer physical/electrical damage, reducing their efficiency.  
This project builds an automated image-based system that classifies solar panels as **Clean**, **Dusty**, or **Damaged**, and deploys the model as a live Streamlit web app.

---

## Key Features
- **Convolutional Autoencoder** trained to detect anomalies (dirty or damaged panels)
- **Data Version Control (DVC)** for tracking datasets, models, and experiments
- **Continuous Integration & Deployment (CI/CD)** with GitHub Actions
- **Containerized deployment** using Docker
- **Hosted on AWS EC2**, accessible via Streamlit UI
- **Automated testing** with Pytest

---

## 🗂️ Project Structure
.
├── data/ # Dataset (versioned via DVC)
├── models/ # Trained model artifacts
├── src/ # Training and preprocessing scripts
├── app.py # Streamlit web application
├── dvc.yaml # DVC pipeline definition
├── Dockerfile # Container setup
├── requirements.txt # Dependencies
├── .github/workflows/ # CI/CD pipeline
└── README.md

---

## MLOps Workflow
1. **Data Versioning:**  
   - DVC tracks datasets and model files.  
   - `.dvc` and `.dvc.lock` files are committed to Git for reproducibility.  

2. **Model Training:**  
   - Autoencoder model trained on clean panel images.  
   - Reconstruction error (“squiggliness score”) used for classification.

3. **CI/CD Pipeline:**  
   - Triggered automatically on every push to the `main` branch.  
   - Runs DVC pipeline, tests with Pytest, builds Docker image, and deploys to EC2.

4. **Deployment:**  
   - Docker container runs Streamlit app on port `8501`.  
   - Users upload solar panel images to get predictions in real-time.

---

##  Dockerfile Summary
- Uses lightweight base image `python:3.10-slim-bookworm`
- Installs only necessary system packages and Python dependencies
- Packages app, model, and dependencies for consistent deployment
- Exposes Streamlit app at port `8501`

```bash
docker build -t mlops-proj .
docker run -p 8501:8501 mlops-proj
CI/CD Pipeline (GitHub Actions)
Workflow: .github/workflows/ci-cd.yml

Installs dependencies and runs dvc repro

Executes unit tests with pytest

Builds Docker image

Deploys files to AWS EC2 using SCP

SSHs into EC2 to rebuild and restart the containerized app

Deployment on AWS EC2
EC2 instance hosts the Dockerized Streamlit app

Connected to AWS S3 for DVC remote storage

Accessible publicly at:
http://<ec2-public-ip>:8501


Tech Stack
Languages: Python, Bash
Libraries: TensorFlow / Keras, OpenCV, NumPy, Streamlit
MLOps Tools: DVC, GitHub Actions, Docker, AWS EC2, S3
Testing: Pytest

 Results
The model successfully classifies solar panels as Clean, Dusty, or Damaged.

Continuous retraining and deployment ensure up-to-date performance.

Fully automated pipeline reduces manual intervention.

