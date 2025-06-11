# 🏠 End-to-End MLOps: House Price Prediction

Welcome! This project is a **comprehensive, production-grade workflow** for building, deploying, and monitoring machine learning models to predict house prices. It is designed as a reference implementation of **best practices in MLOps**, covering the full ML lifecycle from data ingestion to deployment and monitoring.

---

## ✨ Project Highlights

- **Full Automation**: End-to-end pipeline for data ingestion, preprocessing, model training, validation, deployment, and monitoring.
- **Modern MLOps**: Uses tools like ZenML and MLflow for pipeline orchestration, experiment tracking, model versioning, and CI/CD automation.
- **Real-World Use Case**: Predicts house prices using real datasets—ideal as a template for regression problems.
- **Production Ready**: Designed with scalability, maintainability, and extensibility in mind.

---

## 📊 What’s Inside?

- **Modular Workflow**: Swap in your own dataset, model, or deployment target with ease.
- **Dockerized API**: Deploy the trained model as a FastAPI app in seconds—no Python setup required.
- **Experiment Tracking**: Track every model run and metric with MLflow.
- **Continuous Integration**: Automation for retraining, versioning, and monitoring to keep predictions reliable.

---

## 🖼️ Architecture Overview

```
[Data Source] --> [Ingestion] --> [Preprocessing] --> [Model Training] --> [Evaluation] --> [Deployment] --> [Monitoring]
                                              |                                              |
                                        [MLflow, ZenML]                            [FastAPI, Docker, SQL DB]
```

- **ZenML**: Workflow automation and pipeline orchestration
- **MLflow**: Experiment tracking and model registry
- **FastAPI**: Model serving via REST API
- **Docker**: Containerized deployment
- **SQL Database**: Data storage and integration

---

## 🚀 Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/AgniAditya/EndToEnd-MLOps-HousePricePrediction.git
cd EndToEnd-MLOps-HousePricePrediction
```

### 2. Install dependencies

- Ensure Python and Jupyter Notebook are installed.
- Install required packages:
  ```bash
  pip install -r requirements.txt
  ```

### 3. Run the ML pipeline

```bash
python run_pipeline.py
```
- This will process data, train the model, and save outputs in the `models/` directory.

### 4. Launch the API (with Uvicorn)

```bash
uvicorn app:app --reload
```
- Access the interactive API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 🐳 Docker Deployment

No Python? No problem! Deploy the model API in seconds:

```bash
docker pull agniaditya/house-price-fastapi
docker run -d -p 8000:8000 agniaditya/house-price-fastapi
```
- Visit [http://localhost:8000/docs](http://localhost:8000/docs) for API documentation.
- Full Docker Hub details: [Docker Hub](https://hub.docker.com/r/agniaditya/house-price-fastapi)

---

## 🗂️ Project Structure

```
.
├── data/                  # Raw and processed data
├── models/                # Trained model and scaler (created after training)
├── notebooks/             # Jupyter notebooks for exploration and demos
├── app.py                 # FastAPI app for inference
├── run_pipeline.py        # Main pipeline runner
├── requirements.txt
├── Dockerfile
└── ...
```

---

## 🏷️ Technology Stack

- **Python** (pandas, numpy, scikit-learn, matplotlib)
- **ZenML** — Pipeline orchestration
- **MLflow** — Experiment tracking & model registry
- **FastAPI** — REST API for predictions
- **Docker** — Containerized deployment
- **SQL database** — Data storage (optional)

---

## 🧩 MLOps Features

- **Pipeline Orchestration**: Automated end-to-end workflows with ZenML
- **Experiment Tracking**: Log metrics, parameters, and artifacts with MLflow
- **Data & Model Versioning**: Full lineage tracking for reproducibility
- **Monitoring & Alerts**: Automated checks for model drift and data quality

---

## 📦 GitHub Releases

- **Latest Release: [v1.0](https://github.com/AgniAditya/EndToEnd-MLOps-HousePricePrediction/releases/tag/v1.0)**
  - **Date:** 2025-06-04
  - **Description:** A house price prediction model
  - **Assets:** [models.zip](https://github.com/AgniAditya/EndToEnd-MLOps-HousePricePrediction/releases/download/v1.0/models.zip) (Trained model files)

Download and extract `models.zip` from the [Releases page](https://github.com/AgniAditya/EndToEnd-MLOps-HousePricePrediction/releases) to use the latest trained model without retraining.

---

## 🔍 How to Contribute

1. **Fork** this repository
2. **Create a branch:**  
   `git checkout -b feature/my-new-feature`
3. **Commit your changes:**  
   `git commit -am 'Add some feature'`
4. **Push to the branch:**  
   `git push origin feature/my-new-feature`
5. **Open a Pull Request**

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 🤝 Get Support

- **Issues & Bugs:** [GitHub Issues](https://github.com/AgniAditya/EndToEnd-MLOps-HousePricePrediction/issues)
- **General Questions:** [GitHub Profile](https://github.com/AgniAditya)
- **Discussions:** Use the Discussions tab if enabled

---

## 👤 Maintainer

- [AgniAditya](https://github.com/AgniAditya)

---

## 📄 License

This project is open-source—see the repository for license details.

---

## 💬 Feedback

> _Empower your ML projects with robust MLOps—start predicting and deploying with confidence!_

---