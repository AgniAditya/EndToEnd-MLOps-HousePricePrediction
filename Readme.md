# 🏠 End-to-End MLOps: House Price Prediction

An open-source, production-grade workflow for building, deploying, and monitoring machine learning models to predict house prices. This project demonstrates best practices in MLOps by covering the full ML lifecycle.

---

## 🚀 What does this project do?

- **Automates** the full pipeline: data ingestion, preprocessing, model training, validation, deployment, and monitoring.
- **Implements MLOps** principles, enabling rapid iteration, reproducibility, and robust model management.
- **Predicts house prices** using machine learning on real-world datasets, serving as a reference for similar regression problems.

---

## 💡 Why is this project useful?

- **Real-World Template**: Jumpstart your own ML projects with a complete, modular MLOps setup.
- **Learning Resource**: Understand and apply industry-standard patterns for continuous integration/continuous deployment (CI/CD) in ML.
- **Scalable & Maintainable**: Designed for extensibility, so you can adapt it for your own datasets and workflows.
- **Demonstrates Automation**: Covers retraining, model versioning, and monitoring to keep your predictions up-to-date and reliable.

---

## 🛠️ Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/AgniAditya/EndToEnd-MLOps-HousePricePrediction.git
cd EndToEnd-MLOps-HousePricePrediction
```

**2. Set up your environment**
- Make sure you have Python and Jupyter Notebook installed.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

**3. Explore the workflow**
- Open the Jupyter notebooks for hands-on walkthroughs of data processing, model building, and pipeline automation.
- Run individual scripts or the full pipeline as described in the notebooks.

**4. Customize for your needs**
- Use your own dataset by replacing the data source.
- Tweak the model, pipeline stages, or deployment steps as required.

---

## 🐳 Run with Docker

You can easily deploy and use the House Price Prediction model using Docker, without setting up Python or installing any dependencies.

**1. Pull the Docker image:**
```bash
docker pull agniaditya/house-price-fastapi
```

**2. Run the Docker container:**
```bash
docker run -d -p 8000:8000 agniaditya/house-price-fastapi
```
This will start the FastAPI server for the model and bind it to port `8000` on your machine.

**3. Access the API:**

- Visit [http://localhost:8000/docs](http://localhost:8000/docs) in your browser for the interactive FastAPI documentation.
- You can make predictions by sending POST requests to the `/predict` endpoint as described in the docs.

**Docker Hub:**  
Find the published Docker image and more usage instructions at:  
[https://hub.docker.com/r/agniaditya/house-price-fastapi](https://hub.docker.com/r/agniaditya/house-price-fastapi)

---

## 📚 Where to Get Help

- **Issues & Bugs**: [GitHub Issues](https://github.com/AgniAditya/EndToEnd-MLOps-HousePricePrediction/issues)
- **General Questions**: Reach out via [GitHub profile](https://github.com/AgniAditya)
- **Discussions**: (If enabled) Use the Discussions tab for open-ended questions and collaboration.

---

## 👥 Maintainers & Contributors

- **Maintainer:** [AgniAditya](https://github.com/AgniAditya)
- **Contributions:** Welcomed! Fork this repo, create a feature branch, and submit a pull request.  
  See `CONTRIBUTING.md` (if available) for more details.

---

## 🤝 How Can You Contribute?

- **Report bugs** or request features using GitHub Issues.
- **Submit pull requests** to add new features, improve documentation, or fix bugs.
- **Share feedback** to help evolve this project!

---

## 📄 License

This project is open-source. Please check the repository for license details or contact the maintainer for more information.

---

## Model Files

The trained model files are not included in this repository due to their size. To run the API:

1. Train the model using the training pipeline:
   ```bash
   python run_pipeline.py
   ```

2. This will create the required model files in the `models/` directory:
   - `models/model.pkl`: The trained model
   - `models/scaler.pkl`: The feature scaler

3. Then run the API using Uvicorn:
   ```bash
   uvicorn app:app --reload
   ```

---

> _Empower your ML projects with robust MLOps—start predicting and deploying with confidence!_