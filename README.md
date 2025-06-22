# Product Recommendation System

This project delivers intelligent product recommendations using machine learning algorithms, all deployable in a containerized (Docker) environment for easy scalability and integration.

## Key Features
- **ML-Powered Recommendations:**
  - Uses user interaction data and product information to generate personalized recommendations.
  - Machine learning models are trained and served within a Docker container for portability and reliability.

- **LLM-Driven Product Descriptions:**
  - A separate FastAPI application (to be developed) will connect with an LLM agent.
  - This service will provide users with natural language descriptions and summaries of recommended products.

## Structure
- `app.py`: Main application for serving recommendations.
- `training/`: Scripts for model training.
- `recommendation/`: Prediction and recommendation logic.
- `datasets/`: Data files for products, users, and interactions.
- `Dockerfile`: Container setup for deployment.

