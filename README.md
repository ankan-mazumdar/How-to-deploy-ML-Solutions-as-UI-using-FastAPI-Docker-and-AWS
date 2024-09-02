# Deploying ML Solutions with FastAPI, Docker, and AWS

Lets walk through the steps of deploying a machine learning (ML) solution using FastAPI, Docker, and AWS Elastic Container Service (ECS). We will cover creating an API using FastAPI, containerizing the API with Docker, pushing the Docker image to Docker Hub, and finally deploying the container on AWS.

## Prerequisites
Before you start, ensure you have the following installed:

Python 3.10+
Docker

AWS CLI configured with your credentials

Docker Hub account

### 1. Creating the FastAPI Search API
1.1 Setting Up the Project
First, create a directory for your project and initialize a Python environment:

```bash

mkdir fastapi-ml-deploy
cd fastapi-ml-deploy
python3 -m venv venv
source venv/bin/activate
```

## `main.py`

The `main.py` file is the core of your FastAPI application, which serves as the interface for your machine learning model to interact with the outside world through API endpoints.

### Key Components:
- **Imports**:
  - Utilizes libraries like `FastAPI` for building the API, `polars` for data processing, and `SentenceTransformer` for sentence embeddings.
  - Also imports the `returnSearchResultIndexes` helper function from `functions.py`.

- **Model and Data Setup**:
  - Loads a pre-trained Sentence Transformer model (`all-MiniLM-L6-v2`) from a local path.
  - Loads the video index as a Polars DataFrame from a Parquet file (`video-index.parquet`).

- **API Endpoints**:
  - **Health Check** (`GET /`): Returns a status message to confirm the API is running.
  - **Info** (`GET /info`): Provides basic information about the API, including its name and description.
  - **Search** (`GET /search?query=...`): Accepts a search query as input, runs the query through the model to find relevant YouTube videos, and returns the top search results as a dictionary.

## `functions.py`

The `functions.py` file contains the logic for processing the search queries and retrieving relevant results.

### Key Components:
- **Imports**:
  - Uses `numpy` for numerical operations, `polars` for data manipulation, `SentenceTransformer` for embedding the query, and `pairwise_distances` from `sklearn` to calculate distances between the query and video data.

- **`returnSearchResultIndexes` Function**:
  - **Purpose**: Returns the indexes of the top search results based on the user's query.
  - **Steps**:
    - **Embedding the Query**: Converts the query string into a numerical embedding using the `SentenceTransformer`.
    - **Calculating Distances**: Computes the Manhattan distance between the query embedding and the embeddings of the video titles/transcripts.
    - **Filtering Results**: Selects videos within a certain distance (`threshold`) and sorts these by proximity, then selects the top `k` closest matches.
    - **Returning Results**: Outputs the indexes of the top matching search results.



### 1.3 Running the API Locally
To test the API locally, run:
```
bash
Copy code
uvicorn app.main:app --reload```

Visit http://127.0.0.1:8000/seacrh &  http://127.0.0.1:8000/info to interact with the API in the test notebook.

## 2. Containerizing the API with Docker
### 2.1 Creating a Dockerfile
In the root of your project, create a Dockerfile:

Dockerfile
Copy code
FROM python:3.10-slim

WORKDIR /code

COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /code/app

```CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

### 2.2 Building and Running the Docker Image

Build the Docker image:
```
bash
Copy code
docker build -t yt-search-api .
```
Run the container:

```bash
Copy code
docker run -d --name yt-search-container -p 8080:80 yt-search-api
Visit http://localhost:8080/docs to access the API running in Docker.
```

## 3. Pushing the Docker Image to Docker Hub
### 3.1 Tagging the Docker Image

Tag the image to match your Docker Hub repository:

```bash
Copy code
docker tag yt-search-api your-dockerhub-username/yt-search-api:latest
```

### 3.2 Pushing the Image
Push the image to Docker Hub:

```bash
Copy code
docker push your-dockerhub-username/yt-search-api:latest
```

## 4. Deploying to AWS Elastic Container Service (ECS)
### 4.1 Setting Up AWS ECS
Go to the AWS Management Console and open ECS. Create a new task definition, and choose the Fargate launch type.

### 4.2 Configuring the Task Definition
Select the container name and paste the Docker image URL from Docker Hub.
Set the CPU and memory requirements based on your needs.

### 4.3 Creating and Running the Service
Create a new cluster and service to run the container. Once deployed, you can access your API through the public IP provided by AWS ECS.







