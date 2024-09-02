# Deploying ML Solutions with FastAPI, Docker, and AWS

Lets walk through the steps of deploying a machine learning (ML) solution using FastAPI, Docker, and AWS Elastic Container Service (ECS). We will cover creating an API using FastAPI, containerizing the API with Docker, pushing the Docker image to Docker Hub, and finally deploying the container on AWS.
![image](https://github.com/user-attachments/assets/7e022cca-b292-451b-bb1b-69f2e087f2d9)


![image](https://github.com/user-attachments/assets/568d8a0f-c5b0-402d-81dc-40d9b2d54609)


![image](https://github.com/user-attachments/assets/77e59722-51d7-4aa2-92ae-da79e87cf249)

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
![image](https://github.com/user-attachments/assets/9ee3ec9b-a3ae-4487-b34b-aa123ed0bdd2)

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

```
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```
![image](https://github.com/user-attachments/assets/4d6201a7-629b-4acc-8695-7cf7e14d9fce)

### 2.2 Building and Running the Docker Image

Build the Docker image:
```
bash
Copy code
docker build -t yt-search-api .
```
![image](https://github.com/user-attachments/assets/39e5548c-7c92-4d64-b999-5afc4611e300)
![image](https://github.com/user-attachments/assets/2d1ad158-2f93-4b15-a84b-15433b78ed7a)
![image](https://github.com/user-attachments/assets/70ceb0e0-1a82-4bbb-ae49-68716559cfb5)

Run the container:

```bash
Copy code
docker run -d --name yt-search-container -p 8080:80 yt-search-api
Visit http://localhost:8080/docs to access the API running in Docker.
```

![image](https://github.com/user-attachments/assets/036bd34c-d989-4ea4-97bd-5a44c3c0d0dc)

![image](https://github.com/user-attachments/assets/58a16041-4c50-4eac-aa17-67560cb6be59)


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
![image](https://github.com/user-attachments/assets/6ab48034-ee50-4773-b1fd-86c6c2e3c752)

![image](https://github.com/user-attachments/assets/e1005886-886f-43da-a0de-2a1be61e30bd)

## 4. Deploying to AWS Elastic Container Service (ECS)
### 4.1 Setting Up AWS ECS
Go to the AWS Management Console and open ECS. Create a new task definition, and choose the Fargate launch type.

### 4.2 Configuring the Task Definition
Select the container name and paste the Docker image URL from Docker Hub.
Set the CPU and memory requirements based on your needs.
![image](https://github.com/user-attachments/assets/cce6b5b0-8b63-4f02-bfdc-9928f4cc10fa)

![image](https://github.com/user-attachments/assets/7a1de82a-d64c-4b77-8f71-b1b9b16c8493)

![image](https://github.com/user-attachments/assets/6485cbd4-f673-48ed-8f88-69f1b8ec2b99)

![image](https://github.com/user-attachments/assets/2519cbb7-c226-4720-b9c4-7d63070f661c)


![image](https://github.com/user-attachments/assets/ff3412bb-ea35-43f0-bff1-dae5ce0f630f)

![image](https://github.com/user-attachments/assets/862be388-bfb3-49b8-8650-b9cd68618d76)

![image](https://github.com/user-attachments/assets/5ec1360e-0cdf-4f45-9b32-f1446f9eb639)

![image](https://github.com/user-attachments/assets/d4bae1bb-2fd4-48bd-bd83-41c6cb6af2c1)

### 4.3 Creating and Running the Service
Create a new cluster and service to run the container. Once deployed, you can access your API through the public IP provided by AWS ECS.
Then go to tasks to grab Public IP- 
![image](https://github.com/user-attachments/assets/03e69f69-99e5-4cd9-888a-8864d3657b4e)


![image](https://github.com/user-attachments/assets/db0e3813-5e4b-44b4-a5a7-279c9cf6cfdd)

## 5 Gradio UI Setup
Then for all traffic  network permission , go to Tasks-> network and  configuration security groups-> inbound rule
![image](https://github.com/user-attachments/assets/4eff6ac1-8ec7-4b9e-9291-03e45b85b725)

![image](https://github.com/user-attachments/assets/0975abdd-665f-4278-a3d8-3fdfb942da0c)

![image](https://github.com/user-attachments/assets/23e9e3c4-34e5-4f4d-9409-f06a73ed6267)


![image](https://github.com/user-attachments/assets/aeacea9a-91cd-4e8e-a31f-aa91d83f10ad)

