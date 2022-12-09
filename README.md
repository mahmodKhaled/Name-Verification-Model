# Name-Verification-Model
## Description
The task is divided into multiple stages (Data generation - Core model - Interface - containerization).
The core stage of the task pipeline is name verification model with confidence threshold of how much the
name is real.

For example:
1. The name "باسم وحيد السيد" is a real name with high confidence.
2. The name "باسمم وحةد السد" “is a real name with low confidence.

## Instructions

### How to build the docker image
To build the Docker Image we should execute the following docker-compose command
```
docker-compose -f docker-compose.yaml up
```
After completion of building the last part in the output in the terminal should be as following 
```
task                              | INFO:     Started server process [1]
task                              | INFO:     Waiting for application startup.
task                              | INFO:     Application startup complete.
task                              | INFO:     Uvicorn running on http://0.0.0.0:80 (Press CTRL+C to quit)
```
**So, in this case the app has been built successfully!**

