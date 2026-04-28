# 🚀 NOX Project – FastAPI Deployment using Docker & AWS EC2

## 🌟 Overview

This project demonstrates how to take a simple FastAPI application from local development to a fully deployed cloud service using Docker and AWS EC2.

Instead of just running code locally, the goal was to simulate a real-world production setup — containerizing the application and deploying it on a remote server.

---

## 🧠 The Journey (Story Style)

I started with a basic FastAPI application running locally.
To make it production-ready, I containerized the app using Docker, ensuring it could run consistently across environments.

Next, I pushed the entire project to GitHub to maintain version control and enable remote deployment.

Finally, I launched an AWS EC2 instance, configured security rules, installed Docker, and deployed the application — making it accessible over the internet.

This process helped me understand how real-world backend systems are deployed and managed.

---

## 🛠️ Tech Stack

* Python (FastAPI)
* Docker
* AWS EC2
* Git & GitHub

---

## ⚙️ Steps Performed

### 1. Local Development

* Built FastAPI app
* Tested APIs via Swagger (`/docs`)

### 2. Dockerization

* Created Dockerfile
* Built Docker image
* Ran container locally

### 3. Version Control

* Initialized Git
* Added `.gitignore`
* Pushed code to GitHub

### 4. AWS Deployment

* Created EC2 instance (t2.micro)
* Configured security group (port 8000)
* Installed Docker on EC2
* Cloned project from GitHub
* Built and ran Docker container

---

## 🌍 Live Deployment

Application is deployed on AWS EC2 and accessible via:

http://<EC2-PUBLIC-IP>:8000/docs

---

## 📌 Key Learnings

* Difference between local vs cloud deployment
* Importance of Docker for consistency
* AWS EC2 setup and security configuration
* End-to-end deployment workflow

---

## 🚀 Future Improvements

* Add CI/CD pipeline
* Use Nginx + domain
* Deploy using Kubernetes

---
