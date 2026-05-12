# Customer Segmentation Project

## Overview

This project focuses on customer segmentation using Machine Learning techniques. The main objective is to analyze customer behavior and divide customers into different groups based on their purchasing patterns, annual income, and spending habits.

Customer segmentation helps businesses understand their customers better and create targeted marketing strategies. In this project, K-Means Clustering is used to identify different customer groups.

The project also includes an interactive dashboard built using Streamlit for visualizing customer segments and business insights.

---

# Features

- Customer segmentation using K-Means Clustering
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Interactive Streamlit dashboard
- Customer cluster visualization
- Business insights generation
- Download clustered customer data

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Analysis |
| Matplotlib | Data Visualization |
| Scikit-learn | Machine Learning |
| Streamlit | Dashboard Development |

---

# Dataset

The project uses the Mall Customers dataset.

Dataset contains:

- Customer ID
- Gender
- Age
- Annual Income
- Spending Score

---

# Project Structure

```text
Customer-Segmentation-Project/
│
├── app.py
├── segmentation.py
├── Mall_Customers.csv
├── Customer_Segments_Output.csv
└── README.md
```

---

# Installation

## Step 1: Clone Repository

```bash
git clone <your-github-repository-link>
```

## Step 2: Navigate to Project Folder

```bash
cd Customer-Segmentation-Project
```

## Step 3: Install Required Libraries

```bash
pip install pandas matplotlib scikit-learn streamlit
```

---

# Running the Project

## Run Python Segmentation File

```bash
python segmentation.py
```

## Run Streamlit Dashboard

```bash
python -m streamlit run app.py
```

After running the command, open the Local URL and Network URL shown in the terminal.

Example:

```text
Local URL: http://localhost:8501
  Network URL: http://10.58.38.159:8501
```

---

# Machine Learning Algorithm

## K-Means Clustering

K-Means Clustering is an unsupervised machine learning algorithm used to group similar customers together.

The algorithm works by:

1. Selecting the number of clusters
2. Assigning customers to nearest clusters
3. Updating cluster centers
4. Repeating until optimal clusters are formed

---

# Visualizations Included

- Income Distribution
- Spending Score Distribution
- Customer Segmentation Scatter Plot
- Cluster Distribution Chart
- Cluster Summary Table

---

# Business Insights

- Premium customers can be targeted with luxury products.
- Customers with low spending can be encouraged using offers and discounts.
- Businesses can improve marketing strategies using customer segmentation.
- Personalized recommendations can improve customer satisfaction.

---

# Learning Outcomes

This project helps in understanding:

- Customer Analytics
- Data Visualization
- Machine Learning Basics
- Clustering Techniques
- Dashboard Development
- Business Intelligence

---

# Future Improvements

- Add more customer behavior features
- Use advanced clustering algorithms
- Deploy the project online
- Add real-time analytics
- Improve dashboard UI/UX

---

# Author

Developed as part of a Machine Learning and Data Analytics project.

---

# License

This project is for educational and learning purposes.
