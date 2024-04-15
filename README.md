# ExploringGraphRecommendationSystems
COMP4740_FinalProject
Members = George Kaceli and Noah Adams
In this project we will be exploring using graphs from item recommendation with an application to movie 
recommendation using the MovieLens100K dataset, we will utilise Graph Neural Networks, such as Graph Convolution
Networks GCNs, Graph sample and aggregate GraphSAGE, and Graph Attention Networks GATs

## Project Structure

### Directories
- **`models/`**: Contains definitions for various graph-based neural network models.
- **`datasets/`**: Scripts and utilities for managing and preprocessing the dataset used in the models.
- **`visualization/`**: Tools and scripts for visualizing data insights and model performance.
- **`metrics/`**: Utilities for calculating and logging performance metrics like precision, recall, and RMSE.

### Key Files
- `convert.py`: Converts data into the required format for model processing.
- `data.py`: Manages data loading and preprocessing workflows.
- `gat_train_test.py`: Implements the training and testing phases for the Graph Attention Network model.
- `gcn_train_test.py`: Manages the training and testing processes for the Graph Convolutional Network.
- `graph.py`: Provides functionalities for graph data manipulation and visualization.
- `log_to_csv.py`: Utility to log various performance metrics to a CSV file.
- `SAGE_train_test.py`: Script for training and testing the GraphSAGE model.

## Installation and Python Environment Setup

To get started with this project, follow these steps to set up your environment:

for Linux or macOS: 

# Create a virtual environment named 'venv'
`python3 -m venv venv`

# Activate the virtual environment
`source venv/bin/activate`

for Windows: 

# Create a virtual environment named 'venv'
`python -m venv venv`

# Activate the virtual environment
`venv\Scripts\activate`

```bash```
# Clone the repository
`git clone <repository-url>`
`cd <repository-directory>`

# Install the necessary dependencies
`pip install -r requirements.txt`

# Usage

# To train and test the GCN model
`python gcn_train_test.py`

# To train and test the GAT model
`python gat_train_test.py`

# To train and test the GraphSAGE model
`python SAGE_train_test.py`

## Table of Contents for document
**0.** Abstract
**I.** Introduction
**II.** Literature Review 

    a. Graph Convolution Networks

    b. Graph SAGE

    c. Graph Attention Networks

**III.** Methodology
    
    a. Data Preprocessing

    b. Graph Recommender Systems

    c. GCN Implementation

    d. Graph Sage Implementation

    e. GAT Implementation

**IV.** Results

**V.** Discussion

**VI.** Conclusion

**VII.** References

**VIII.** Appendix

## more notes 
**Group Members.** George Kaceli and Noah Adams 
