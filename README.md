# RAG Pipeline for Textbook Question Answering

This project implements a Retrieval-Augmented Generation (RAG) pipeline to answer questions based on a textbook. The pipeline uses a vector database (Pinecone) for efficient retrieval of relevant text chunks and a language model (FLAN-T5) for generating accurate and contextually relevant answers.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Libraries Used](#libraries-used)
4. [Installation](#installation)
5. [Set Up Pinecone and Prepare Your Dataset](#set-up-pinecone-and-prepare-your-dataset)
6. [Usage](#usage)
7. [Pipeline Workflow](#pipeline-workflow)


---

## Overview

The goal of this project is to build a RAG pipeline that:
1. **Retrieves** relevant context from a textbook using a vector database.
2. **Generates** accurate and concise answers to user queries using a language model.

The pipeline is designed to be scalable, efficient, and easy to use. It uses the `SentenceTransformer` model for generating text embeddings and Pinecone for vector storage and retrieval.

---

## Features

- **Document Embedding**: Uses `SentenceTransformer` to generate embeddings for textbook chunks.
- **Vector Database**: Stores and retrieves embeddings using Pinecone.
- **Adaptive Retrieval**: Dynamically adjusts retrieval parameters based on query length.
- **Reranking**: Reranks retrieved chunks based on similarity and keyword overlap.
- **Answer Generation**: Uses FLAN-T5 to generate concise and accurate answers.
- **Batch Processing**: Processes multiple queries efficiently.

---

## Libraries Used

- **Pinecone**: Vector database for storing and retrieving embeddings.
- **SentenceTransformers**: Open-source library for generating text embeddings.
- **Transformers**: Library for using pre-trained language models (FLAN-T5).
- **pdfplumber**: Extracts text from PDF files.
- **pandas**: Handles data processing and CSV exports.
- **tqdm**: Displays progress bars for batch processing.
- **jsonlines**: Reads JSON files containing queries.

---

## Installation

1. Clone the repository:
   
   git clone https://github.com/raslen10/PDF-RAG-Pipeline.git
   cd rag-pipeline

2. Install the required libraries:

   pip install pinecone-client sentence-transformers transformers pdfplumber pandas tqdm jsonlines
3. Set up Pinecone and prepare your dataset (see the next section).
4. Run the script:

   python rag_pipeline.py

## Set Up Pinecone and Prepare Your Dataset

1. Sign up for a Pinecone Account:

   Go to Pinecone and create an account if you don’t already have one.
   Once registered, navigate to the Pinecone dashboard to create an index.

2. Create an Index:

   In the Pinecone dashboard, create a new index with the following settings:
   Dimension: 384 (this matches the dimension of the all-MiniLM-L6-v2 embeddings).
   Metric: cosine (used for similarity search).
   Cloud Provider: Choose AWS or any other preferred provider.
   Region: Select a region (e.g., us-east-1).
   Get Your Pinecone API Key:

3. After creating the index, locate your API key in the Pinecone dashboard.

4. Replace your_pinecone_api_key in the code with your actual Pinecone API key.

## Dataset Preparation

1. Textbook PDF:

   Place your textbook PDF file (e.g., book.pdf) in the Dataset folder.
   Ensure the PDF is readable and contains the text you want to query.

2. Queries JSON File:

   Place your JSON file named queries.json in the Dataset folder.

3. Folder Structure:

   Dataset/
├── book.pdf                  # Textbook PDF
└── queries.json              # Queries in JSON format

4. Update File Paths in the Code:

   In the script, update the following variables with the correct paths:

   pdf_file: Path to your textbook PDF (e.g., "/content/Dataset/book.pdf").
   queries_file: Path to your queries JSON file (e.g., "/content/Dataset/queries.json").

## Usage

1. Update the file paths in the script:

   pdf_file: Path to your textbook PDF.
   queries_file: Path to your queries JSON file.

2. Run the script:

   python rag_pipeline.py

3. Check the output:

   The script will generate a submission.csv file containing the answers and references.

## Pipeline Workflow

1. Text Extraction:

   Extract text from the textbook PDF using pdfplumber.
   Split the text into chunks of 512 tokens with a 50-token overlap.

2. Embedding Generation:

   Generate embeddings for each chunk using SentenceTransformer.

3. Vector Database:

   Store embeddings in Pinecone.
   Retrieve relevant chunks for each query using cosine similarity.

4. Adaptive Retrieval:

   Adjust the number of retrieved chunks based on query length.

5. Reranking:

   Rerank retrieved chunks based on similarity score and keyword overlap.

6. Answer Generation:

   Use FLAN-T5 to generate answers based on the retrieved context.

7. Batch Processing:

   Process all queries in batch mode for efficiency.

