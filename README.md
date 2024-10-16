# NCERT Data RAG Chatbot

* This is built using langchain. 
* It uses gemini api for both ecoding and encoding and chroma for vector db.
* It supports three types of queries:
    * Simple queries that are not related to NCERT database:
        * For this classification method is used. 
        * First an embdding space is created of all the document chunks and are stored as numpy array.
        * The during initialization of chatbot, this numpy array is loaded.
        * And every query is checked if it belongs in the embedding cluster using ecludian distance between the query embedding and centroid of embedding cluster.
    * For Queries related to NCERT database we define two tools, first being calling rag workflow.
        * This queries the database using query embedding and generates the answer.
    * Second get metadata workflow.
        * This queries the database similar to above but returns the metadata like document name and page number of the matching query.

## Setup

1. Store the documents for chatbot in /backend/data folder.
2. Create a vector database by running create_database script.
3. Create embeddings array by running create_classifier script.
4. Fill in the Gemini api key in the env file. You can get it from [ai studio](https://aistudio.google.com/).
5. Turn up the backend by first going to the directory /backend/app, then running the command
    ```bash
    fastapi dev main.py 
    ```
6. Turn up the frontend by first going to the directory /frontend/app, then running the command
    ```bash
    streamlit run main.py
    ```

## Demo

![](resources/Screen%20Recording%202024-10-17%20035359.gif)

### Things to add
* History
* Streaming

