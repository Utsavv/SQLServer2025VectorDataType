import pyodbc
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Database connection parameters from environment variables
server = os.getenv('SERVER')
database = os.getenv('DATABASE')
username = os.getenv('USERNAME')
password = os.getenv('PASSWORD')
driver = '{ODBC Driver 18 for SQL Server}'

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Connect to Azure SQL Server
def connect_to_db():
    try:
        connection = pyodbc.connect(
            f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'
        )
        print("Connection successful!")
        return connection
    except Exception as e:
        print("Error connecting to database:", e)
        return None

# Create the table to store embeddings
def create_table(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("""
            DROP TABLE IF EXISTS VectorStagingTable;
            CREATE TABLE VectorStagingTable(
                Id INT IDENTITY(1,1) PRIMARY KEY,
                DocumentText NVARCHAR(MAX),
                Embedding NVARCHAR(MAX)
            );
        """)
        connection.commit()
        print("Table created successfully!")
    except Exception as e:
        print("Error creating table:", e)

# Generate embeddings for each line of the file
def generate_embeddings_per_line(file_path, model):
    try:
        embeddings_data = []
        # Read the file line by line
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Generate embeddings for each line
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                embedding = model.encode(line).tolist()
                embeddings_data.append((line, embedding))
        
        print("Embeddings generated for all lines!")
        return embeddings_data
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

# Insert data into the table
def insert_data(connection, embeddings_data):
    try:
        cursor = connection.cursor()
        for document_text, embedding in embeddings_data:
            embedding_json = json.dumps(embedding)  # Convert embedding to JSON string
            cursor.execute("""
                INSERT INTO VectorStagingTable (DocumentText, Embedding)
                VALUES (?, ?)
            """, document_text, embedding_json)
        connection.commit()
        print("All data inserted successfully!")
    except Exception as e:
        print(f"Error inserting data: {e}")

# Fetch all embeddings from the database
def fetch_embeddings(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DocumentText, Embedding FROM VectorStagingTable")
        rows = cursor.fetchall()
        
        # Parse embeddings from JSON strings
        embeddings_data = []
        for row in rows:
            document_text = row[0]
            embedding = json.loads(row[1])  # Convert stored JSON to a list
            embeddings_data.append((document_text, embedding))
        print("Embeddings fetched from the database.")
        return embeddings_data
    except Exception as e:
        print(f"Error fetching embeddings: {e}")
        return []

# Search for similar texts
def search_similar_texts(query_text, top_n=3):
    # Load the model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embedding for the query text
    query_embedding = model.encode(query_text).tolist()
    print("Query embedding generated.")

    # Connect to the database
    connection = connect_to_db()
    if connection is None:
        return

    # Fetch stored embeddings
    stored_data = fetch_embeddings(connection)
    connection.close()

    # Calculate cosine similarity
    similarities = []
    for document_text, embedding in stored_data:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((document_text, similarity))

    # Sort by similarity and return top N results
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    print(f"\nTop {top_n} most similar results:")
    for i, (text, sim) in enumerate(similarities[:top_n]):
        print(f"{i+1}. Similarity: {sim:.4f} | Text: {text}")

# Main function
def main():
    # Connect to the database
    connection = connect_to_db()
    if connection is None:
        return
    
    # Create the VectorStagingTable table
    create_table(connection)
    
    # Load the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings per line from the document
    embeddings_data = generate_embeddings_per_line('Documentation.txt', model)
    if not embeddings_data:
        print("No embeddings generated. Exiting.")
        connection.close()
        return
    
    # Insert embeddings into the database
    insert_data(connection, embeddings_data)

    # Close the connection
    connection.close()
    print("Connection closed!")

    # Search for similar texts
    query_text = input("\nEnter a query text to search: ")
    search_similar_texts(query_text, top_n=3)

if __name__ == "__main__":
    main()
