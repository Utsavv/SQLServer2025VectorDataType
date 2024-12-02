import pyodbc
from sentence_transformers import SentenceTransformer
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Database connection parameters from environment variables
server = os.getenv('SERVER')
database = os.getenv('DATABASE')
username = os.getenv('USERNAME')
password = os.getenv('PASSWORD')
driver = '{ODBC Driver 17 for SQL Server}'

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

# Create the Vectors table
def create_table(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS VectorDB (
                Id INT IDENTITY(1,1) PRIMARY KEY,
                DocumentText NVARCHAR(MAX),
                Embedding VECTOR(768) -- Adjust dimension based on model output
            )
        """)
        connection.commit()
        print("Table created successfully!")
    except Exception as e:
        print("Error creating table:", e)

# Generate embeddings from a document
def generate_embeddings(file_path):
    try:
        # Load a pre-trained Sentence Transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other models if preferred

        # Read document
        with open(file_path, 'r', encoding='utf-8') as file:
            document_text = file.read()
        
        # Generate embeddings
        embeddings = model.encode([document_text])
        print("Embeddings generated successfully!")
        return document_text, embeddings[0].tolist()
    except Exception as e:
        print("Error generating embeddings:", e)
        return None, None

# Insert data into the table
def insert_data(connection, document_text, embedding):
    try:
        cursor = connection.cursor()
        embedding_json = json.dumps(embedding)  # Convert the embedding to a JSON format
        cursor.execute("""
            INSERT INTO VectorDB (DocumentText, Embedding)
            VALUES (?, ?)
        """, document_text, embedding_json)
        connection.commit()
        print("Data inserted successfully!")
    except Exception as e:
        print("Error inserting data:", e)

# Main function
def main():
    # Connect to the database
    connection = connect_to_db()
    if connection is None:
        return
    
    # Create the VectorDB table
    create_table(connection)
    
    # Generate embeddings from the document
    document_text, embedding = generate_embeddings('Documentation.txt')
    if document_text is None or embedding is None:
        return

    # Insert embeddings into the database
    insert_data(connection, document_text, embedding)

    # Close the connection
    connection.close()
    print("Connection closed!")

if __name__ == "__main__":
    main()
