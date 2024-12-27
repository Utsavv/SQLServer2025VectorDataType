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
driver = '{ODBC Driver 18 for SQL Server}'

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
            DROP TABLE IF EXISTS VectorDB
            CREATE TABLE VectorDB (
                Id INT IDENTITY(1,1) PRIMARY KEY,
                DocumentText NVARCHAR(MAX),
                Embedding VECTOR(384)
            )
        """)
        connection.commit()
        print("Table created successfully!")
    except Exception as e:
        print("Error creating table:", e)

# Generate embeddings for each line of the input (file or text)
def generate_embeddings(input_data):
    """
    Generate embeddings for each line of the input, which can be either a file path or a text string.
    
    Args:
        input_data (str): Path to a file or a string containing text.

    Returns:
        list of tuples: Each tuple contains a line and its corresponding embedding.
    """
    try:
        # Load the model, in prod implementation its better to load model once.
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        embeddings_data = []
        lines = []
        
        # Check if input_data is a file path or a text string
        if "\n" in input_data or not input_data.strip().endswith(('.txt', '.csv', '.json')):
            # Treat as a single text string with potential multiline content
            lines = input_data.splitlines()
        else:
            # Treat as a file path
            with open(input_data, 'r', encoding='utf-8') as file:
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
                INSERT INTO VectorDB (DocumentText, Embedding)
                VALUES (?, CAST(CAST(? as NVARCHAR(MAX)) AS VECTOR(384)))
            """, document_text, embedding_json)
        connection.commit()
        print("All data inserted successfully!")
    except Exception as e:
        print(f"Error inserting data: {e}")


def vector_search_sql(query, conn, num_results=5):
    
    # Create a cursor object
    cursor = conn.cursor()

    # Generate the query embedding for the user's search query
    user_query_embedding = generate_embeddings(query)
    
    # SQL query for similarity search using the function vector_distance to calculate cosine similarity
    sql_similarity_search = f"""
    SELECT TOP(?) DocumentText,
           1-vector_distance('cosine', CAST(CAST(? AS NVARCHAR(MAX)) AS VECTOR(384)), embedding) AS similarity_score,
           vector_distance('cosine', CAST(CAST(? AS NVARCHAR(MAX)) AS VECTOR(384)), embedding) AS distance_score
    FROM dbo.VectorDB
    ORDER BY distance_score 
    """

    cursor.execute(sql_similarity_search, num_results, json.dumps(user_query_embedding), json.dumps(user_query_embedding))
    results = cursor.fetchall()

    # Close the database connection
    conn.close()

    return results
    


# Main function
def main():
    # Connect to the database
    connection = connect_to_db()
    if connection is None:
        return
    
    # Create the VectorStagingTable table
    create_table(connection)   
    
    # Generate embeddings per line from the document
    embeddings_data = generate_embeddings('Documentation.txt')
    
    if not embeddings_data:
        print("No embeddings generated. Exiting.")
        connection.close()
        return
    
    # Insert embeddings into the database
    insert_data(connection, embeddings_data)
    
    #example usage
    vector_search_sql("How can you manage and simplify multiple conditions together in the Rule Engine for complex scenarios like promotions?", connection, num_results=3)

    # Close the connection
    connection.close()
    print("Connection closed!")

if __name__ == "__main__":
    main()