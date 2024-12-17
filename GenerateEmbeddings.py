import json
from sentence_transformers import SentenceTransformer
import os

# Generate embeddings from a document and store in a JSON file
def generate_embeddings(file_path, model_name='all-MiniLM-L6-v2'):
    try:
        # Load a pre-trained Sentence Transformer model
        model = SentenceTransformer(model_name)
        
        # Read document content
        with open(file_path, 'r', encoding='utf-8') as file:
            document_text = file.read()
        
        # Generate embeddings
        embeddings = model.encode([document_text])
        
        print("Embeddings generated successfully!")
        
        return document_text, embeddings[0].tolist()
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None, None

# Save embeddings to a JSON file
def save_embeddings_to_json(document_text, embedding, output_file):
    try:
        # Create a dictionary to store text and embeddings
        data = {
            "DocumentText": document_text,
            "Embedding": embedding
        }
        
        # Write to JSON file
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)
        
        print(f"Embeddings saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving embeddings to JSON: {e}")

# Main function
def main():
    input_file = 'Documentation.txt'  # Input text file
    output_file = 'embeddings.json'   # Output JSON file
    
    # Generate embeddings
    document_text, embedding = generate_embeddings(input_file)
    
    if document_text is not None and embedding is not None:
        # Save embeddings to JSON
        save_embeddings_to_json(document_text, embedding, output_file)

if __name__ == "__main__":
    main()
