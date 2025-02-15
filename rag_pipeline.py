# Import required libraries
from pinecone import Pinecone, ServerlessSpec
import json
import pdfplumber
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
import pandas as pd

# Initialize Pinecone
PINECONE_API_KEY = "your_pinecone_api_key"  # Replace with your Pinecone API key
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag-textbook-index"

# Delete existing index if it exists
if index_name in [index.name for index in pc.list_indexes()]:
    pc.delete_index(index_name)
    print(f"Deleted existing index: {index_name}")

# Create a new Pinecone index with the correct dimension (384 for SentenceTransformer)
pc.create_index(
    name=index_name,
    dimension=384,  # Match the dimension of SentenceTransformer embeddings
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
print(f"Created new index: {index_name} with dimension 384")

# Verify the index
index_info = pc.describe_index(index_name)
print(f"Index dimension: {index_info.dimension}")  # Should print 384

# Connect to the index
index = pc.Index(index_name)
print(f"Connected to Pinecone index: {index_name}")

# Load queries from JSON file
queries_file = "C:/Users/HP/Desktop/RAG_tasks/Dataset_RAG (1)/queries.json"
with open(queries_file, "r") as f:
    queries = json.load(f)

# Display sample queries
print("Sample queries:")
print(json.dumps(queries[:2], indent=4))  # Display first two queries

# Load PDF content using pdfplumber
pdf_file = "C:/Users/HP/Desktop/RAG_tasks/Dataset_RAG (1)/book.pdf"
book_content = []
with pdfplumber.open(pdf_file) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        book_content.append({"page": i+1, "text": text})

print("Sample book content:")
print(book_content[:3])  # Display first three pages

# Define chunking parameters
chunk_size = 512  # Max number of tokens per chunk
overlap = 50  # Number of overlapping tokens between chunks

# Function to chunk text
def chunk_text(text, chunk_size, overlap):
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Generate chunks from book content
chunks = []
for page in book_content:
    page_text = page.get("text", "")
    page_number = page.get("page", "N/A")
    if page_text.strip():  # Skip empty pages
        page_chunks = chunk_text(page_text, chunk_size, overlap)
        for chunk in page_chunks:
            chunks.append({"text": chunk, "metadata": {"page": page_number}})

print("Sample chunks:")
for chunk in chunks[:2]:
    print(chunk)

# Initialize Sentence Transformers model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and effective model

# Function to generate embeddings
def get_embedding(text):
    return embedding_model.encode(text)

# Upsert chunks into Pinecone with embeddings
for i, chunk in enumerate(chunks):
    chunk_text = chunk["text"]
    chunk_metadata = chunk["metadata"]

    embedding = get_embedding(chunk_text).tolist()  # Convert NumPy array to list
    chunk_metadata["text"] = chunk_text

    index.upsert([
        {
            "id": f"chunk-{i+1}",
            "values": embedding,
            "metadata": chunk_metadata
        }
    ])

    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{len(chunks)} chunks...")

print("All chunks have been embedded and stored in Pinecone.")

# Function to retrieve context from Pinecone
def retrieve_context(query, top_k=30, min_score=0.5):
    query_embedding = get_embedding(query).tolist()  # Convert NumPy array to list

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )

    # Extract retrieved text and scores
    retrieved_chunks = []
    for match in results.get('matches', []):
        score = match.get('score', 0)
        metadata = match.get('metadata', {})

        # Only keep results with a score higher than min_score
        if score >= min_score and 'text' in metadata:
            retrieved_chunks.append({
                "text": metadata["text"],
                "page": metadata.get("page", "N/A"),
                "score": score
            })

    return retrieved_chunks

# Adaptive retrieval for shorter queries
def adaptive_retrieve_context(query):
    short_query = len(query.split()) <= 5
    top_k = 10 if short_query else 30
    return retrieve_context(query, top_k=top_k)

# Function to rerank retrieved context
def rerank_retrieved_context(retrieved_chunks, query):
    query_terms = set(query.lower().split())

    for chunk in retrieved_chunks:
        text_terms = set(chunk["text"].lower().split())
        chunk["keyword_overlap"] = len(query_terms & text_terms)  # Count shared words

    # Sort by highest similarity score + keyword overlap
    retrieved_chunks.sort(key=lambda x: (x["score"], x["keyword_overlap"]), reverse=True)
    return retrieved_chunks[:10]  # Keep only top-ranked results

# Initialize QA model (FLAN-T5)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
qa_model = pipeline("text2text-generation", model="google/flan-t5-large", device=0)

# Function to generate answers
def generate_answer(query, retrieved_chunks, max_input_tokens=400):
    context_text = " ".join([chunk["text"] for chunk in retrieved_chunks])
    tokenized_context = tokenizer.encode(context_text, truncation=True, max_length=max_input_tokens, return_tensors="pt")
    trimmed_context = tokenizer.decode(tokenized_context[0], skip_special_tokens=True)
    prompt = (
        "Using the textbook excerpts below, provide a complete and accurate answer to the question. "
        "Ensure your response is well-structured and relevant to the text. "
        "If the textbook does not contain enough information, say so.\n\n"
        f"Textbook Excerpts:\n{trimmed_context}\n\n"
        f"Question: {query}\n\n"
        "Answer in 3-4 sentences:"
    )

    response = qa_model(prompt, max_length=100, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"]
    return response

# Test retrieval and answer generation
sample_query = "What is the scientific method in psychology?"
retrieved_context = retrieve_context(sample_query)
generated_answer = generate_answer(sample_query, retrieved_context)

print("Generated Answer:")
print(generated_answer)

# Batch processing for all queries
batch_inputs = []
query_ids = []
retrieved_contexts = []

for query in tqdm(queries, desc="🔍 Retrieving Contexts"):
    query_id = query["query_id"]
    question = query["question"]

    retrieved_context = adaptive_retrieve_context(question)
    retrieved_context = rerank_retrieved_context(retrieved_context, question)

    referenced_pages = list(set(str(chunk["page"]) for chunk in retrieved_context if "page" in chunk))
    references_json = json.dumps({"sections": [], "pages": referenced_pages})

    context_text = " ".join([chunk["text"] for chunk in retrieved_context])
    tokenized_context = tokenizer.encode(context_text, truncation=True, max_length=400, return_tensors="pt")
    trimmed_context = tokenizer.decode(tokenized_context[0], skip_special_tokens=True)

    prompt = (
        "Using the textbook excerpts below, provide a complete and accurate answer to the question. "
        "Ensure your response is well-structured and relevant to the text. "
        "If the textbook does not contain enough information, say so.\n\n"
        f"Textbook Excerpts:\n{trimmed_context}\n\n"
        f"Question: {question}\n\n"
        "Answer in 3-4 sentences:"
    )

    # Store inputs for batch processing
    batch_inputs.append(prompt)
    query_ids.append(query_id)
    retrieved_contexts.append((context_text, references_json))

# Generate answers in batch
print("Generating Answers in batch...")
batch_answers = qa_model(batch_inputs, max_length=100, do_sample=True, temperature=0.7, top_p=0.9)

# Convert results to submission format
submission_data = []
for i in range(len(query_ids)):
    submission_data.append({
        "ID": query_ids[i],
        "context": retrieved_contexts[i][0],
        "answer": batch_answers[i]["generated_text"],
        "references": retrieved_contexts[i][1]
    })

# Save submission file
submission_df = pd.DataFrame(submission_data)
submission_df.to_csv("C:/Users/HP/Desktop/RAG_tasks/Submission/submission.csv", index=False)
print("Submission file saved as submission.csv.")