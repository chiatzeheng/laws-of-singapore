import pinecone
import os 
from dotenv import load_dotenv
load_dotenv('/home/tim/Documents/Personal/SingaporeLaws/.env')
from transformers import AutoTokenizer, AutoModel
import pinecone
import redis 

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")



# Initialize Pinecone client
pinecone.init(api_key=API_KEY)

# Load pre-trained language model and redis client
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=512)
model = AutoModel.from_pretrained(model_name)
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# # Process markdown files and convert text to vectors
# vectors = []
# metadata = []
# folder_path = "/home/tim/Documents/Personal/SingaporeLaws/scripts/sg-statutes/Acts"
# for file_name in os.listdir(folder_path):
#     if file_name.endswith(".md"):
#         with open(os.path.join(folder_path, file_name), "r") as f:
#             text = f.read()
#             inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
#             outputs = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
#             vectors.append(outputs)
#             metadata.append({"filename": file_name})
#             r.hset("sg-laws", mapping = {file_name: text})
#             print(f"Processed {file_name}")
            

vectors = []
metadata = []
for file_name in r.hkeys('sg-laws'):
    vector = [x for x in r.hget('sg-laws', file_name).split(',')]
    vectors.append(vector)
    metadata.append({'filename': file_name})
    print(f"Processed {file_name}")

def chunks(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]
        
# Initialize Pinecone index and add vectors and metadata
with pinecone.Index('sg-laws', pool_threads=30) as index:
    async_results = [
        index.upsert(vectors=vectors_chunk, ids=list(range(i, i+len(vectors_chunk))), metadata=metadata_chunk, async_req=True)
        for i, (vectors_chunk, metadata_chunk) in enumerate(zip(chunks(vectors, batch_size=100), chunks(metadata, batch_size=100)))
    ]
    print("Waiting for upserts to complete...")
    # Wait for and retrieve responses (this raises in case of error)
    [async_result.get() for async_result in async_results]

# Search for similar vectors based on a query vector
# query_text = "search query"
# query_inputs = tokenizer(query_text, return_tensors="pt")
# query_vector = model(**query_inputs).last_hidden_state.mean(dim=1).detach().numpy()
# results = pinecone_index.query(queries=[query_vector], top_k=10)
# print(results[0].metadata)


# print(r.hgetall())

