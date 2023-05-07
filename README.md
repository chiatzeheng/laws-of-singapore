
<h2>About</h2>
<p>This is a Python script that uses the Pinecone API to create an index of text documents and perform vector similarity search on them.</p>
<h2>Functionality</h2>
<p>The script uses the <code>AutoTokenizer</code> and <code>AutoModel</code> classes from the <code>transformers</code> library to load a pre-trained language model. It then uses this model to encode text documents into numerical vectors. The encoded vectors and corresponding metadata (e.g., file name) are stored in a Redis database.</p>
<p>The script then uses the Pinecone API to create an index and add the encoded vectors and metadata to it. The <code>chunks</code> function is used to split the vectors and metadata into smaller batches to be added to the index.</p>
<p>The script also includes commented out code for processing markdown files in a directory to create the encoded vectors and metadata.</p>
<p>Finally, there is commented out code for performing a vector similarity search on the index using a query vector generated from a query text. The results of the search include metadata for the most similar documents.</p>
