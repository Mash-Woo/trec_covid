import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
import math
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BM25:
    """
    BM25 (Best Matching 25) ranking function implementation.
    
    Parameters:
    -----------
    k1 : float, default=1.5
        Term frequency saturation parameter
    b : float, default=0.75
        Length normalization parameter
    """
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_count = 0
        
    def tokenize(self, text):
        """Simple tokenization: lowercase and split on non-alphanumeric"""
        if pd.isna(text):
            return []
        text = str(text).lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def fit(self, corpus):
        """
        Fit BM25 parameters on document corpus.
        
        Parameters:
        -----------
        corpus : list of str
            List of documents (strings)
        """
        print("Tokenizing documents...")
        tokenized_corpus = [self.tokenize(doc) for doc in tqdm(corpus)]
        
        self.doc_count = len(tokenized_corpus)
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / self.doc_count
        
        # Calculate document frequencies
        print("Calculating document frequencies...")
        df = defaultdict(int)
        for doc in tqdm(tokenized_corpus):
            unique_tokens = set(doc)
            for token in unique_tokens:
                df[token] += 1
        
        # Calculate IDF scores
        print("Calculating IDF scores...")
        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
        
        # Store term frequencies for each document
        self.doc_freqs = []
        for doc in tokenized_corpus:
            freq_dict = Counter(doc)
            self.doc_freqs.append(freq_dict)
        
        print(f"Fitted BM25 on {self.doc_count} documents")
        return self
    
    def get_scores(self, query):
        """
        Calculate BM25 scores for a query against all documents.
        
        Parameters:
        -----------
        query : str
            Query string
            
        Returns:
        --------
        scores : numpy array
            BM25 scores for each document
        """
        query_tokens = self.tokenize(query)
        scores = np.zeros(self.doc_count)
        
        for token in query_tokens:
            if token not in self.idf:
                continue
            
            idf_score = self.idf[token]
            
            for doc_idx in range(self.doc_count):
                if token not in self.doc_freqs[doc_idx]:
                    continue
                
                freq = self.doc_freqs[doc_idx][token]
                doc_len = self.doc_len[doc_idx]
                
                # BM25 formula
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                scores[doc_idx] += idf_score * (numerator / denominator)
        
        return scores
    
    def get_top_n(self, query, n=10):
        """
        Get top N documents for a query.
        
        Parameters:
        -----------
        query : str
            Query string
        n : int
            Number of top documents to return
            
        Returns:
        --------
        top_indices : numpy array
            Indices of top N documents
        """
        scores = self.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:n]
        return top_indices


def load_data():
    """Load topics and metadata"""
    print("Loading topics...")
    topics = pd.read_csv('topics-rnd3.csv')
    
    print("Loading metadata (this may take a while)...")
    # Load metadata with selected columns to reduce memory usage
    metadata = pd.read_csv('CORD-19/CORD-19/metadata.csv', 
                          usecols=['cord_uid', 'title', 'abstract'],
                          low_memory=False)
    
    # Remove rows with missing cord_uid
    metadata = metadata[metadata['cord_uid'].notna()]
    
    print(f"Loaded {len(topics)} topics and {len(metadata)} documents")
    return topics, metadata


def create_corpus(metadata):
    """Create text corpus by combining title and abstract"""
    print("Creating document corpus...")
    corpus = []
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        title = str(row['title']) if pd.notna(row['title']) else ''
        abstract = str(row['abstract']) if pd.notna(row['abstract']) else ''
        
        # Combine title and abstract (give more weight to title by including it twice)
        doc_text = f"{title} {title} {abstract}"
        corpus.append(doc_text)
    
    return corpus


def generate_submission(topics, metadata, bm25, top_n):
    """Generate submission file in required format"""
    print(f"\nGenerating submission with top {top_n} documents per topic...")
    
    results = []
    
    for idx, row in tqdm(topics.iterrows(), total=len(topics)):
        topic_id = row['topic-id']
        
        # Combine query, question, and narrative for better retrieval
        query_text = f"{row['query']} {row['question']} {row['narrative']}"
        
        # Get top N documents
        top_indices = bm25.get_top_n(query_text, n=top_n)
        
        # Add to results
        for doc_idx in top_indices:
            cord_id = metadata.iloc[doc_idx]['cord_uid']
            results.append({
                'topic-id': topic_id,
                'cord-id': cord_id
            })
    
    # Create DataFrame and save
    submission_df = pd.DataFrame(results)
    output_file = 'bm25_submission' + str(top_n) + '.csv'
    submission_df.to_csv(output_file, index=False)
    
    print(f"\nSubmission saved to: {output_file}")
    print(f"Total entries: {len(submission_df)}")
    print(f"Topics covered: {submission_df['topic-id'].nunique()}")
    
    return submission_df


def main():
    """Main execution function"""
    print("="*60)
    print("BM25 Document Retrieval for TREC-COVID")
    print("="*60)
    
    # Load data
    topics, metadata = load_data()
    
    # Create corpus
    corpus = create_corpus(metadata)
    
    # Initialize and fit BM25
    print("\n" + "="*60)
    print("Training BM25 Model")
    print("="*60)
    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(corpus)
    
    # Generate submission
    print("\n" + "="*60)
    print("Generating Submission")
    print("="*60)
    generate_submission(topics, metadata, bm25, top_n=10)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()