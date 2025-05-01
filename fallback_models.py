import numpy as np
import re
import string

class FallbackEmbeddingModel:
    """
    A simple fallback model that can be used when SentenceTransformer fails to load.
    Uses basic TF-IDF like approach to generate embeddings.
    """
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.word_vectors = {}
        self.rng = np.random.RandomState(42)  # Fixed seed for consistency
        
    def _preprocess_text(self, text):
        """Simple preprocessing function"""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def _get_word_vector(self, word):
        """Get or create a vector for a word"""
        if word not in self.word_vectors:
            # Create a consistent vector for this word
            self.word_vectors[word] = self.rng.rand(self.embedding_dim)
        return self.word_vectors[word]
    
    def encode(self, sentences, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        """Create embeddings for the given sentences"""
        if isinstance(sentences, str):
            sentences = [sentences]
            
        result = np.zeros((len(sentences), self.embedding_dim))
        
        for i, sentence in enumerate(sentences):
            words = self._preprocess_text(sentence).split()
            if not words:
                continue  # Skip empty sentences
                
            # Average the word vectors
            sentence_vector = np.zeros(self.embedding_dim)
            for word in words:
                sentence_vector += self._get_word_vector(word)
            
            if len(words) > 0:
                sentence_vector /= len(words)
                
            # Normalize the vector
            norm = np.linalg.norm(sentence_vector)
            if norm > 0:
                sentence_vector /= norm
                
            result[i] = sentence_vector
            
        return result

class FallbackGeminiAPI:
    """
    A fallback for the Gemini API that returns predictable responses
    for evaluation functions.
    """
    def __init__(self):
        print("Using fallback Gemini API implementation")
    
    def generate(self, prompt, schema=None):
        """
        Generate a response based on the prompt content.
        Returns JSON-compatible dummy data.
        """
        # Check what type of evaluation we're doing
        if "clarity" in prompt.lower():
            return '{"clarity_score": 7, "analysis": "This explanation is fairly clear.", "strengths": ["Good structure"], "weaknesses": ["Could use more examples"], "suggestions": ["Add examples"]}'
        elif "accuracy" in prompt.lower():
            return '{"accuracy": 7, "accuracy_exp": "The explanation is mostly accurate with minor issues."}'
        elif "completeness" in prompt.lower() or "relevance" in prompt.lower():
            return '{"completeness": 6, "clarity": 7, "relevance": 8}'
        elif "safety" in prompt.lower() or "toxicity" in prompt.lower():
            return '{"safety_score": 0.1}'
        else:
            # Generic response for other types
            return '{"score": 7, "analysis": "The content is adequate.", "strengths": ["Reasonable quality"], "weaknesses": ["Room for improvement"], "suggestions": ["Consider revising"]}'
