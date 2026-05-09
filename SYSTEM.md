# Memory
## Short-term memory (recent_history): 
The LLM is given exactly the last 5 rounds of history. (In the code: "recent_history": self.history[-5:])

## Long-term associative memory (similar_past_experiences): 
The LLM is also given the top 3 most similar past rounds retrieved from the FAISS vector database. The agent creates a vector embedding of the current situation (accuracies, detections, etc.) and asks the FAISS index to find past rounds that looked similar.