# Travel Itinerary Planner

A Python-based travel itinerary planner built with LangChain, LangGraph, FAISS, and Gradio. It generates personalized day trip itineraries using user inputs (city, interests), Retrieval-Augmented Generation (RAG) with travel data, real-time API integration (mock), and a feedback loop for itinerary refinement.

## Features
- **FAISS Destination Retrieval**: Validates cities using a vector store for efficient matching.
- **RAG**: Enhances itineraries with travel blog/Wikivoyage data.
- **LLM Memory**: Retains conversation history for context-aware responses.
- **Real-Time APIs**: Mock integration for flights and hotels (replace with real APIs like Amadeus).
- **Feedback Loop**: Allows users to rate and refine itineraries via Gradio UI.
- **Gradio Interface**: User-friendly interface for inputting city, interests, and feedback.

