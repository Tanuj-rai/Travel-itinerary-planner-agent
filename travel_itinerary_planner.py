#!/usr/bin/env python
# travel_itinerary_planner.py
# Enhanced travel itinerary planner with FAISS, RAG, LLM memory, real-time APIs, and feedback loop
# Designed to run in Google Colab

import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import gradio as gr
import requests
import json
import pandas as pd

# Initialize the language model
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# Initialize conversation memory
memory = ConversationBufferMemory(return_messages=True)

# Initialize embeddings for FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Mock destination dataset (replace with actual CSV/JSON file in practice)
destination_data = [
    {"city": "Paris", "description": "Known for art, culture, Eiffel Tower, Louvre Museum, and cuisine."},
    {"city": "Tokyo", "description": "Vibrant city with temples, modern tech, sushi, and Shibuya crossing."},
    {"city": "New York", "description": "Iconic skyline, Broadway shows, Central Park, and diverse food."}
]
# Create FAISS index
texts = [d["description"] for d in destination_data]
cities = [d["city"] for d in destination_data]
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=[{"city": c} for c in cities])

# Mock travel blog/Wikivoyage data for RAG (replace with actual data)
travel_texts = [
    "Paris is famous for its art galleries like the Louvre and Orsay, romantic Seine River cruises, and Michelin-starred restaurants.",
    "Tokyo offers unique experiences like visiting Senso-ji Temple, exploring Akihabara for tech, and dining at Tsukiji Market.",
    "New York’s Broadway shows are a must-see, with Central Park offering a relaxing escape and food trucks galore."
]
rag_vectorstore = FAISS.from_texts(travel_texts, embeddings)

# Define the state for the travel planner
class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    city: str
    interests: List[str]
    itinerary: str
    feedback: str
    rating: int

# Define the itinerary prompt with RAG context
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a travel assistant. Create a day trip itinerary for {city} based on interests: {interests}. Use this context: {context}. Provide a bulleted itinerary."),
    ("human", "Create an itinerary for my day trip. Previous conversation: {history}")
])

# Mock API for flight and hotel data (replace with real API calls, e.g., Amadeus)
def fetch_flight_data(city: str) -> str:
    return f"Flights to {city}: Sample flight at $500 round-trip from major hubs (mock data)."

def fetch_hotel_data(city: str) -> str:
    return f"Hotels in {city}: Sample 4-star hotel at $150/night (mock data)."

# State machine functions
def input_city(city: str, state: PlannerState) -> PlannerState:
    query = f"City: {city}"
    results = vectorstore.similarity_search(query, k=1)
    if results:
        validated_city = results[0].metadata["city"]
    else:
        validated_city = city
    return {
        **state,
        "city": validated_city,
        "messages": state["messages"] + [HumanMessage(content=city)],
    }

def input_interests(interests: str, state: PlannerState) -> PlannerState:
    return {
        **state,
        "interests": [interest.strip() for interest in interests.split(",")],
        "messages": state["messages"] + [HumanMessage(content=interests)],
    }

def create_itinerary(state: PlannerState) -> str:
    query = f"{state['city']} {', '.join(state['interests'])}"
    rag_results = rag_vectorstore.similarity_search(query, k=2)
    context = " ".join([res.page_content for res in rag_results])
    history = memory.load_memory_variables({})["history"]
    response = llm.invoke(itinerary_prompt.format_messages(
        city=state["city"],
        interests=", ".join(state["interests"]),
        context=context,
        history=history
    ))
    state["itinerary"] = response.content
    state["messages"] += [AIMessage(content=response.content)]
    memory.save_context({"input": f"City: {state['city']}, Interests: {', '.join(state['interests'])}"}, {"output": response.content})
    flight_info = fetch_flight_data(state["city"])
    hotel_info = fetch_hotel_data(state["city"])
    return f"{response.content}\n\n**Travel Info**:\n- {flight_info}\n- {hotel_info}"

def refine_itinerary(feedback: str, rating: int, state: PlannerState) -> str:
    state["feedback"] = feedback
    state["rating"] = rating
    if rating < 3 and feedback:
        refined_prompt = ChatPromptTemplate.from_messages([
            ("system", "Refine the itinerary for {city} based on interests: {interests}. Original itinerary: {original}. User feedback: {feedback}."),
            ("human", "Revise my itinerary.")
        ])
        response = llm.invoke(refined_prompt.format_messages(
            city=state["city"],
            interests=", ".join(state["interests"]),
            original=state["itinerary"],
            feedback=feedback
        ))
        state["itinerary"] = response.content
        state["messages"] += [HumanMessage(content=feedback), AIMessage(content=response.content)]
        memory.save_context({"input": f"Feedback: {feedback}"}, {"output": response.content})
        return response.content
    return state["itinerary"]

# Define the Gradio application
def travel_planner(city: str, interests: str, feedback: str = "", rating: int = 5):
    state = {
        "messages": [],
        "city": "",
        "interests": [],
        "itinerary": "",
        "feedback": "",
        "rating": 5
    }
    state = input_city(city, state)
    state = input_interests(interests, state)
    if feedback and rating < 5:
        itinerary = refine_itinerary(feedback, rating, state)
    else:
        itinerary = create_itinerary(state)
    return itinerary

# Build and launch the Gradio interface
def main():
    interface = gr.Interface(
        fn=travel_planner,
        inputs=[
            gr.Textbox(label="Enter the city for your day trip"),
            gr.Textbox(label="Enter your interests (comma-separated)"),
            gr.Textbox(label="Provide feedback (optional)", placeholder="E.g., Add more evening activities"),
            gr.Slider(minimum=1, maximum=5, step=1, label="Rate the itinerary (1-5)", value=5)
        ],
        outputs=gr.Textbox(label="Generated Itinerary"),
        title="Enhanced Travel Itinerary Planner",
        description="Enter a city, interests, and optional feedback to generate or refine a personalized day trip itinerary.",
        theme='Yntec/HaleyCH_Theme_Orange_Green'
    )
    interface.launch(share=True)

if __name__ == "__main__":
    main()
