"""
Load testing script for Tourism RAG Chatbot using Locust.
Tests the performance of both API endpoints and UI components.

Usage:
    locust -f tests/stress/load_test.py
"""
import os
import time
import json
import random
from typing import Dict, Any, List, Optional

from locust import HttpUser, task, between, events
import requests
from bs4 import BeautifulSoup

# Sample tourism-related queries
TOURISM_QUERIES = [
    "What are the major travel trends for 2025?",
    "How do payment methods differ between luxury and budget travelers?",
    "What are the key market segments in sustainable tourism?",
    "Compare the travel preferences of Gen Z versus Millennials",
    "What is the impact of digital payments on tourism industry?",
    "How are luxury travel experiences evolving?",
    "What are the most popular destinations for adventure tourism?",
    "How is AI transforming the tourism booking experience?",
    "What sustainability initiatives are gaining traction in hotels?",
    "How has social media influenced destination marketing?",
    "What are the key challenges in tourism post-pandemic?",
    "How are travelers using mobile payments during international trips?",
    "What are the emerging trends in food tourism?",
    "How do different age groups research travel destinations?",
    "What payment security concerns do travelers have?",
    "How are hotels adapting to changing customer preferences?",
    "What role does influencer marketing play in tourism?",
    "How is data analytics being used in tourism market segmentation?",
    "What are the most effective loyalty programs in tourism?",
    "How are travel experiences being personalized with technology?"
]

# Tourism PDF sample URLs for testing
SAMPLE_PDFS = [
    "https://www.who.int/docs/default-source/coronaviruse/risk-comms-updates/update-28-covid-19-and-travel.pdf",
    "https://wttc.org/Portals/0/Documents/Reports/2021/Global%20Economic%20Impact%20and%20Trends%202021.pdf",
    "https://www.oecd.org/cfe/tourism/OECD-Tourism-Trends-Policies%202020-Highlights-ENG.pdf"
]

class TourismRagUser(HttpUser):
    """
    Simulated user for the Tourism RAG Chatbot.
    Tests both Streamlit and Chainlit interfaces.
    """
    # Wait between 3-10 seconds between user actions
    wait_time = between(3, 10)
    
    def on_start(self):
        """Initialize the user session."""
        # Choose interface: streamlit or chainlit
        self.interface = random.choice(["streamlit", "chainlit"])
        self.pdf_uploaded = False
        self.session_id = f"test_session_{random.randint(1000, 9999)}"
        self.expertise = random.choice([
            "Travel Trends Analyst", 
            "Payment Specialist",
            "Market Segmentation Expert", 
            "Sustainability Tourism Advisor"
        ])
        
        print(f"Starting test with {self.interface} interface")
        
        # Initialize session
        if self.interface == "streamlit":
            self.client.get("/")
        else:
            self.client.get("/")
    
    @task(1)
    def visit_home_page(self):
        """Visit the home page to initialize session."""
        if self.interface == "streamlit":
            self.client.get("/")
        else:
            self.client.get("/")
    
    @task(2)
    def upload_pdf(self):
        """Upload a sample PDF for analysis."""
        if self.pdf_uploaded:
            return
            
        # Choose a random PDF URL to download and upload
        pdf_url = random.choice(SAMPLE_PDFS)
        
        try:
            # Download the PDF (with timeout)
            response = requests.get(pdf_url, timeout=30)
            
            if response.status_code == 200:
                # Create a temporary file
                filename = os.path.basename(pdf_url)
                temp_file_path = f"/tmp/{filename}"
                
                with open(temp_file_path, "wb") as f:
                    f.write(response.content)
                
                # Upload the PDF
                if self.interface == "streamlit":
                    # Streamlit upload simulation
                    files = {'file': (filename, open(temp_file_path, 'rb'), 'application/pdf')}
                    upload_response = self.client.post(
                        "/_stcore/upload_file",
                        files=files
                    )
                else:
                    # Chainlit upload simulation
                    files = {'file': (filename, open(temp_file_path, 'rb'), 'application/pdf')}
                    upload_response = self.client.post(
                        "/api/upload_file",
                        files=files,
                        headers={"X-Chainlit-Session-Id": self.session_id}
                    )
                
                # Clean up
                try:
                    os.remove(temp_file_path)
                except:
                    pass
                
                self.pdf_uploaded = True
                print(f"Uploaded PDF: {filename}")
            else:
                print(f"Failed to download PDF from {pdf_url}: {response.status_code}")
        except Exception as e:
            print(f"Error during PDF upload test: {str(e)}")
    
    @task(5)
    def ask_question(self):
        """Send a random tourism question to the chatbot."""
        if not self.pdf_uploaded:
            # Upload a PDF first
            self.upload_pdf()
            return
            
        # Select a random query
        query = random.choice(TOURISM_QUERIES)
        
        try:
            if self.interface == "streamlit":
                # Streamlit query simulation
                # This is a simplified version that doesn't accurately represent
                # Streamlit's WebSocket communication
                data = {
                    "query": query,
                    "session_id": self.session_id,
                    "expertise": self.expertise
                }
                response = self.client.post(
                    "/_streamlit/query",
                    json=data
                )
            else:
                # Chainlit query simulation
                data = {
                    "message": query,
                    "chatId": self.session_id,
                    "streaming": True
                }
                response = self.client.post(
                    "/api/chat/message",
                    json=data,
                    headers={"X-Chainlit-Session-Id": self.session_id}
                )
            
            print(f"Asked: {query}")
        except Exception as e:
            print(f"Error during query test: {str(e)}")
    
    @task(1)
    def change_expertise(self):
        """Change the chatbot expertise."""
        if not self.pdf_uploaded:
            return
            
        # Choose a new expertise
        new_expertise = random.choice([
            "Travel Trends Analyst", 
            "Payment Specialist",
            "Market Segmentation Expert", 
            "Sustainability Tourism Advisor",
            "Gen Z Travel Specialist",
            "Luxury Tourism Consultant"
        ])
        
        if new_expertise == self.expertise:
            return
            
        try:
            if self.interface == "streamlit":
                # Streamlit expertise change simulation
                data = {
                    "expertise": new_expertise,
                    "session_id": self.session_id
                }
                response = self.client.post(
                    "/_streamlit/change_expertise",
                    json=data
                )
            else:
                # Chainlit expertise change simulation
                data = {
                    "action": "change_expertise",
                    "value": new_expertise,
                    "chatId": self.session_id
                }
                response = self.client.post(
                    "/api/action",
                    json=data,
                    headers={"X-Chainlit-Session-Id": self.session_id}
                )
            
            self.expertise = new_expertise
            print(f"Changed expertise to: {new_expertise}")
        except Exception as e:
            print(f"Error during expertise change test: {str(e)}")
    
    @task(1)
    def update_settings(self):
        """Update chatbot settings."""
        if not self.pdf_uploaded:
            return
            
        # Random settings
        settings = {
            "use_hybrid_retrieval": random.choice([True, False]),
            "use_reranker": random.choice([True, False]),
            "top_n": random.randint(3, 15)
        }
        
        try:
            if self.interface == "streamlit":
                # Streamlit settings update simulation
                data = {
                    "settings": settings,
                    "session_id": self.session_id
                }
                response = self.client.post(
                    "/_streamlit/update_settings",
                    json=data
                )
            else:
                # Chainlit settings update simulation
                data = {
                    "action": "update_settings",
                    "value": settings,
                    "chatId": self.session_id
                }
                response = self.client.post(
                    "/api/settings",
                    json=data,
                    headers={"X-Chainlit-Session-Id": self.session_id}
                )
            
            print(f"Updated settings: {settings}")
        except Exception as e:
            print(f"Error during settings update test: {str(e)}")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test environment."""
    print("Starting Tourism RAG Chatbot load test")
    print("="*60)
    print("Testing both Streamlit and Chainlit interfaces")
    print("Upload test will use public tourism PDFs")
    print("="*60)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Clean up after test."""
    print("="*60)
    print("Tourism RAG Chatbot load test completed")
    print("="*60)

# If running directly, print information
if __name__ == "__main__":
    print("""
    Tourism RAG Chatbot Load Test
    =============================
    
    This script tests the performance of the Tourism RAG Chatbot
    under various load conditions. It simulates users interacting
    with both the Streamlit and Chainlit interfaces.
    
    To run the test:
        locust -f tests/stress/load_test.py
        
    Then open http://localhost:8089 in your browser to
    configure and start the test.
    
    Recommended settings:
    - 10-50 users for moderate load testing
    - 50-200 users for stress testing
    - 5-10 users/second spawn rate
    """)