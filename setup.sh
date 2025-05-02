#!/bin/bash

# Create directory structure for Tourism RAG Chatbot
echo "Creating directory structure for Tourism RAG Chatbot..."

# Create assets folder for Chainlit
mkdir -p assets/roles

# Create placeholder images for tourism expert roles
echo "Creating placeholder images for tourism expert roles..."

# Function to create a colored square image as a placeholder
create_placeholder_image() {
  local role=$1
  local color=$2
  
  # Convert role name to lowercase and replace spaces with underscores
  local filename=$(echo $role | tr '[:upper:]' '[:lower:]' | tr ' ' '_').png
  
  # Create a colored square image
  echo "Creating placeholder image for $role"
  convert -size 512x512 xc:$color -gravity center -pointsize 36 -fill white \
    -annotate 0 "$role" "assets/roles/$filename"
}

# Create placeholder images for different tourism expert roles
create_placeholder_image "travel_trends_analyst" "#1E88E5"
create_placeholder_image "payment_specialist" "#D81B60"
create_placeholder_image "market_segmentation_expert" "#8E24AA"
create_placeholder_image "sustainability_tourism_advisor" "#43A047"
create_placeholder_image "gen_z_travel_specialist" "#FB8C00"
create_placeholder_image "luxury_tourism_consultant" "#6D4C41"
create_placeholder_image "tourism_analytics_expert" "#546E7A"
create_placeholder_image "general_tourism_assistant" "#26A69A"

# Create logs directory for tourism logger
mkdir -p logs

echo "Directory structure created successfully!"
echo "To run the Streamlit app: streamlit run app.py"
echo "To run the Chainlit app: chainlit run chainlit_app.py"