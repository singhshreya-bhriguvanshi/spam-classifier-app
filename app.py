import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained("./spam_classifier_model")
tokenizer = BertTokenizer.from_pretrained("./spam_classifier_model")

# Define the prediction function
def classify_spam(input_text):
    # Tokenize the input text
    encodings = tokenizer(input_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    # Run the model on the input text
    with torch.no_grad():
        outputs = model(**encodings)

    # Get the predictions and confidence (softmax output)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, prediction = torch.max(probabilities, dim=-1)

    # Map predictions to labels
    label_map = {0: "Not spam", 1: "Spam"}
    result = label_map[prediction.item()]

    # Return result with confidence percentage
    return f"{result} (Confidence: {confidence.item() * 100:.2f}%)"

# Create Gradio interface
iface = gr.Interface(
    fn=classify_spam,  # Function to be called
    inputs=gr.Textbox(label="Enter your message"),  # Input component for user to type message
    outputs=gr.Textbox(label="Prediction Result"),  # Output component to show prediction
    live=True,  # Update output in real-time
    title="Spam Message Classifier",  # Title of the app
    description="Enter a message to check if it's spam or not along with the confidence score.",  # Description
)

# Launch the app
iface.launch()
