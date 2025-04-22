import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from gemini_utility import (load_gemini_pro_model,
                            gemini_pro_vision_response,
                            embedding_model_response,gemini_pro_response)

working_directory = os.path.dirname(os.path.abspath(__file__))

# Setting up the page configuration
st.set_page_config(
    page_title="Gemini AI",
    page_icon="🧠",
    layout="centered"
)

with st.sidebar:
    selected = option_menu("Gemini AI",
                           ["ChatBot",
                            "Image Captioning",
                            "Embed text",
                            "Ask me anything"],
                           menu_icon="robot", icons=["chat-dots-fill", "image-fill",
                                                     "textarea-t", "patch-question-fill"],
                           default_index=0
                           )


# Function to translate role between gemini-pro and streamlit
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role


if selected == "ChatBot":
    model = load_gemini_pro_model()

    # Initialize chat session in streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])
        st.info("New chat session initialized")

    # Streamlit page title
    st.title("🤖 ChatBot")

    # Debug information
    st.sidebar.subheader("Debug Info")
    if st.sidebar.checkbox("Show session history info"):
        history_length = len(st.session_state.chat_session.history) if hasattr(st.session_state.chat_session,
                                                                               'history') else 0
        st.sidebar.write(f"History length: {history_length}")

        if history_length > 0:
            st.sidebar.write("History entries:")
            for i, msg in enumerate(st.session_state.chat_session.history):
                st.sidebar.write(f"{i}: Role={msg.role}, Content={msg.parts[0].text[:20]}...")

    # Display the chat history
    if hasattr(st.session_state.chat_session, 'history'):
        for message in st.session_state.chat_session.history:
            role = translate_role_for_streamlit(message.role)
            with st.chat_message(role):
                st.markdown(message.parts[0].text)
    else:
        st.warning("Chat history not available in the expected format")

    # Input field for user's message
    user_prompt = st.chat_input("Ask Gemini Pro...")

    if user_prompt:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Get and display Gemini response
        with st.spinner("Thinking..."):
            gemini_response = st.session_state.chat_session.send_message(user_prompt)

        # Display gemini pro response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

# Image captioning page
if selected == "Image Captioning":

    st.title("📷 Snap Narrate")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if st.button("Generate Caption"):
        image = Image.open(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((800, 500))
            st.image(resized_img)

        default_prompt = "write a short caption for this image"  # change this prompt as per your requirement

        # get the caption of the image from the gemini-pro-vision LLM
        caption = gemini_pro_vision_response(default_prompt, image)

        with col2:
            st.info(caption)

# text embedding page
if selected == "Embed text":

    st.title("🔡 Embed Text")

    # text box to enter prompt
    user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")

    if st.button("Get Response"):
        response = embedding_model_response(user_prompt)
        st.markdown(response)


# text embedding model
if selected == "Ask me anything":

    st.title("❓ Ask me a question")

    # text box to enter prompt
    user_prompt = st.text_area(label='', placeholder="Ask me anything...")

    if st.button("Get Response"):
        response = gemini_pro_response(user_prompt)
        st.markdown(response)