"""
Streamlit application for book recommendations - Simplified Demo Version
"""
import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add the current directory to path to find model_utils
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import functions
try:
    from model_utils import (
        get_books_df, get_ratings_df, get_recommender_model, 
        recommend_for_user, get_similar_books, get_popular_books,
        get_all_users, get_book_by_id
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .book-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = 1
if 'book_id' not in st.session_state:
    st.session_state['book_id'] = None

# Load data and model
@st.cache_resource
def load_data():
    try:
        books_df = get_books_df()
        ratings_df = get_ratings_df()
        model = get_recommender_model()
        return books_df, ratings_df, model
    except Exception as e:
        st.error(f"Error loading data or model: {str(e)}")
        return None, None, None

try:
    books_df, ratings_df, model = load_data()
    if books_df is None or ratings_df is None or model is None:
        st.error("Failed to load required data or model. Please check the logs.")
except Exception as e:
    st.error(f"Error initializing application: {str(e)}")
    books_df, ratings_df, model = None, None, None

# Sidebar
st.sidebar.title("📚 Book Recommender")
page = st.sidebar.radio(
    "Navigation",
    ["User Recommendations", "Similar Books", "Popular Books"]
)

# Model info in sidebar
model_path = os.path.join(current_dir, 'models', 'collaborative.pkl')
if os.path.exists(model_path):
    model_timestamp = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d")
else:
    model_timestamp = "Unknown"
    
st.sidebar.divider()
st.sidebar.subheader("System Info")
st.sidebar.info(f"Model updated: {model_timestamp}\nBooks: {len(books_df) if books_df is not None else 0}\nRatings: {len(ratings_df) if ratings_df is not None else 0}")

# Display book card
def display_book_card(book, col):
    with col:
        with st.container():
            st.markdown(f'<div class="book-card">', unsafe_allow_html=True)
            
            # Limit title length to prevent layout issues
            title = book["title"]
            if len(title) > 40:
                title = title[:37] + "..."
            st.subheader(title)
            
            # Display book image with simpler placeholder approach
            image_url = book.get("image_url", "")
            if image_url and not pd.isna(image_url) and image_url.startswith(('http://', 'https://')):
                try:
                    st.image(image_url, width=150)
                except Exception:
                    # Use a simple colored box instead of HTML/CSS
                    st.markdown("**No image available**")
            else:
                # Use a simple colored box instead of HTML/CSS
                st.markdown("**No cover image**")
            
            # Display book info
            st.write(f"**Author:** {book['authors']}")
            st.write(f"**Rating:** ⭐ {book.get('average_rating', 0.0):.1f}")
            
            if "similarity_score" in book:
                st.write(f"**Similarity:** {book['similarity_score']:.2f}")
            
            # Add buttons for interaction
            if st.button(f"More like this", key=f"similar_{book['book_id']}"):
                st.session_state['book_id'] = book['book_id']
                st.session_state['page'] = "Similar Books"
                st.experimental_rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# Function to get a list of books for the dropdown
@st.cache_data
def get_book_list():
    if books_df is not None:
        # Get a subset of books for the dropdown (limit to prevent overwhelming UI)
        book_subset = books_df.sort_values('average_rating', ascending=False).head(1000)
        
        # Create tuples of (book_id, title - author) for the dropdown
        book_list = []
        for _, row in book_subset.iterrows():
            book_id = row['book_id']
            title = row['title']
            author = row['authors']
            # Truncate long titles
            if len(title) > 50:
                title = title[:47] + "..."
            # Create a formatted string for dropdown
            display_text = f"{title} - {author}"
            book_list.append((book_id, display_text))
        
        return book_list
    return []

# Main content based on selected page
if page == "User Recommendations":
    st.header("Personalized Book Recommendations")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        user_list = get_all_users(limit=1000)
        user_id = st.selectbox("Select a user ID", user_list, 
                              index=user_list.index(st.session_state['user_id']) if st.session_state['user_id'] in user_list else 0)
        st.session_state['user_id'] = user_id
    
    with col2:
        num_recommendations = st.slider("Number of recommendations", 3, 12, 6)
    
    force_diverse = st.checkbox("Force diversity in recommendations", value=True)
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            recommendations = recommend_for_user(user_id, n=num_recommendations, force_diverse=force_diverse)
            
        if recommendations:
            # Ensure we only display the exact number of requested recommendations
            recommendations = recommendations[:num_recommendations]
            st.success(f"Found {len(recommendations)} recommendations for user {user_id}")
            
            # Display recommendations in a grid with 2 columns
            cols = st.columns(2)
            for i, book in enumerate(recommendations):
                display_book_card(book, cols[i % 2])
        else:
            st.warning(f"No recommendations found for user {user_id}")

elif page == "Similar Books":
    st.header("Find Similar Books")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Get books for dropdown menu
        book_list = get_book_list()
        
        # Create a dictionary to lookup book_id from the selected option
        book_options = {f"{display_text} (ID: {book_id})": book_id for book_id, display_text in book_list}
        
        # Default option - either from session state or first book
        default_option = None
        if st.session_state['book_id'] and book_list:
            # Try to find the book in our list
            for book_id, display_text in book_list:
                if book_id == st.session_state['book_id']:
                    default_option = f"{display_text} (ID: {book_id})"
                    break
        
        # If no default found, use the first book
        if not default_option and book_options:
            default_option = list(book_options.keys())[0]
        
        # Create the dropdown menu
        selected_book_option = st.selectbox(
            "Select a book", 
            options=list(book_options.keys()) if book_options else ["No books available"],
            index=list(book_options.keys()).index(default_option) if default_option in book_options else 0
        )
        
        # Extract book_id from selection
        if book_options and selected_book_option in book_options:
            book_id = book_options[selected_book_option]
            st.session_state['book_id'] = book_id
        else:
            book_id = 1  # Fallback
            st.session_state['book_id'] = book_id
            
    with col2:
        num_similar = st.slider("Number of similar books", 3, 12, 6)
    
    # Get book details to display
    if book_id:
        book_details = get_book_by_id(book_id)
        if book_details:
            st.subheader(f"Selected Book: {book_details['title']}")
            col1, col2 = st.columns([1, 3])
            with col1:
                if book_details.get('image_url'):
                    st.image(book_details['image_url'], width=200)
                else:
                    st.markdown("**No cover image available**")
            
            with col2:
                st.write(f"**Author:** {book_details['authors']}")
                st.write(f"**Rating:** ⭐ {book_details.get('average_rating', 0.0):.1f}")
                if book_details.get('ratings_count'):
                    st.write(f"**Ratings:** {book_details['ratings_count']} ratings")
                if book_details.get('description'):
                    st.write("**Description:**")
                    st.write(book_details['description'])
    
    if st.button("Find Similar Books", type="primary"):
        st.session_state['book_id'] = book_id
        with st.spinner("Finding similar books..."):
            similar_books = get_similar_books(book_id, n=num_similar)
        
        if similar_books:
            # Ensure we only display the exact number of requested similar books
            similar_books = similar_books[:num_similar]
            st.success(f"Found {len(similar_books)} similar books")
            
            # Display similar books in a grid with 2 columns
            cols = st.columns(2)
            for i, book in enumerate(similar_books):
                display_book_card(book, cols[i % 2])
        else:
            st.warning(f"No similar books found for book ID: {book_id}")
            
            # Suggest popular books instead
            st.info("Here are some popular books you might enjoy instead:")
            with st.spinner("Loading popular books..."):
                popular_books = get_popular_books(limit=6)
                if popular_books:
                    cols = st.columns(2)
                    for i, book in enumerate(popular_books):
                        display_book_card(book, cols[i % 2])

elif page == "Popular Books":
    st.header("Popular Books")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        num_books = st.slider("Number of books", 3, 12, 6)
    with col2:
        randomize = st.checkbox("Randomize selection", value=True)
    with col3:
        seed = st.number_input("Random seed", min_value=1, value=42, help="Set a seed for reproducible results")
    
    if st.button("Get Popular Books", type="primary"):
        with st.spinner("Finding popular books..."):
            popular_books = get_popular_books(limit=num_books, randomize=randomize, seed=seed)
        
        if popular_books:
            # Ensure we only display the exact number of requested popular books
            popular_books = popular_books[:num_books]
            st.success(f"Found {len(popular_books)} popular books")
            
            # Display popular books in a grid with 2 columns
            cols = st.columns(2)
            for i, book in enumerate(popular_books):
                display_book_card(book, cols[i % 2])
        else:
            st.warning("No popular books found")

# Footer
st.divider()
st.caption("Book Recommender System | Made with Streamlit | MLOps Demo")
