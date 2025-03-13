import streamlit as st
import requests
import pandas as pd
import json
import os

# Configure page
st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
    }
    .book-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #3B82F6;
    }
    .book-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1F2937;
    }
    .book-author {
        font-size: 1rem;
        color: #4B5563;
    }
    .book-rank {
        font-size: 0.9rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<div class="main-header">📚 Book Recommender System</div>', unsafe_allow_html=True)
st.markdown("""
This application provides personalized book recommendations based on collaborative filtering.
Choose a user to see their recommended books or search for similar books to a specific title.
""")

# API configuration - Get from environment variable with fallback
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
st.sidebar.text(f"API URL: {API_BASE_URL}")

# Function to load book data
@st.cache_data
def load_book_data():
    try:
        # Try to get book data from API
        response = requests.get(f"{API_BASE_URL}/books")
        if response.status_code == 200:
            books = response.json()
            return pd.DataFrame(books)
    except:
        st.info("Could not load books from API, trying local files instead.")
    
    # If API doesn't provide book data, use a local fallback
    try:
        # Look for processed book data in different possible locations
        potential_paths = [
            os.path.join('..', 'data', 'processed', 'books.csv'),
            os.path.join('..', 'data', 'processed', 'book_id_mapping.csv'),
            os.path.join('..', 'data', 'raw', 'books.csv'),
            os.path.join('..', 'data', 'backup', 'books.csv'),
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                st.success(f"Found book data at {path}")
                df = pd.read_csv(path)
                
                # Ensure we have the expected columns
                required_cols = ['book_id', 'title']
                
                # If we have book_id_mapping.csv, it might have different column names
                if 'original_id' in df.columns and 'book_id' not in df.columns:
                    df = df.rename(columns={'original_id': 'book_id'})
                
                # If we're missing the authors column, add a placeholder
                if 'authors' not in df.columns and 'author' in df.columns:
                    df = df.rename(columns={'author': 'authors'})
                elif 'authors' not in df.columns:
                    df['authors'] = 'Unknown'
                
                # Verify we have the minimum required columns
                if all(col in df.columns for col in required_cols):
                    # Limit to 1000 books for performance
                    return df.head(1000)
                
        # If no files worked, create a sample dataset
        st.warning("Could not find valid book data files, using sample data.")
        return pd.DataFrame({
            'book_id': range(1, 11),
            'title': [f"Example Book {i}" for i in range(1, 11)],
            'authors': [f"Author {i}" for i in range(1, 11)]
        })
    except Exception as e:
        st.error(f"Error loading book data: {str(e)}")
        # Return a small example dataset as last resort
        return pd.DataFrame({
            'book_id': range(1, 11),
            'title': [f"Example Book {i}" for i in range(1, 11)],
            'authors': [f"Author {i}" for i in range(1, 11)]
        })

# Sidebar configuration
st.sidebar.markdown('<div class="sub-header">Options</div>', unsafe_allow_html=True)
recommendation_type = st.sidebar.radio(
    "Select Recommendation Type",
    ["User Recommendations", "Similar Books"]
)

# Function to get user recommendations
def get_user_recommendations(user_id, max_recommendations=5):
    try:
        response = requests.get(f"{API_BASE_URL}/recommend/user/{user_id}")
        if response.status_code == 200:
            data = response.json()
            return data.get("recommendations", [])[:max_recommendations]
        else:
            st.error(f"Error fetching recommendations: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []

# Function to get similar books
def get_similar_books(book_id, max_recommendations=5):
    try:
        response = requests.get(f"{API_BASE_URL}/similar-books/{book_id}")
        if response.status_code == 200:
            data = response.json()
            return data.get("recommendations", [])[:max_recommendations]
        else:
            st.error(f"Error fetching similar books: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []

# Check if API is available
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except:
        return False

# Display recommendation results
def display_book_recommendations(recommendations):
    if not recommendations:
        st.info("No recommendations found.")
        return
    
    for book in recommendations:
        st.markdown(f"""
        <div class="book-card">
            <div class="book-title">{book['title']}</div>
            <div class="book-author">By: {book['authors']}</div>
            <div class="book-rank">Rank: {book['rank']}</div>
        </div>
        """, unsafe_allow_html=True)

# Main application logic
if not check_api_health():
    st.error(f"""
    ⚠️ Cannot connect to the recommendation API!
    
    Please ensure:
    1. The Docker containers are running (`docker-compose up`)
    2. The API is accessible at {API_BASE_URL}
    3. No firewall is blocking the connection
    """)
else:
    st.success("✅ Connected to Book Recommendation API successfully!")
    
    # Load book data for the dropdown
    books_df = load_book_data()
    
    if recommendation_type == "User Recommendations":
        st.markdown('<div class="sub-header">User Recommendations</div>', unsafe_allow_html=True)
        
        # User selection
        user_id = st.number_input("Enter User ID", min_value=1, max_value=1000, value=250)
        num_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
        
        if st.button("Get Recommendations"):
            with st.spinner("Fetching recommendations..."):
                recommendations = get_user_recommendations(user_id, num_recommendations)
                st.markdown(f"#### Top {len(recommendations)} Recommendations for User {user_id}")
                display_book_recommendations(recommendations)
    
    else:  # Similar Books
        st.markdown('<div class="sub-header">Similar Books</div>', unsafe_allow_html=True)
        
        # Book selection via searchable dropdown
        if not books_df.empty:
            # Create a search box for filtering books
            search_term = st.text_input("Search for a book title", "")
            
            # Filter books based on search term
            if search_term:
                filtered_books = books_df[books_df['title'].str.contains(search_term, case=False)]
            else:
                filtered_books = books_df.head(100)  # Show first 100 books if no search term
            
            if filtered_books.empty:
                st.warning("No books match your search. Try a different term.")
                book_id = None
            else:
                # Format books for selection
                book_options = [f"{row['title']} (by {row['authors']})" if 'authors' in row else row['title'] 
                                for _, row in filtered_books.iterrows()]
                
                # Add a dropdown for selecting from filtered books
                selected_book = st.selectbox("Select a book", book_options)
                
                # Get the book_id from the selected book
                selected_idx = book_options.index(selected_book)
                book_id = filtered_books.iloc[selected_idx]['book_id']
                
                st.write(f"Selected Book ID: {book_id}")
        else:
            st.error("Unable to load book data for selection")
            book_id = st.number_input("Enter Book ID manually", min_value=1, max_value=1000, value=57)
        
        num_recommendations = st.slider("Number of Similar Books", min_value=1, max_value=10, value=5)
        
        if st.button("Find Similar Books") and book_id:
            with st.spinner("Finding similar books..."):
                similar_books = get_similar_books(book_id, num_recommendations)
                st.markdown(f"#### Top {len(similar_books)} Similar Books")
                display_book_recommendations(similar_books)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Powered by FastAPI and Collaborative Filtering")
