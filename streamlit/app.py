import streamlit as st
import requests
import pandas as pd
import json
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Flag for local model availability - set to False initially, will be updated if successful
MODELS_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Main title
st.title("📚 Book Recommender System")
st.markdown("""
This application provides personalized book recommendations based on collaborative filtering. Choose a user to see their recommended books or search for similar books to a specific title.
""")

# API configuration - Get from environment variable with fallback
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:9998")
st.sidebar.text(f"API URL: {API_BASE_URL}")

# Function to load book data
def load_book_data():
    try:
        # First try to load from books.csv
        books_path = os.path.join("..", "data", "processed", "books.csv")
        if os.path.exists(books_path):
            books_df = pd.read_csv(books_path)
        else:
            # Then try merged_train.csv 
            merged_path = os.path.join("..", "data", "processed", "merged_train.csv")
            if os.path.exists(merged_path):
                # Load all books from merged_train, but only keep unique book_ids
                books_df = pd.read_csv(merged_path)
                books_df = books_df.drop_duplicates(subset=['book_id'])
            else:
                # Fallback to API
                try:
                    response = requests.get(f"{API_BASE_URL}/books", timeout=5)
                    if response.status_code == 200:
                        books_df = pd.DataFrame(response.json())
                    else:
                        st.error(f"Error loading books from API: {response.status_code}")
                        books_df = pd.DataFrame(columns=['book_id', 'title', 'authors'])
                except Exception as e:
                    st.error(f"Error connecting to API: {e}")
                    books_df = pd.DataFrame(columns=['book_id', 'title', 'authors'])
        
        # Ensure required columns exist
        if 'book_id' not in books_df.columns and 'original_id' in books_df.columns:
            books_df['book_id'] = books_df['original_id']
        
        if 'authors' not in books_df.columns and 'author' in books_df.columns:
            books_df['authors'] = books_df['author']
        
        # Filter for English books only when language_code is available
        if 'language_code' in books_df.columns:
            books_df = books_df[books_df['language_code'] == 'en']
            
        # Extract genre information if available
        if 'genres' in books_df.columns:
            # Handle different genre formats (list/string)
            if isinstance(books_df['genres'].iloc[0], str):
                # Try to convert string representations of lists to actual lists
                try:
                    books_df['genres'] = books_df['genres'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [x])
                except:
                    # If conversion fails, keep as is
                    pass
        else:
            # Add empty genres column if not present
            books_df['genres'] = [[] for _ in range(len(books_df))]
        
        return books_df
    
    except Exception as e:
        st.error(f"Error loading book data: {e}")
        return pd.DataFrame(columns=['book_id', 'title', 'authors', 'genres'])

# Function to get user recommendations
def get_user_recommendations(user_id, max_recommendations=5):
    try:
        # Try to get recommendations from the API
        response = requests.get(
            f"{API_BASE_URL}/recommend/user/{user_id}", 
            params={
                "include_images": True,
                "num_recommendations": max_recommendations
            },
            timeout=10  # Increase timeout
        )
        
        if response.status_code == 200:
            return response.json()["recommendations"]
        else:
            st.error(f"Error getting recommendations: Status {response.status_code}, Message: {response.text}")
            return []
            
    except requests.exceptions.RequestException as e:
        st.error(f"API request error: {str(e)}")
        
        # Try with alternate URL if the primary fails
        try:
            alt_url = "http://127.0.0.1:9998"  # Direct IP address
            alt_response = requests.get(
                f"{alt_url}/recommend/user/{user_id}", 
                params={
                    "include_images": True,
                    "num_recommendations": max_recommendations
                },
                timeout=10
            )
            
            if alt_response.status_code == 200:
                return alt_response.json()["recommendations"]
        except Exception as inner_e:
            st.error(f"Alternate API request failed: {str(inner_e)}")
        
        return []

# Function to get similar books
def get_similar_books(book_id, max_recommendations=5):
    try:
        # Try to get similar books from the API
        response = requests.get(
            f"{API_BASE_URL}/similar-books/{book_id}", 
            params={
                "include_images": True,
                "num_recommendations": max_recommendations
            },
            timeout=10  # Increase timeout
        )
        
        if response.status_code == 200:
            return response.json()["recommendations"]
        else:
            st.error(f"Error getting similar books: Status {response.status_code}, Message: {response.text}")
            return []
            
    except requests.exceptions.RequestException as e:
        st.error(f"API request error: {str(e)}")
        
        # Try with alternate URL if the primary fails
        try:
            alt_url = "http://127.0.0.1:9998"  # Direct IP address
            alt_response = requests.get(
                f"{alt_url}/similar-books/{book_id}", 
                params={
                    "include_images": True,
                    "num_recommendations": max_recommendations
                },
                timeout=10
            )
            
            if alt_response.status_code == 200:
                return alt_response.json()["recommendations"]
        except Exception as inner_e:
            st.error(f"Alternate API request failed: {str(inner_e)}")
        
        return []

# Function to display book recommendations in a visually appealing format
def display_book_recommendations(recommendations):
    if not recommendations:
        st.info("No recommendations found.")
        return
        
    # Create columns for the grid layout - using Streamlit's native components
    # Display books in a grid layout
    num_cols = 3  # Number of columns in the grid
    cols = st.columns(num_cols)
    
    # Loop through books and add them to columns
    for i, book in enumerate(recommendations):
        col_idx = i % num_cols
        with cols[col_idx]:
            # Get image URL or use placeholder
            image_url = book.get('image_url', '')
            if not image_url or 'nophoto' in image_url:
                image_url = "https://islandpress.org/sites/default/files/default_book_cover_2015.jpg"
            
            # Display book image
            st.image(image_url, width=180)
            
            # Truncate long titles
            title = book.get('title', 'Unknown Title')
            if len(title) > 40:
                title = title[:37] + '...'
                
            # Truncate long author names
            author = book.get('authors', 'Unknown Author')
            if len(author) > 30:
                author = author[:27] + '...'
            
            # Display book info using markdown with styling
            st.markdown(f"**{title}**")
            st.markdown(f"*{author}*")
            st.markdown("---")

# Sidebar configuration
st.sidebar.markdown('<div class="sub-header">Options</div>', unsafe_allow_html=True)
recommendation_type = st.sidebar.radio(
    "Select Recommendation Type",
    ["User Recommendations", "Similar Books"]
)

# Check if API is available
def check_api_health():
    global API_BASE_URL
    try:
        st.sidebar.text(f"Trying to connect to {API_BASE_URL}/health")
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)  # Increased timeout
        if response.status_code == 200:
            st.sidebar.success(f"API connected successfully!")
            return True
        else:
            st.sidebar.error(f"API responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError as e:
        st.sidebar.error(f"Connection error: {str(e)}")
        # Try alternate localhost formats with increased timeout
        alt_urls = [
            "http://127.0.0.1:9998", 
            "http://0.0.0.0:9998", 
            "http://localhost:9998"
        ]
        for alt_url in alt_urls:
            if alt_url != API_BASE_URL:
                try:
                    st.sidebar.text(f"Trying alternate URL: {alt_url}")
                    alt_response = requests.get(f"{alt_url}/health", timeout=5)
                    if alt_response.status_code == 200:
                        st.sidebar.success(f"Connection successful via {alt_url}")
                        API_BASE_URL = alt_url
                        return True
                except Exception as e:
                    st.sidebar.text(f"Failed with {alt_url}: {str(e)}")
                    pass
        return False
    except Exception as e:
        st.sidebar.error(f"Error checking API health: {str(e)}")
        return False

# Main application logic
api_available = check_api_health()
if not api_available:
    st.error(f"""
    ⚠️ Cannot connect to the recommendation API!
    
    Please ensure:
    1. The Docker containers are running (`docker-compose up`)
    2. The API is accessible at {API_BASE_URL}
    3. No firewall is blocking the connection
    """)
    # Show a progress bar to simulate API connection attempt
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        progress_bar.progress(percent_complete + 1)
    st.info("Your API is not running. Please start it with `python src/api/api.py` in the project root directory.")

# Continue with app logic regardless of API status to allow testing UI with sample data
# Load book data for the dropdown
books_df = load_book_data()

if recommendation_type == "User Recommendations":
    st.markdown('<div class="sub-header">User Recommendations</div>', unsafe_allow_html=True)
    
    # User selection
    user_id = st.number_input("Enter User ID", min_value=1, max_value=1000, value=250)
    num_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
    
    if st.button("Get Recommendations"):
        if api_available:
            with st.spinner("Fetching recommendations..."):
                recommendations = get_user_recommendations(user_id, num_recommendations)
                st.markdown(f"#### Top {len(recommendations)} Recommendations for User {user_id}")
                display_book_recommendations(recommendations)
        else:
            st.error("Cannot get recommendations without API connection. Please start your API server.")
            
else:  # Similar Books
    st.markdown('<div class="sub-header">Find Similar Books</div>', unsafe_allow_html=True)
    
    # Load book data before creating filters
    books_df = load_book_data()
    
    # Extract unique genres and authors for filtering
    all_genres = set()
    if 'genres' in books_df.columns:
        # Flatten list of genres and get unique values
        for genres_list in books_df['genres']:
            if isinstance(genres_list, list):
                all_genres.update(genres_list)
            elif isinstance(genres_list, str):
                all_genres.add(genres_list)
    
    # Normalize and consolidate genres
    normalized_genres = {}
    genre_mapping = {}
    
    # Helper function to normalize genre names
    def normalize_genre(genre):
        if not genre or not isinstance(genre, str):
            return None
        
        # Normalize case
        normalized = genre.strip().title()
        
        # Handle specific consolidations
        if any(term in normalized for term in ["Adventure", "Adventures"]):
            return "Adventure"
        elif any(term in normalized for term in ["Juvenile", "Children", "Young Adult"]):
            return "Children & Young Adult"
        elif any(term in normalized for term in ["Mystery", "Thriller", "Detective", "Crime"]):
            return "Mystery & Thriller"
        elif any(term in normalized for term in ["Sci-Fi", "Science Fiction", "SciFi"]):
            return "Science Fiction"
        elif any(term in normalized for term in ["Fantasy", "Supernatural", "Magic"]):
            return "Fantasy"
        elif any(term in normalized for term in ["Romance", "Love"]):
            return "Romance"
        elif any(term in normalized for term in ["Biography", "Autobiography", "Memoir"]):
            return "Biography & Memoir"
        elif any(term in normalized for term in ["History", "Historical"]):
            return "History"
        elif any(term in normalized for term in ["Literature", "Classics", "Literary"]):
            return "Literature & Classics"
        elif any(term in normalized for term in ["Fiction", "Novel"]) and "Non-Fiction" not in normalized:
            return "Fiction"
        elif any(term in normalized for term in ["Philosophy", "Psychology", "Religion"]):
            return "Philosophy & Religion"
        elif any(term in normalized for term in ["Science", "Technology", "Math"]):
            return "Science & Technology"
        elif any(term in normalized for term in ["Art", "Music", "Photography"]):
            return "Arts & Music"
        
        return normalized
    
    # Create mapping from original to normalized genres
    for genre in all_genres:
        norm_genre = normalize_genre(genre)
        if norm_genre:
            genre_mapping[genre] = norm_genre
            normalized_genres[norm_genre] = True
    
    # Sort normalized genres alphabetically
    consolidated_genres = sorted(normalized_genres.keys())
    
    # Get unique authors
    all_authors = sorted(books_df['authors'].unique().tolist()) if 'authors' in books_df.columns else []
    
    # Create columns for a more organized layout
    col1, col2 = st.columns(2)
    
    # Move genre filter to main page in the first column
    with col1:
        st.markdown("### Filter by Genre")
        selected_genres = st.multiselect(
            "Select one or more genres", 
            options=consolidated_genres,
            default=[]
        )
    
    # Author filter in the second column
    with col2:
        st.markdown("### Filter by Author")
        author_search = st.text_input("Search for author")
        filtered_authors = [author for author in all_authors if author_search.lower() in author.lower()] if author_search else all_authors[:100]
        selected_author = st.selectbox(
            "Select an author",
            options=["All Authors"] + filtered_authors
        )
    
    # Apply filters to books_df
    filtered_books = books_df.copy()
    
    # Filter by genre if genres are selected
    if selected_genres:
        # Keep only books that have at least one of the selected genres
        filtered_books = filtered_books[filtered_books['genres'].apply(
            lambda x: any(genre_mapping.get(genre, genre) in selected_genres for genre in x) if isinstance(x, list) else genre_mapping.get(x, x) in selected_genres
        )]
    
    # Filter by author if an author is selected
    if selected_author != "All Authors":
        filtered_books = filtered_books[filtered_books['authors'] == selected_author]
    
    # Book search/selection
    st.markdown("### Search by Title")
    search_term = st.text_input("Enter a book title", "")
    
    # Show top books when no search term is provided
    if not search_term:
        st.info("Enter a book title to search, or use the filters in the sidebar to narrow down options")
        
        # Check if we have books that match the current filters
        if filtered_books.empty:
            st.warning("⚠️ No books match the current combination of filters")
            
            # Provide helpful suggestions based on what's currently selected
            if selected_genres and selected_author != "All Authors":
                st.markdown("**Suggestions:**")
                st.markdown("- Try removing the author filter")
                st.markdown("- Try selecting different genres")
                st.markdown("- Try selecting only one filter type (either genre or author)")
                
                # Show authors who have written in the selected genres
                genre_authors = books_df[books_df['genres'].apply(
                    lambda x: any(genre_mapping.get(genre, genre) in selected_genres for genre in x) 
                    if isinstance(x, list) else genre_mapping.get(x, x) in selected_genres
                )]['authors'].unique()
                
                if len(genre_authors) > 0:
                    st.markdown(f"**Authors who write in the selected genres:**")
                    for author in sorted(genre_authors)[:5]:  # Show top 5 authors
                        st.markdown(f"- {author}")
                    if len(genre_authors) > 5:
                        st.markdown(f"... and {len(genre_authors) - 5} more")
            
            elif selected_genres:
                st.markdown("**Try a different genre or select fewer genres**")
            
            elif selected_author != "All Authors":
                # Show genres for the selected author
                author_genres = set()
                for _, row in books_df[books_df['authors'] == selected_author].iterrows():
                    genres = row.get('genres', [])
                    if isinstance(genres, list):
                        for g in genres:
                            if g and genre_mapping.get(g):
                                author_genres.add(genre_mapping.get(g))
                    elif genres and genre_mapping.get(genres):
                        author_genres.add(genre_mapping.get(genres))
                
                if author_genres:
                    st.markdown(f"**Genres for {selected_author}:**")
                    for genre in sorted(author_genres):
                        st.markdown(f"- {genre}")
            
            # Reset filters button
            if st.button("Reset Filters"):
                # This will cause a page reload with default values
                pass
                
            # Display unfiltered books as fallback
            st.markdown("### Popular Books (Filters Reset)")
            display_books = books_df.head(20)
        else:
            # Display a selection of books from the dataset
            display_books = filtered_books.head(50)  # Show first 50 books after filtering
            st.markdown("#### Popular Books")
    else:
        # Filter books based on search term, with expanded results
        display_books = filtered_books[filtered_books['title'].str.contains(search_term, case=False)].head(100)
        
        # Check if search term yielded no results
        if display_books.empty and not filtered_books.empty:
            st.warning(f"No books containing '{search_term}' found with the current filters.")
            st.markdown("**Try removing some filters or modifying your search term.**")
            
            # Show some alternative books that match the filters but not the search term
            st.markdown("#### Books matching your filters:")
            display_books = filtered_books.head(20)
        elif display_books.empty and filtered_books.empty:
            st.warning("No books match the current combination of filters and search term.")
            st.markdown("**Try removing some filters or modifying your search term.**")
            
            # Reset filters button
            if st.button("Reset Filters"):
                # This will cause a page reload with default values
                pass
                
            # Display unfiltered books that match search
            unfiltered_search = books_df[books_df['title'].str.contains(search_term, case=False)].head(20)
            if not unfiltered_search.empty:
                st.markdown("#### Books matching your search term (without filters):")
                display_books = unfiltered_search
            else:
                # No books match search at all, show popular books
                st.markdown("#### Popular Books:")
                display_books = books_df.head(20)
    

    # Create a more comprehensive dropdown with both ID and title for easier selection
    if not display_books.empty:
        book_options = {f"{row['title']} by {row['authors']} (ID: {row['book_id']})": row['book_id'] 
                       for _, row in display_books.iterrows()}
        selected_book = st.selectbox("Select a book", list(book_options.keys()))
        selected_book_id = book_options[selected_book]
        
        num_similar = st.slider("Number of Similar Books", min_value=1, max_value=10, value=5)
        
        if st.button("Find Similar Books"):
            if api_available:
                with st.spinner("Finding similar books..."):
                    similar_books = get_similar_books(selected_book_id, num_similar)
                    st.markdown(f"#### Top {len(similar_books)} Books Similar to '{selected_book}'")
                    display_book_recommendations(similar_books)
            else:
                st.error("Cannot find similar books without API connection. Please start your API server.")
    else:
        st.error("No books available to select. Please try different filters or reset your search.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Powered by FastAPI and Collaborative Filtering")
