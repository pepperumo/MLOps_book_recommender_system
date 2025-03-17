import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Paper,
  Button,
  Alert,
  AlertTitle,
  CircularProgress,
  useTheme,
  TextField,
  Rating,
  Autocomplete
} from '@mui/material';
import LocalLibraryIcon from '@mui/icons-material/LocalLibrary';
import SearchIcon from '@mui/icons-material/Search';
import BookCard from '../components/BookCard';
import LogoLoading from '../components/LogoLoading';
import { useLocation } from 'react-router-dom';

const SimilarBooks = () => {
  const theme = useTheme();
  const location = useLocation();

  // State variables
  const [allBooks, setAllBooks] = useState([]);
  const [selectedBook, setSelectedBook] = useState(null);
  const [similarBooks, setSimilarBooks] = useState([]);
  const [recommendationCount, setRecommendationCount] = useState(5);
  const [loading, setLoading] = useState(false);
  const [booksLoading, setBooksLoading] = useState(true);
  const [error, setError] = useState(null);
  const [noResults, setNoResults] = useState(false);

  // Fetch similar books from API
  const fetchSimilarBooks = useCallback(async (bookId) => {
    if (!bookId) {
      console.error('Book ID is required to fetch similar books');
      setError('Book ID is required');
      setLoading(false);
      return;
    }
    
    console.log('Starting to fetch similar books for bookId:', bookId);
    setLoading(true);
    setError(null);
    setSimilarBooks([]);
    
    try {
      const apiUrl = `http://localhost:5000/api/similar-books/${bookId}?n=${recommendationCount}&include_images=true`;
      console.log(`Calling API URL: ${apiUrl}`);
      
      const response = await fetch(apiUrl);
      console.log('Response received:', response.status, response.statusText);
      
      if (!response.ok) {
        throw new Error(`Error fetching similar books: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('API Response data:', data);
      
      // Check the structure of the response
      let books = [];
      if (Array.isArray(data)) {
        // The API directly returned an array of books
        books = data;
        console.log('Response is an array with', books.length, 'books');
      } else if (data.recommendations && Array.isArray(data.recommendations)) {
        // The API returned an object with a recommendations array
        books = data.recommendations;
        console.log('Response has recommendations array with', books.length, 'books');
      } else if (data.book_id && data.recommendations) {
        // The API returned a different structure with book_id and recommendations
        books = data.recommendations;
        console.log('Response has book_id and recommendations with', books.length, 'books');
      } else {
        // Fall back to using the entire response data
        books = data;
        console.log('Using full response as books:', books);
      }
      
      // Update state with books
      console.log('Setting similar books:', books);
      setSimilarBooks(books);
      setNoResults(books.length === 0);
      
    } catch (err) {
      console.error('Error in fetchSimilarBooks:', err);
      setError(`Failed to load similar books: ${err.message}`);
      setSimilarBooks([]);
    } finally {
      console.log('Finished fetching similar books, setting loading to false');
      setLoading(false);
    }
  }, [recommendationCount]);

  // Simple fetch for all books - now called only once on component mount
  const fetchAllBooks = useCallback(async () => {
    setBooksLoading(true);
    setError(null);
    try {
      console.log('Fetching all books...');
      // Remove the limit to get all books from the API
      const response = await fetch('http://localhost:5000/api/books');
      if (!response.ok) {
        throw new Error(`Error fetching books: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Received books data:', data);
      
      // Handle the API response format where books are in the "books" property
      if (data.books && Array.isArray(data.books)) {
        console.log(`Fetched ${data.books.length} books from 'books' property`);
        setAllBooks(data.books);
      } else if (Array.isArray(data)) {
        console.log(`Fetched ${data.length} books from direct array`);
        setAllBooks(data);
      } else {
        console.error('Unexpected data format from books API:', data);
        setAllBooks([]);
        setError('Received invalid data format from server');
      }
    } catch (err) {
      console.error('Error fetching books:', err);
      setError(`Failed to load books: ${err.message}`);
      setAllBooks([]);
    } finally {
      setBooksLoading(false);
    }
  }, []);

  // Handle changing recommendation count
  const handleRecommendationCountChange = (event) => {
    const value = parseInt(event.target.value, 10);
    if (!isNaN(value) && value > 0 && value <= 12) {
      setRecommendationCount(value);
    }
  };

  // Handle book selection change from dropdown
  const handleBookChange = useCallback((event, newValue) => {
    if (newValue) {
      setSelectedBook(newValue);
    } else {
      setSimilarBooks([]);
    }
  }, []);

  // Handle book selection and fetchSimilarBooks
  const handleGetSimilar = (book) => {
    // Set selected book
    setSelectedBook(book);
    // Fetch similar books for this book
    fetchSimilarBooks(book.book_id);
  };

  // Effect to load initial data
  useEffect(() => {
    // Fetch all books on component mount - only done once
    fetchAllBooks();
  }, [fetchAllBooks]);

  // Separate effect for handling URL parameters after books are loaded
  useEffect(() => {
    if (allBooks.length === 0) return; // Wait until books are loaded
    
    // Check if there's a book_id in the URL
    const params = new URLSearchParams(location.search);
    const bookIdFromUrl = params.get('book_id');
    const fetchFromUrl = params.get('fetch') === 'true';
    
    if (bookIdFromUrl) {
      // Find the book in allBooks and set it as selected
      const bookInList = allBooks.find(b => 
        b.book_id && b.book_id.toString() === bookIdFromUrl.toString()
      );
      
      if (bookInList) {
        console.log('Found book from URL:', bookInList.title);
        setSelectedBook(bookInList);
        
        // If fetch=true is in the URL, automatically fetch recommendations
        if (fetchFromUrl) {
          console.log('Auto-fetching recommendations due to fetch=true in URL');
          fetchSimilarBooks(bookInList.book_id);
          
          // Remove the fetch parameter from URL to prevent re-fetching on navigation
          const newUrl = new URL(window.location);
          newUrl.searchParams.delete('fetch');
          window.history.replaceState({}, '', newUrl);
        }
      }
    }
  }, [location.search, allBooks, fetchSimilarBooks]);

  useEffect(() => {
    if (selectedBook?.book_id) {
      // Update URL without page reload
      window.history.pushState(
        { bookId: selectedBook.book_id },
        '',
        `/similar-books?book_id=${selectedBook.book_id}`
      );
    }
  }, [selectedBook]);

  useEffect(() => {
    const handlePopState = () => {
      const params = new URLSearchParams(window.location.search);
      const bookId = params.get('book_id');
      
      if (bookId) {
        // Check if book is in our allBooks array first
        const bookInList = allBooks.find(b => 
          b.book_id && b.book_id.toString() === bookId.toString()
        );
        
        if (bookInList) {
          setSelectedBook(bookInList);
        } else {
          setSelectedBook(null);
          setSimilarBooks([]);
        }
      } else {
        setSelectedBook(null);
        setSimilarBooks([]);
      }
    };
    
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [allBooks]);

  return (
    <Container maxWidth="xl" sx={{ py: 4, mt: 8 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center" sx={{ mb: 4 }}>
        Book Recommendation Engine
      </Typography>
      
      {/* Book Selection and Recommendation Count */}
      <Paper 
        elevation={3} 
        sx={{ 
          p: 3, 
          mb: 4, 
          borderRadius: '12px',
          background: theme.palette.mode === 'light' 
            ? '#fff' 
            : 'rgba(45, 45, 45, 0.98)',
          boxShadow: theme.shadows[4]
        }}
      >
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={4}>
            <Autocomplete
              id="book-select"
              options={allBooks || []}
              value={selectedBook}
              onChange={handleBookChange}
              getOptionLabel={(option) => option?.title || ''}
              isOptionEqualToValue={(option, value) => option?.book_id === value?.book_id}
              loading={booksLoading}
              loadingText="Loading books..."
              noOptionsText="No books found"
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Select a Book"
                  variant="outlined"
                  placeholder="Search for a book title..."
                  InputProps={{
                    ...params.InputProps,
                    endAdornment: (
                      <>
                        {booksLoading ? <CircularProgress color="inherit" size={20} /> : null}
                        {params.InputProps.endAdornment}
                      </>
                    ),
                  }}
                />
              )}
              renderOption={(props, option) => (
                <li {...props}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <div style={{ marginRight: '10px' }}>
                      {option.image_url ? (
                        <img
                          src={option.image_url}
                          alt={option.title}
                          style={{ width: '40px', height: '60px', objectFit: 'cover' }}
                        />
                      ) : (
                        <div
                          style={{
                            width: '40px',
                            height: '60px',
                            backgroundColor: '#f0f0f0',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                          }}
                        >
                          <span>No Cover</span>
                        </div>
                      )}
                    </div>
                    <div>
                      <Typography variant="body1">{option.title}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        {option.authors}
                      </Typography>
                    </div>
                  </div>
                </li>
              )}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth variant="outlined">
              <InputLabel id="recommendation-count-label">Number of Recommendations</InputLabel>
              <Select
                labelId="recommendation-count-label"
                id="recommendation-count"
                value={recommendationCount}
                onChange={handleRecommendationCountChange}
                label="Number of Recommendations"
              >
                {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((count) => (
                  <MenuItem key={count} value={count}>
                    {count}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <Button
              variant="contained"
              color="primary"
              onClick={() => {
                if (selectedBook && selectedBook.book_id) {
                  console.log('Fetching recommendations for book ID:', selectedBook.book_id);
                  fetchSimilarBooks(selectedBook.book_id);
                } else {
                  console.error('No book selected or book_id is missing');
                  setError('Please select a valid book first');
                }
              }}
              disabled={!selectedBook}
              startIcon={<SearchIcon />}
              sx={{
                borderRadius: '8px',
                textTransform: 'none',
                px: 3,
                py: 1
              }}
            >
              Get Recommendations
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Selected Book Section */}
      {selectedBook && (
        <Paper 
          elevation={3} 
          sx={{ 
            p: 3, 
            mb: 4, 
            borderRadius: '12px',
            background: theme.palette.mode === 'light' 
              ? '#fff' 
              : 'rgba(40, 40, 40, 0.95)',
          }}
        >
          <Grid container spacing={3}>
            <Grid item xs={12} md={3} sx={{ display: 'flex', justifyContent: 'center' }}>
              <Box
                component="img"
                src={selectedBook.image_url || 'https://via.placeholder.com/200x300?text=No+Cover'}
                alt={selectedBook.title}
                sx={{
                  width: 'auto',
                  height: { xs: 220, md: 280 },
                  objectFit: 'contain',
                  borderRadius: 2,
                  boxShadow: 4
                }}
              />
            </Grid>
            <Grid item xs={12} md={9}>
              <Typography variant="h4" component="h2" gutterBottom>
                {selectedBook.title}
              </Typography>
              <Typography variant="h6" component="h3" gutterBottom color="text.secondary">
                by {selectedBook.authors}
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Rating 
                  value={parseFloat(selectedBook.average_rating) || 0} 
                  precision={0.1} 
                  readOnly 
                  sx={{ mr: 1 }}
                />
                <Typography variant="body1">
                  {selectedBook.average_rating?.toFixed(1) || 'N/A'}
                </Typography>
                {selectedBook.ratings_count && (
                  <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                    ({selectedBook.ratings_count.toLocaleString()} ratings)
                  </Typography>
                )}
              </Box>
              
              {selectedBook.description && (
                <>
                  <Typography variant="body1" paragraph>
                    {selectedBook.description}
                  </Typography>
                </>
              )}
            </Grid>
          </Grid>
        </Paper>
      )}
      
      {/* Similar Books Results */}
      <Paper
        elevation={0}
        sx={{
          p: 3,
          borderRadius: 2,
          border: (theme) => `1px solid ${theme.palette.divider}`,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <LocalLibraryIcon sx={{ fontSize: 24, mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
            Similar Books
          </Typography>
        </Box>

        {loading ? (
          <Box sx={{ py: 4, textAlign: 'center' }}>
            <LogoLoading size="large" message="Fetching similar books..." />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ mb: 3 }}>
            <AlertTitle>Error</AlertTitle>
            {error}
          </Alert>
        ) : noResults ? (
          <Alert severity="info" sx={{ mb: 3 }}>
            <AlertTitle>No Results</AlertTitle>
            No similar books found. Try selecting a different book.
          </Alert>
        ) : !selectedBook ? (
          <Alert severity="info" sx={{ mb: 3 }}>
            <AlertTitle>Select a Book</AlertTitle>
            Please select a book to see similar recommendations.
          </Alert>
        ) : (
          <Grid container spacing={3}>
            {similarBooks.map((book, index) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
                <BookCard book={book} onGetSimilar={handleGetSimilar} />
              </Grid>
            ))}
          </Grid>
        )}
      </Paper>
    </Container>
  );
};

export default SimilarBooks;
