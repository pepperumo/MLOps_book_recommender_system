import React, { useState, useEffect, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Container,
  Paper,
  Alert,
  AlertTitle,
  useTheme,
  Switch,
  FormControlLabel
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import BookCard from '../components/BookCard';
import LogoLoading from '../components/LogoLoading';
import { checkMcpAvailability, getSimilarBooksViaMcp } from '../services/mcpService';

const SimilarBooks = () => {
  // State variables
  const [selectedBookId, setSelectedBookId] = useState('');
  const [allBooks, setAllBooks] = useState([]);
  const [similarBooks, setSimilarBooks] = useState([]);
  const [recommendationCount, setRecommendationCount] = useState(5);
  const [loading, setLoading] = useState(false);
  const [booksLoading, setBooksLoading] = useState(true);
  const [error, setError] = useState(null);
  const [noResults, setNoResults] = useState(false);
  const [mcpAvailable, setMcpAvailable] = useState(false);
  const [useMcp, setUseMcp] = useState(false);

  const location = useLocation();
  const navigate = useNavigate();
  const theme = useTheme();

  // Parse query parameters from URL
  const queryParams = new URLSearchParams(location.search);
  const bookIdFromQuery = queryParams.get('book_id');
  const shouldFetch = queryParams.get('fetch') === 'true';

  // Check if MCP is available on component mount
  useEffect(() => {
    checkMcpStatus();
  }, []);

  // Check if MCP is available
  const checkMcpStatus = async () => {
    try {
      const available = await checkMcpAvailability();
      setMcpAvailable(available);
      setUseMcp(available); // Default to using MCP if available
    } catch (error) {
      console.error('Error checking MCP status:', error);
      setMcpAvailable(false);
      setUseMcp(false);
    }
  };

  // Handle MCP toggle
  const handleMcpToggle = (event) => {
    setUseMcp(event.target.checked);
  };

  // Fetch similar books function
  const fetchSimilarBooks = useCallback(async (bookId) => {
    if (!bookId) {
      setError('Please select a book first');
      return;
    }

    console.log('Fetching similar books for book ID:', bookId);
    setLoading(true);
    setError(null);
    setSimilarBooks([]);
    setNoResults(false);

    try {
      let books = [];

      if (useMcp && mcpAvailable) {
        // Use MCP for similar books
        console.log('Using MCP for similar books');
        const data = await getSimilarBooksViaMcp(bookId, recommendationCount);
        books = data || [];
      } else {
        // Use standard API endpoint
        console.log('Using standard API for similar books');
        const response = await fetch(
          `http://localhost:5000/api/similar-books/${bookId}?count=${recommendationCount}`
        );

        if (!response.ok) {
          if (response.status === 404) {
            throw new Error(`No similar books found for book ${bookId}`);
          }
          throw new Error(`Error fetching similar books: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        console.log('API response:', data);

        // Handle different response formats
        if (data.recommendations) {
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
  }, [recommendationCount, mcpAvailable, useMcp]);

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
  const handleCountChange = (event) => {
    const value = parseInt(event.target.value, 10);
    if (!isNaN(value) && value > 0 && value <= 12) {
      setRecommendationCount(value);
    }
  };

  // Handle book selection change from dropdown
  const handleBookChange = useCallback((event) => {
    setSelectedBookId(event.target.value);
  }, []);

  // Handle book selection and fetchSimilarBooks
  const handleFindSimilar = () => {
    if (selectedBookId) {
      console.log('Fetching recommendations for book ID:', selectedBookId);
      fetchSimilarBooks(selectedBookId);
    } else {
      console.error('No book selected or book_id is missing');
      setError('Please select a valid book first');
    }
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
      const bookInList = allBooks.find((b) =>
        b.book_id && b.book_id.toString() === bookIdFromUrl.toString()
      );

      if (bookInList) {
        console.log('Found book from URL:', bookInList.title);
        setSelectedBookId(bookInList.book_id);

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
    if (selectedBookId) {
      // Update URL without page reload
      window.history.pushState(
        { bookId: selectedBookId },
        '',
        `/similar-books?book_id=${selectedBookId}`
      );
    }
  }, [selectedBookId]);

  useEffect(() => {
    const handlePopState = () => {
      const params = new URLSearchParams(window.location.search);
      const bookId = params.get('book_id');

      if (bookId) {
        // Check if book is in our allBooks array first
        const bookInList = allBooks.find((b) =>
          b.book_id && b.book_id.toString() === bookId.toString()
        );

        if (bookInList) {
          setSelectedBookId(bookInList.book_id);
        } else {
          setSelectedBookId(null);
          setSimilarBooks([]);
        }
      } else {
        setSelectedBookId(null);
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

      {/* Controls Panel */}
      <Paper
        elevation={3}
        sx={{
          p: 3,
          mb: 4,
          borderRadius: '12px',
          background: theme.palette.mode === 'light'
            ? '#fff'
            : 'rgba(45, 45, 45, 0.98)',
          boxShadow: theme.shadows[4],
        }}
      >
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={5}>
            <FormControl
              fullWidth
              variant="outlined"
              disabled={booksLoading || Boolean(bookIdFromQuery)}
            >
              <InputLabel id="book-select-label">Select a Book</InputLabel>
              <Select
                labelId="book-select-label"
                id="book-select"
                value={selectedBookId}
                onChange={handleBookChange}
                label="Select a Book"
              >
                {(allBooks || []).map((book) => (
                  <MenuItem key={book.book_id} value={book.book_id}>
                    {book.title} - {book.authors?.substring(0, 30) || 'Unknown Author'}
                    {book.authors?.length > 30 ? '...' : ''}
                  </MenuItem>
                ))}
              </Select>
              {booksLoading && <LogoLoading message="Loading books..." />}
            </FormControl>
          </Grid>

          <Grid item xs={12} md={3}>
            <FormControl fullWidth variant="outlined">
              <InputLabel id="count-select-label">Number of Recommendations</InputLabel>
              <Select
                labelId="count-select-label"
                id="count-select"
                value={recommendationCount}
                onChange={handleCountChange}
                label="Number of Recommendations"
              >
                {[3, 5, 8, 10, 12].map((count) => (
                  <MenuItem key={count} value={count}>
                    {count}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          {mcpAvailable && (
            <Grid item xs={12} md={2}>
              <FormControlLabel
                control={
                  <Switch
                    checked={useMcp}
                    onChange={handleMcpToggle}
                    color="primary"
                  />
                }
                label="Use MCP"
              />
            </Grid>
          )}

          <Grid item xs={12} md={2}>
            <Button
              variant="contained"
              color="primary"
              startIcon={<SearchIcon />}
              onClick={handleFindSimilar}
              disabled={!selectedBookId || loading}
              fullWidth
              sx={{ height: '56px' }}
            >
              Find Similar
            </Button>
          </Grid>
        </Grid>
      </Paper>

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
          <SearchIcon sx={{ fontSize: 24, mr: 1, color: 'primary.main' }} />
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
        ) : !selectedBookId ? (
          <Alert severity="info" sx={{ mb: 3 }}>
            <AlertTitle>Select a Book</AlertTitle>
            Please select a book to see similar recommendations.
          </Alert>
        ) : (
          <Grid container spacing={3}>
            {similarBooks.map((book, index) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
                <BookCard book={book} />
              </Grid>
            ))}
          </Grid>
        )}
      </Paper>
    </Container>
  );
};

export default SimilarBooks;
