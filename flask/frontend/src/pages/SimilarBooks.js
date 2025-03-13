import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardMedia,
  CardActions,
  Button,
  CircularProgress,
  Paper,
  Alert,
  AlertTitle,
  Chip,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  FormHelperText,
  Container,
  Skeleton,
  CardActionArea,
  Stack,
  Autocomplete,
  useTheme
} from '@mui/material';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import SearchIcon from '@mui/icons-material/Search';
import BookIcon from '@mui/icons-material/Book';
import LocalLibraryIcon from '@mui/icons-material/LocalLibrary';
import InfoIcon from '@mui/icons-material/Info';

const SimilarBooks = () => {
  const location = useLocation();
  const theme = useTheme();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [booksLoading, setBooksLoading] = useState(true);
  
  // Data states
  const [allBooks, setAllBooks] = useState([]);
  const [genres, setGenres] = useState([]);
  const [authors, setAuthors] = useState([]);
  const [similarBooks, setSimilarBooks] = useState([]);
  
  // Selection states
  const [selectedBook, setSelectedBook] = useState(null);
  const [selectedGenre, setSelectedGenre] = useState('');
  const [selectedAuthor, setSelectedAuthor] = useState('');
  const [recommendationCount, setRecommendationCount] = useState(5);
  const [noResults, setNoResults] = useState(false);
  
  // Fetch all books, genres, and authors on component mount
  useEffect(() => {
    fetchAllBooks();
    fetchGenres();
    fetchAuthors();
    
    // Check for book_id in URL parameters
    const params = new URLSearchParams(location.search);
    const bookIdFromUrl = params.get('book_id');
    
    if (bookIdFromUrl) {
      // If book_id is in the URL, fetch similar books for this book
      fetchBookById(bookIdFromUrl);
    }
  }, [location]);
  
  // Fetch book by ID
  const fetchBookById = async (bookId) => {
    setBooksLoading(true);
    try {
      // Use the books endpoint with book_id parameter
      const response = await fetch(`http://localhost:5000/api/books?book_id=${bookId}`);
      if (!response.ok) {
        throw new Error(`Error fetching book: ${response.statusText}`);
      }
      const data = await response.json();
      
      // Check if we got any books back
      if (data.books && data.books.length > 0) {
        // Set the first book as selected
        setSelectedBook(data.books[0]);
        // Fetch similar books for this book
        fetchSimilarBooks(bookId);
      } else {
        throw new Error(`Book with ID ${bookId} not found`);
      }
    } catch (err) {
      console.error('Error fetching book by ID:', err);
      setError(`Failed to load book details: ${err.message}`);
    } finally {
      setBooksLoading(false);
    }
  };

  // Fetch all books
  const fetchAllBooks = async () => {
    setBooksLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/books?limit=100');
      if (!response.ok) {
        throw new Error(`Error fetching books: ${response.statusText}`);
      }
      const data = await response.json();
      setAllBooks(data.books || []);
    } catch (err) {
      console.error('Error fetching books:', err);
      setError('Failed to load books. Please try again later.');
    } finally {
      setBooksLoading(false);
    }
  };
  
  // Fetch all genres
  const fetchGenres = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/genres');
      if (!response.ok) {
        throw new Error(`Error fetching genres: ${response.statusText}`);
      }
      const data = await response.json();
      setGenres(data.genres || []);
    } catch (err) {
      console.error('Error fetching genres:', err);
    }
  };
  
  // Fetch all authors
  const fetchAuthors = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/authors');
      if (!response.ok) {
        throw new Error(`Error fetching authors: ${response.statusText}`);
      }
      const data = await response.json();
      setAuthors(data.authors || []);
    } catch (err) {
      console.error('Error fetching authors:', err);
    }
  };
  
  // Fetch books by genre and/or author
  const fetchBooksByFilter = async () => {
    setBooksLoading(true);
    setNoResults(false);
    
    let url = 'http://localhost:5000/api/books?limit=100';
    if (selectedGenre) {
      url += `&genre=${encodeURIComponent(selectedGenre)}`;
    }
    if (selectedAuthor) {
      url += `&author=${encodeURIComponent(selectedAuthor)}`;
    }
    
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Error fetching filtered books: ${response.statusText}`);
      }
      const data = await response.json();
      const books = data.books || [];
      setAllBooks(books);
      
      if (books.length === 0) {
        setNoResults(true);
        setSelectedBook(null);
      } else if (selectedBook === null && books.length > 0) {
        // Select the first book if none is selected
        setSelectedBook(books[0]);
        fetchSimilarBooks(books[0].book_id);
      }
    } catch (err) {
      console.error('Error fetching filtered books:', err);
      setError('Failed to load books with the selected filters.');
    } finally {
      setBooksLoading(false);
    }
  };
  
  // Fetch similar books
  const fetchSimilarBooks = async (bookId) => {
    if (!bookId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Call the FastAPI endpoint directly
      const response = await fetch(`http://localhost:9998/similar-books/${bookId}?num_recommendations=${recommendationCount}&include_images=true`);
      
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error(`No similar books found for this book`);
        } else {
          throw new Error(`Error fetching similar books: ${response.statusText}`);
        }
      }
      
      const data = await response.json();
      setSimilarBooks(data.recommendations || []);
      
      if (!data.recommendations || data.recommendations.length === 0) {
        setNoResults(true);
      } else {
        setNoResults(false);
      }
    } catch (err) {
      console.error('Error fetching similar books:', err);
      setError(err.message);
      setSimilarBooks([]);
      setNoResults(true);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle book selection change
  const handleBookChange = (event, newValue) => {
    setSelectedBook(newValue);
    if (newValue) {
      fetchSimilarBooks(newValue.book_id);
    } else {
      setSimilarBooks([]);
    }
  };
  
  // Handle genre selection change
  const handleGenreChange = (event) => {
    setSelectedGenre(event.target.value);
  };
  
  // Handle author selection change
  const handleAuthorChange = (event) => {
    setSelectedAuthor(event.target.value);
  };
  
  // Handle recommendation count change
  const handleRecommendationCountChange = (event) => {
    const value = parseInt(event.target.value, 10);
    if (!isNaN(value) && value > 0 && value <= 20) {
      setRecommendationCount(value);
      if (selectedBook) {
        fetchSimilarBooks(selectedBook.book_id);
      }
    }
  };
  
  // Apply genre and author filters
  const applyFilters = () => {
    fetchBooksByFilter();
  };
  
  // Reset all filters
  const resetFilters = () => {
    setSelectedGenre('');
    setSelectedAuthor('');
    fetchAllBooks();
  };
  
  // Get a random pastel color based on index
  const getRandomPastelColor = (index) => {
    const colors = [
      '#bbdefb', // light blue
      '#c8e6c9', // light green
      '#d1c4e9', // light purple
      '#ffecb3', // light yellow
      '#ffccbc', // light red
      '#cfd8dc', // light gray
    ];
    return colors[index % colors.length];
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4, mt: 8 }}>
      <Box sx={{ mb: 4 }}>
        <Typography
          variant="h4"
          component="h1"
          gutterBottom
          sx={{
            fontWeight: 700,
            position: 'relative',
            '&:after': {
              content: '""',
              position: 'absolute',
              bottom: -8,
              left: 0,
              width: 60,
              height: 4,
              backgroundColor: 'primary.main',
              borderRadius: 2
            }
          }}
        >
          Similar Books
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" sx={{ mb: 4 }}>
          Find books similar to ones you already love based on our collaborative filtering algorithm
        </Typography>
      </Box>

      {/* Book Selection Section */}
      <Paper
        elevation={0}
        sx={{
          p: 3,
          mb: 4,
          borderRadius: 2,
          border: (theme) => `1px solid ${theme.palette.divider}`
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <BookIcon sx={{ fontSize: 24, mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
            Select a Book
          </Typography>
        </Box>

        <Grid container spacing={3}>
          {/* Filters */}
          <Grid item xs={12} md={8}>
            <Autocomplete
              id="book-select"
              options={allBooks}
              getOptionLabel={(option) => `${option.title} by ${option.authors}`}
              value={selectedBook}
              onChange={handleBookChange}
              loading={booksLoading}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Search for a book"
                  variant="outlined"
                  fullWidth
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
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Box
                      component="img"
                      sx={{
                        width: 40,
                        height: 60,
                        objectFit: 'contain',
                        mr: 2,
                        borderRadius: 1,
                      }}
                      src={option.image_url || "https://islandpress.org/sites/default/files/default_book_cover_2015.jpg"}
                      alt={option.title}
                    />
                    <Box>
                      <Typography variant="body1" sx={{ fontWeight: 500 }}>
                        {option.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        by {option.authors}
                      </Typography>
                    </Box>
                  </Box>
                </li>
              )}
            />
          </Grid>
          
          <Grid item xs={12} md={4}>
            <TextField
              label="Number of Recommendations"
              type="number"
              value={recommendationCount}
              onChange={handleRecommendationCountChange}
              fullWidth
              variant="outlined"
              InputProps={{ inputProps: { min: 1, max: 20 } }}
              helperText="Between 1 and 20"
            />
          </Grid>
        </Grid>

        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
            Filter Books By:
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={5}>
              <FormControl fullWidth variant="outlined" size="small">
                <InputLabel>Genre</InputLabel>
                <Select
                  value={selectedGenre}
                  onChange={handleGenreChange}
                  label="Genre"
                >
                  <MenuItem value="">
                    <em>All Genres</em>
                  </MenuItem>
                  {genres.map((genre) => (
                    <MenuItem key={genre} value={genre}>
                      {genre}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={5}>
              <FormControl fullWidth variant="outlined" size="small">
                <InputLabel>Author</InputLabel>
                <Select
                  value={selectedAuthor}
                  onChange={handleAuthorChange}
                  label="Author"
                >
                  <MenuItem value="">
                    <em>All Authors</em>
                  </MenuItem>
                  {authors.map((author) => (
                    <MenuItem key={author} value={author}>
                      {author}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={2}>
              <Stack direction="row" spacing={1} sx={{ height: '100%' }}>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={applyFilters}
                  startIcon={<SearchIcon />}
                  sx={{
                    height: '100%',
                    borderRadius: '8px',
                    textTransform: 'none'
                  }}
                >
                  Apply
                </Button>
                <Button
                  variant="text"
                  size="small"
                  onClick={resetFilters}
                  sx={{
                    height: '100%',
                    borderRadius: '8px',
                    textTransform: 'none'
                  }}
                >
                  Reset
                </Button>
              </Stack>
            </Grid>
          </Grid>
        </Box>
      </Paper>

      {/* Selected Book Info */}
      {selectedBook && (
        <Paper
          elevation={0}
          sx={{
            p: 3,
            mb: 4,
            borderRadius: 2,
            border: (theme) => `1px solid ${theme.palette.divider}`
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <InfoIcon sx={{ fontSize: 24, mr: 1, color: 'primary.main' }} />
            <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
              Selected Book
            </Typography>
          </Box>
          
          <Grid container spacing={3}>
            <Grid item xs={12} sm={3}>
              <Box
                component="img"
                sx={{
                  width: '100%',
                  maxHeight: 300,
                  objectFit: 'contain',
                  borderRadius: 2,
                  boxShadow: (theme) => theme.shadows[4],
                  backgroundColor: theme.palette.mode === 'light' ? 'rgba(245, 245, 245, 0.8)' : 'rgba(30, 30, 30, 0.8)'
                }}
                src={selectedBook.image_url || "https://islandpress.org/sites/default/files/default_book_cover_2015.jpg"}
                alt={selectedBook.title}
              />
            </Grid>
            <Grid item xs={12} sm={9}>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
                {selectedBook.title}
              </Typography>
              <Typography variant="subtitle1" gutterBottom>
                by {selectedBook.authors}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Average Rating: {selectedBook.average_rating || "N/A"}
                </Typography>
              </Box>
              {selectedBook.description && (
                <Typography variant="body2" color="text.secondary" paragraph>
                  {selectedBook.description.substring(0, 300)}
                  {selectedBook.description.length > 300 ? '...' : ''}
                </Typography>
              )}
              {selectedBook.genres && (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {selectedBook.genres.split('|').map((genre, index) => (
                    <Chip 
                      key={index} 
                      label={genre.trim()} 
                      size="small" 
                      sx={{ 
                        backgroundColor: theme.palette.mode === 'light' 
                          ? getRandomPastelColor(index)
                          : 'rgba(255, 255, 255, 0.08)',
                        color: theme.palette.mode === 'light' 
                          ? 'rgba(0, 0, 0, 0.7)'
                          : 'rgba(255, 255, 255, 0.8)',
                      }}
                    />
                  ))}
                </Box>
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
          border: (theme) => `1px solid ${theme.palette.divider}`
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <LocalLibraryIcon sx={{ fontSize: 24, mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
            Similar Books
          </Typography>
        </Box>
        
        {loading ? (
          <Grid container spacing={3}>
            {[...Array(recommendationCount)].map((_, index) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
                <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <Skeleton variant="rectangular" height={200} />
                  <CardContent>
                    <Skeleton variant="text" width="80%" height={30} />
                    <Skeleton variant="text" width="40%" />
                    <Skeleton variant="text" width="60%" />
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
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
            {similarBooks.map((book) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={book.book_id}>
                <Card 
                  elevation={0}
                  sx={{ 
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    borderRadius: 2,
                    border: (theme) => `1px solid ${theme.palette.divider}`,
                    transition: 'transform 0.3s, box-shadow 0.3s',
                    '&:hover': {
                      transform: 'translateY(-8px)',
                      boxShadow: (theme) => theme.shadows[4],
                    }
                  }}
                >
                  <CardMedia
                    component="img"
                    height="200"
                    image={book.image_url || "https://islandpress.org/sites/default/files/default_book_cover_2015.jpg"}
                    alt={book.title}
                    sx={{ 
                      objectFit: 'contain', 
                      p: 2,
                      backgroundColor: theme.palette.mode === 'light' ? 'rgba(245, 245, 245, 0.8)' : 'rgba(30, 30, 30, 0.8)'
                    }}
                  />
                  <CardContent sx={{ flexGrow: 1, pb: 1 }}>
                    <Typography 
                      gutterBottom 
                      variant="h6" 
                      component="div" 
                      sx={{ 
                        fontWeight: 600,
                        fontSize: '1rem',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        height: '3rem'
                      }}
                    >
                      {book.title}
                    </Typography>
                    <Typography 
                      variant="body2" 
                      color="text.secondary"
                      sx={{
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        height: '2.5rem',
                        mb: 1
                      }}
                    >
                      by {book.authors}
                    </Typography>
                  </CardContent>
                  <Divider />
                  <CardActions sx={{ p: 2, pt: 1, pb: 1.5 }}>
                    <Button 
                      size="small" 
                      variant="outlined" 
                      fullWidth
                      startIcon={<SearchIcon />}
                      onClick={() => navigate(`/similar-books?book_id=${book.book_id}`)}
                      sx={{
                        borderRadius: '8px',
                        textTransform: 'none',
                        fontWeight: 500
                      }}
                    >
                      Find Similar
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </Paper>
    </Container>
  );
};

export default SimilarBooks;
