import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardMedia,
  CardActions,
  Button,
  TextField,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  CircularProgress,
  Pagination,
  Rating,
  Chip,
  Divider,
  Paper,
  InputAdornment
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import FilterListIcon from '@mui/icons-material/FilterList';
import SortIcon from '@mui/icons-material/Sort';
import { useNavigate } from 'react-router-dom';

const BookSearch = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [language, setLanguage] = useState('en');
  const [sortBy, setSortBy] = useState('popularity');
  const [books, setBooks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const booksPerPage = 12;

  // Handle search term change
  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
  };

  // Handle language filter change
  const handleLanguageChange = (event) => {
    setLanguage(event.target.value);
  };

  // Handle sort option change
  const handleSortChange = (event) => {
    setSortBy(event.target.value);
  };

  // Handle page change
  const handlePageChange = (event, value) => {
    setPage(value);
    window.scrollTo(0, 0);
  };

  // Fetch books based on search parameters
  const fetchBooks = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Build query parameters
      const params = new URLSearchParams({
        limit: 100, // Fetch more to handle pagination on client side
        language: language,
        sort: sortBy
      });
      
      // Add search term if provided
      if (searchTerm) {
        params.append('search', searchTerm);
      }
      
      const response = await fetch(`/api/books?${params.toString()}`);
      
      if (!response.ok) {
        throw new Error(`Error fetching books: ${response.statusText}`);
      }
      
      const data = await response.json();
      setBooks(data.books || []);
      setTotalPages(Math.ceil((data.books?.length || 0) / booksPerPage));
    } catch (err) {
      console.error('Error fetching books:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle search submission
  const handleSearch = (event) => {
    event.preventDefault();
    setPage(1); // Reset to first page
    fetchBooks();
  };

  // Get current page of books
  const getCurrentBooks = () => {
    const startIndex = (page - 1) * booksPerPage;
    const endIndex = startIndex + booksPerPage;
    return books.slice(startIndex, endIndex);
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom component="div">
        Book Search
      </Typography>
      
      <Paper 
        component="form" 
        onSubmit={handleSearch}
        elevation={3}
        sx={{ p: 3, mb: 4 }}
      >
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Search Books"
              variant="outlined"
              value={searchTerm}
              onChange={handleSearchChange}
              placeholder="Enter book title or author"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon color="action" />
                  </InputAdornment>
                ),
              }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <FormControl fullWidth variant="outlined">
              <InputLabel id="language-select-label">Language</InputLabel>
              <Select
                labelId="language-select-label"
                id="language-select"
                value={language}
                onChange={handleLanguageChange}
                label="Language"
                startAdornment={
                  <InputAdornment position="start">
                    <FilterListIcon fontSize="small" />
                  </InputAdornment>
                }
              >
                <MenuItem value="en">English</MenuItem>
                <MenuItem value="fr">French</MenuItem>
                <MenuItem value="es">Spanish</MenuItem>
                <MenuItem value="de">German</MenuItem>
                <MenuItem value="it">Italian</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <FormControl fullWidth variant="outlined">
              <InputLabel id="sort-select-label">Sort By</InputLabel>
              <Select
                labelId="sort-select-label"
                id="sort-select"
                value={sortBy}
                onChange={handleSortChange}
                label="Sort By"
                startAdornment={
                  <InputAdornment position="start">
                    <SortIcon fontSize="small" />
                  </InputAdornment>
                }
              >
                <MenuItem value="popularity">Popularity</MenuItem>
                <MenuItem value="rating">Rating</MenuItem>
                <MenuItem value="title">Title</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <Button
              variant="contained"
              color="primary"
              fullWidth
              type="submit"
              sx={{ height: '56px' }}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} /> : 'Search'}
            </Button>
          </Grid>
        </Grid>
      </Paper>
      
      {/* Results Section */}
      {error ? (
        <Typography color="error" variant="body1" sx={{ mt: 2 }}>
          {error}
        </Typography>
      ) : books.length > 0 ? (
        <Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h5">
              Results
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Showing {getCurrentBooks().length} of {books.length} books
            </Typography>
          </Box>
          
          <Grid container spacing={3}>
            {getCurrentBooks().map((book) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={book.book_id}>
                <Card 
                  sx={{ 
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    transition: 'transform 0.3s, box-shadow 0.3s',
                    '&:hover': {
                      transform: 'translateY(-5px)',
                      boxShadow: '0 8px 16px rgba(0,0,0,0.1)',
                    }
                  }}
                >
                  <CardMedia
                    component="img"
                    height="200"
                    image={book.image_url || "https://islandpress.org/sites/default/files/default_book_cover_2015.jpg"}
                    alt={book.title}
                    sx={{ objectFit: 'contain', p: 1 }}
                  />
                  <CardContent sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" component="div" gutterBottom noWrap title={book.title}>
                      {book.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom noWrap>
                      by {book.authors}
                    </Typography>
                    
                    {book.average_rating && (
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Rating 
                          value={parseFloat(book.average_rating)} 
                          precision={0.1} 
                          readOnly 
                          size="small"
                        />
                        <Typography variant="body2" sx={{ ml: 1 }}>
                          ({book.average_rating})
                        </Typography>
                      </Box>
                    )}
                    
                    {book.genres && book.genres.length > 0 && (
                      <Box sx={{ mt: 1 }}>
                        {typeof book.genres === 'string' 
                          ? book.genres.split(',').slice(0, 2).map((genre, index) => (
                              <Chip key={index} label={genre.trim()} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
                            ))
                          : book.genres.slice(0, 2).map((genre, index) => (
                              <Chip key={index} label={genre} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
                            ))
                        }
                      </Box>
                    )}
                  </CardContent>
                  <Divider />
                  <CardActions>
                    <Button 
                      size="small" 
                      color="primary"
                      onClick={() => navigate(`/similar-books?bookId=${book.book_id}`)}
                    >
                      Find Similar
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
          
          {/* Pagination */}
          {totalPages > 1 && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
              <Pagination 
                count={totalPages} 
                page={page} 
                onChange={handlePageChange} 
                color="primary"
                showFirstButton
                showLastButton
              />
            </Box>
          )}
        </Box>
      ) : !loading && (
        <Typography variant="body1" sx={{ mt: 4, textAlign: 'center' }}>
          {searchTerm ? 'No books found matching your search. Try different keywords or filters.' : 'Enter a search term to find books.'}
        </Typography>
      )}
    </Box>
  );
};

export default BookSearch;
