import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Container,
  Grid,
  Button,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Alert,
  AlertTitle,
  Paper,
  useTheme
} from '@mui/material';
import BookIcon from '@mui/icons-material/Book';
import SearchIcon from '@mui/icons-material/Search';
import BookCard from '../components/BookCard';
import LogoLoading from '../components/LogoLoading';
import { useNavigate } from 'react-router-dom';

const UserRecommendations = () => {
  // State variables
  const [userId, setUserId] = useState('');
  const [userIds, setUserIds] = useState([]);
  const [recommendationCount, setRecommendationCount] = useState(5);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fetchingUsers, setFetchingUsers] = useState(true);
  const navigate = useNavigate();
  const theme = useTheme();

  // Fetch user IDs on component mount
  useEffect(() => {
    fetchUserIds();
  }, []);

  // Fetch all available user IDs
  const fetchUserIds = async () => {
    setFetchingUsers(true);
    try {
      const response = await fetch('http://localhost:5000/api/users');
      if (!response.ok) {
        throw new Error(`Error fetching users: ${response.statusText}`);
      }
      const data = await response.json();
      setUserIds(data.users || []);
      
      // If there are user IDs available, default to the first one
      if (data.users && data.users.length > 0) {
        setUserId(data.users[0]);
      }
    } catch (err) {
      console.error('Error fetching user IDs:', err);
      setError('Failed to load user IDs. Please try again later.');
    } finally {
      setFetchingUsers(false);
    }
  };

  // Fetch user details
  const fetchUserDetails = async (id) => {
    try {
      const response = await fetch(`http://localhost:5000/api/users/${id}`);
      if (!response.ok) {
        throw new Error(`Error fetching user details: ${response.statusText}`);
      }
      // We're not using the data yet, but we could store it in state if needed
      await response.json();
    } catch (err) {
      console.error('Error fetching user details:', err);
    }
  };

  // Fetch book recommendations for a user
  const fetchRecommendations = async () => {
    if (!userId) {
      setError('Please select a user ID first');
      return;
    }
    
    setLoading(true);
    setError(null);
    setRecommendations([]);
    
    try {
      // Fetch user details first
      await fetchUserDetails(userId);
      
      // Use Flask API endpoint instead of FastAPI for recommendations
      const response = await fetch(
        `http://localhost:5000/api/recommend/user/${userId}?count=${recommendationCount}&include_images=true`
      );
      
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error(`No recommendations found for user ${userId}`);
        } else {
          throw new Error(`Error fetching recommendations: ${response.statusText}`);
        }
      }
      
      const data = await response.json();
      setRecommendations(data.recommendations || []);
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle user ID selection change
  const handleUserChange = (event) => {
    setUserId(event.target.value);
  };

  // Handle recommendation count change
  const handleRecommendationCountChange = (event) => {
    const value = parseInt(event.target.value, 10);
    if (!isNaN(value) && value > 0 && value <= 12) {
      setRecommendationCount(value);
    }
  };

  // Handle "Get Similar" button click on a book card
  const handleGetSimilar = (book) => {
    console.log('Get similar for book:', book);
    // Navigate to Similar Books page and pass the book ID
    navigate(`/similar-books?book_id=${book.book_id}&fetch=true`);
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4, mt: 8 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center" sx={{ mb: 4 }}>
        Book Recommendation Engine
      </Typography>

      {/* User Selection and Recommendation Count */}
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
            <FormControl fullWidth variant="outlined">
              <InputLabel id="user-select-label">Select a User</InputLabel>
              <Select
                labelId="user-select-label"
                id="user-select"
                value={userId}
                label="Select a User"
                onChange={handleUserChange}
                disabled={fetchingUsers}
              >
                {userIds.map((id) => (
                  <MenuItem key={id} value={id}>
                    User {id}
                  </MenuItem>
                ))}
              </Select>
              {fetchingUsers && (
                <LogoLoading message="Loading user IDs..." />
              )}
            </FormControl>
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
          
          <Grid item xs={12} md={4} sx={{ display: 'flex', alignItems: 'center' }}>
            <Button
              variant="contained"
              color="primary"
              onClick={fetchRecommendations}
              disabled={!userId || fetchingUsers}
              startIcon={<SearchIcon />}
              sx={{
                borderRadius: '8px',
                textTransform: 'none',
                px: 3,
                py: 1,
                height: '56px',
                width: { xs: '100%', md: 'auto' }
              }}
            >
              Get Recommendations
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Recommendations Section */}
      <Paper
        elevation={0}
        sx={{
          p: 3,
          borderRadius: 2,
          border: (theme) => `1px solid ${theme.palette.divider}`
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <BookIcon sx={{ fontSize: 24, mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
            Recommended Books
          </Typography>
        </Box>

        {loading ? (
          <Box sx={{ py: 4 }}>
            <LogoLoading message="Loading recommendations..." />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ mb: 3 }}>
            <AlertTitle>Error</AlertTitle>
            {error}
          </Alert>
        ) : recommendations.length === 0 ? (
          <Alert severity="info" sx={{ mb: 3 }}>
            <AlertTitle>No Recommendations</AlertTitle>
            {userId ? 'Click "Get Recommendations" to see personalized book suggestions.' : 'Please select a user first.'}
          </Alert>
        ) : (
          <Grid container spacing={3}>
            {recommendations.map((book, index) => (
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

export default UserRecommendations;
