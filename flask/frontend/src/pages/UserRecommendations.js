import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Container,
  Grid,
  Card,
  CardContent,
  CardMedia,
  CardActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  CircularProgress,
  Rating,
  Divider,
  Chip,
  Alert,
  AlertTitle,
  Paper,
  Skeleton,
  useTheme,
  Stack
} from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import BookIcon from '@mui/icons-material/Book';
import TuneIcon from '@mui/icons-material/Tune';
import SearchIcon from '@mui/icons-material/Search';

const UserRecommendations = () => {
  const theme = useTheme();
  
  // State variables
  const [userId, setUserId] = useState('');
  const [userIds, setUserIds] = useState([]);
  const [recommendationCount, setRecommendationCount] = useState(5);
  const [recommendations, setRecommendations] = useState([]);
  const [userDetails, setUserDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fetchingUsers, setFetchingUsers] = useState(true);
  
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
      setUserIds(data.user_ids || []);
      
      // If there are user IDs available, default to the first one
      if (data.user_ids && data.user_ids.length > 0) {
        setUserId(data.user_ids[0]);
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
      const data = await response.json();
      setUserDetails(data);
    } catch (err) {
      console.error('Error fetching user details:', err);
      setUserDetails(null);
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
      
      // Use FastAPI endpoint to get recommendations
      const response = await fetch(
        `http://localhost:9998/recommend/user/${userId}?num_recommendations=${recommendationCount}&include_images=true`
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
    if (!isNaN(value) && value > 0 && value <= 20) {
      setRecommendationCount(value);
    }
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
          User Recommendations
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" sx={{ mb: 4 }}>
          Get personalized book recommendations based on user reading history and preferences
        </Typography>
      </Box>

      {/* User Selection Section */}
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
          <PersonIcon sx={{ fontSize: 24, mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
            Select a User
          </Typography>
        </Box>

        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <FormControl fullWidth variant="outlined">
              <InputLabel id="user-select-label">User ID</InputLabel>
              <Select
                labelId="user-select-label"
                id="user-select"
                value={userId}
                label="User ID"
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
                <CircularProgress
                  size={24}
                  sx={{
                    position: 'absolute',
                    top: '50%',
                    right: 14,
                    marginTop: '-12px',
                  }}
                />
              )}
            </FormControl>
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
              py: 1
            }}
          >
            Get Recommendations
          </Button>
        </Box>
      </Paper>

      {/* User Details Section */}
      {userDetails && (
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
            <TuneIcon sx={{ fontSize: 24, mr: 1, color: 'primary.main' }} />
            <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
              User Profile
            </Typography>
          </Box>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card 
                elevation={0}
                sx={{ 
                  border: (theme) => `1px solid ${theme.palette.divider}`,
                  height: '100%',
                  borderRadius: 2
                }}
              >
                <CardContent>
                  <Typography variant="h6" component="div" gutterBottom sx={{ fontWeight: 600 }}>
                    Reading History
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    This user has rated {userDetails.books_rated || 0} books with an average rating of {userDetails.average_rating || "N/A"}.
                  </Typography>
                  {userDetails.favorite_genres && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                        Favorite Genres:
                      </Typography>
                      <Stack direction="row" spacing={1} flexWrap="wrap" gap={1}>
                        {userDetails.favorite_genres.map((genre, index) => (
                          <Chip
                            key={index}
                            label={genre}
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
                      </Stack>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card 
                elevation={0}
                sx={{ 
                  border: (theme) => `1px solid ${theme.palette.divider}`,
                  height: '100%',
                  borderRadius: 2
                }}
              >
                <CardContent>
                  <Typography variant="h6" component="div" gutterBottom sx={{ fontWeight: 600 }}>
                    Most Recent Ratings
                  </Typography>
                  {userDetails.recent_ratings ? (
                    <Box>
                      {userDetails.recent_ratings.map((book, index) => (
                        <Box key={index} sx={{ mb: 2, pb: 2, borderBottom: index < userDetails.recent_ratings.length - 1 ? '1px solid' : 'none', borderColor: 'divider' }}>
                          <Box sx={{ display: 'flex', alignItems: 'flex-start' }}>
                            <Box
                              component="img"
                              sx={{
                                width: 40,
                                height: 60,
                                objectFit: 'contain',
                                mr: 2,
                                borderRadius: 1,
                              }}
                              src={book.image_url || "https://islandpress.org/sites/default/files/default_book_cover_2015.jpg"}
                              alt={book.title}
                            />
                            <Box>
                              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                {book.title}
                              </Typography>
                              <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.8rem' }}>
                                by {book.authors}
                              </Typography>
                              <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                                <Rating 
                                  value={book.rating} 
                                  readOnly 
                                  size="small" 
                                  precision={0.5}
                                />
                                <Typography variant="body2" color="text.secondary" sx={{ ml: 1, fontSize: '0.8rem' }}>
                                  {new Date(book.date_rated).toLocaleDateString()}
                                </Typography>
                              </Box>
                            </Box>
                          </Box>
                        </Box>
                      ))}
                    </Box>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      No recent ratings found for this user.
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Paper>
      )}

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
        ) : recommendations.length === 0 ? (
          <Alert severity="info" sx={{ mb: 3 }}>
            <AlertTitle>No Recommendations</AlertTitle>
            {userId ? 'Click "Get Recommendations" to see personalized book suggestions.' : 'Please select a user first.'}
          </Alert>
        ) : (
          <Grid container spacing={3}>
            {recommendations.map((book) => (
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
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Rating 
                        value={parseFloat(book.average_rating || 0)} 
                        precision={0.1} 
                        size="small" 
                        readOnly 
                      />
                      <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                        ({parseFloat(book.predicted_rating || 0).toFixed(2)})
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </Paper>
    </Container>
  );
};

export default UserRecommendations;
