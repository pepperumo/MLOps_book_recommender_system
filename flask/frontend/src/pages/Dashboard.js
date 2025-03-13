import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Button,
  CircularProgress,
  Paper,
  Chip,
  Container,
  Divider,
  Rating,
  CardActionArea,
  CardActions,
  Skeleton,
  useTheme
} from '@mui/material';
import LocalLibraryIcon from '@mui/icons-material/LocalLibrary';
import RecommendIcon from '@mui/icons-material/Recommend';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import BarChartIcon from '@mui/icons-material/BarChart';
import BookIcon from '@mui/icons-material/Book';
import PersonIcon from '@mui/icons-material/Person';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const Dashboard = () => {
  const navigate = useNavigate();
  const theme = useTheme();
  
  // State variables
  const [popularBooks, setPopularBooks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [genreChartData, setGenreChartData] = useState([]);

  useEffect(() => {
    fetchPopularBooks();
  }, []);

  // Fetch popular books from the backend
  const fetchPopularBooks = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/popular-books?limit=6&randomize=true');
      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }
      
      const data = await response.json();
      setPopularBooks(data.books || []);
      
      // Process genre data for chart
      const genreCounts = {};
      data.books.forEach(book => {
        if (book.genres) {
          const genresList = book.genres.split('|');
          genresList.forEach(genre => {
            if (genre.trim()) {
              genreCounts[genre.trim()] = (genreCounts[genre.trim()] || 0) + 1;
            }
          });
        }
      });
      
      // Convert to chart data format and sort by count
      const chartData = Object.entries(genreCounts)
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 5); // Top 5 genres
      
      setGenreChartData(chartData);
    } catch (err) {
      console.error('Failed to fetch popular books:', err);
      setError('Failed to load popular books. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

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
          Dashboard
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" sx={{ mb: 4 }}>
          Discover top books and recommendations based on your preferences
        </Typography>
      </Box>

      {/* Feature Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {/* User Recommendations Card */}
        <Grid item xs={12} md={6}>
          <Paper
            elevation={0}
            sx={{
              p: 3,
              height: '100%',
              borderRadius: 2,
              border: (theme) => `1px solid ${theme.palette.divider}`,
              transition: 'transform 0.3s, box-shadow 0.3s',
              '&:hover': {
                boxShadow: (theme) => theme.shadows[4],
                transform: 'translateY(-4px)'
              }
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <PersonIcon sx={{ fontSize: 24, mr: 1, color: 'primary.main' }} />
              <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
                User Recommendations
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary" paragraph>
              Get personalized book recommendations based on your reading history and preferences.
            </Typography>
            <Button
              variant="contained"
              color="primary"
              endIcon={<RecommendIcon />}
              onClick={() => navigate('/user-recommendations')}
              sx={{
                borderRadius: '8px',
                textTransform: 'none',
                fontWeight: 500,
                boxShadow: 'none',
                '&:hover': {
                  boxShadow: 'none',
                  backgroundColor: 'primary.dark'
                }
              }}
            >
              Get Recommendations
            </Button>
          </Paper>
        </Grid>

        {/* Similar Books Card */}
        <Grid item xs={12} md={6}>
          <Paper
            elevation={0}
            sx={{
              p: 3,
              height: '100%',
              borderRadius: 2,
              border: (theme) => `1px solid ${theme.palette.divider}`,
              transition: 'transform 0.3s, box-shadow 0.3s',
              '&:hover': {
                boxShadow: (theme) => theme.shadows[4],
                transform: 'translateY(-4px)'
              }
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <CompareArrowsIcon sx={{ fontSize: 24, mr: 1, color: 'primary.main' }} />
              <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
                Similar Books
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary" paragraph>
              Find books similar to ones you already love based on our collaborative filtering algorithm.
            </Typography>
            <Button
              variant="contained"
              color="primary"
              endIcon={<CompareArrowsIcon />}
              onClick={() => navigate('/similar-books')}
              sx={{
                borderRadius: '8px',
                textTransform: 'none',
                fontWeight: 500,
                boxShadow: 'none',
                '&:hover': {
                  boxShadow: 'none',
                  backgroundColor: 'primary.dark'
                }
              }}
            >
              Find Similar
            </Button>
          </Paper>
        </Grid>
      </Grid>

      {/* Popular Books Section */}
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
          <LocalLibraryIcon sx={{ fontSize: 24, mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
            Popular Books
          </Typography>
        </Box>

        {loading ? (
          <Grid container spacing={3}>
            {[...Array(6)].map((_, index) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <Skeleton variant="rectangular" height={200} />
                  <CardContent>
                    <Skeleton variant="text" width="80%" height={30} />
                    <Skeleton variant="text" width="40%" />
                    <Skeleton variant="text" width="60%" />
                  </CardContent>
                  <Box sx={{ p: 2, pt: 0 }}>
                    <Skeleton variant="rectangular" height={36} width="100%" />
                  </Box>
                </Card>
              </Grid>
            ))}
          </Grid>
        ) : error ? (
          <Typography color="error">{error}</Typography>
        ) : (
          <Grid container spacing={3}>
            {popularBooks.map((book) => (
              <Grid item xs={12} sm={6} md={4} key={book.book_id}>
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
                  <CardActionArea onClick={() => navigate(`/similar-books?book_id=${book.book_id}`)}>
                    <CardMedia
                      component="img"
                      height="240"
                      image={book.image_url && book.image_url !== "" ? book.image_url : "https://islandpress.org/sites/default/files/default_book_cover_2015.jpg"}
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
                          fontSize: '1.1rem',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                          height: '3.3rem'
                        }}
                      >
                        {book.title}
                      </Typography>
                      <Typography 
                        variant="body2" 
                        color="text.secondary"
                        sx={{ mb: 1 }}
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
                          ({book.average_rating})
                        </Typography>
                      </Box>
                      {book.genres && (
                        <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {book.genres.split('|').slice(0, 2).map((genre, index) => (
                            <Chip 
                              key={index} 
                              label={genre.trim()} 
                              size="small" 
                              sx={{ 
                                fontSize: '0.7rem',
                                height: 24,
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
                    </CardContent>
                  </CardActionArea>
                  <Divider />
                  <CardActions sx={{ p: 2, pt: 1, pb: 1.5 }}>
                    <Button 
                      size="small" 
                      variant="outlined" 
                      fullWidth
                      startIcon={<CompareArrowsIcon />}
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

      {/* Book Categories Chart */}
      <Paper 
        elevation={0}
        sx={{ 
          p: 3,
          borderRadius: 2,
          border: (theme) => `1px solid ${theme.palette.divider}`
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <BarChartIcon sx={{ fontSize: 24, mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
            Book Categories
          </Typography>
        </Box>
        
        {genreChartData.length > 0 ? (
          <Box sx={{ height: 300, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={genreChartData}
                margin={{ top: 5, right: 30, left: 20, bottom: 70 }}
              >
                <XAxis 
                  dataKey="name" 
                  angle={-45} 
                  textAnchor="end" 
                  height={70}
                  tick={{ fontSize: 12 }}
                />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip 
                  formatter={(value) => [`${value} books`, 'Count']}
                  contentStyle={{
                    borderRadius: 8,
                    border: 'none',
                    boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                    backgroundColor: theme.palette.mode === 'dark' ? '#333' : 'white'
                  }}
                />
                <Bar dataKey="count" fill="#1976d2" radius={[4, 4, 0, 0]}>
                  {genreChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getRandomPastelColor(index)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Box>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No genre data available
          </Typography>
        )}
      </Paper>
    </Container>
  );
};

export default Dashboard;
