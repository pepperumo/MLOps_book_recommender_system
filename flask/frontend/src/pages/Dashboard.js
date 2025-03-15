import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Button,
  Paper,
  Container,
  Alert,
  AlertTitle,
  useTheme
} from '@mui/material';
import RecommendIcon from '@mui/icons-material/Recommend';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import PersonIcon from '@mui/icons-material/Person';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import BookCard from '../components/BookCard';
import Slider from 'react-slick';
import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';

const Dashboard = () => {
  const navigate = useNavigate();
  const theme = useTheme();
  
  // State variables
  const [popularBooks, setPopularBooks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchPopularBooks();
  }, []);

  // Fetch popular books from the backend
  const fetchPopularBooks = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/popular-books?limit=6&randomize=true&include_genres=true', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        mode: 'cors',
        credentials: 'same-origin',
      });
      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }
      
      const data = await response.json();
      setPopularBooks(data.books || []);
    } catch (err) {
      console.error('Failed to fetch popular books:', err);
      setError('Failed to load popular books. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4, mt: 8 }}>
      {/* Hero section with logo and background */}
      <Box 
        sx={{ 
          mb: 6,
          p: 4,
          borderRadius: '12px',
          position: 'relative',
          overflow: 'hidden',
          backgroundImage: 'url(/background.png)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          boxShadow: theme.shadows[4],
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: theme.palette.mode === 'light' 
              ? 'rgba(255, 255, 255, 0.8)' 
              : 'rgba(0, 0, 0, 0.75)',
            zIndex: 1
          }
        }}
      >
        <Grid container spacing={3} sx={{ position: 'relative', zIndex: 2 }}>
          <Grid item xs={12} md={3} sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <Box
              component="img"
              src="/logo512.png"
              alt="Book Recommender Logo"
              sx={{ 
                width: { xs: 150, md: 200 },
                height: { xs: 150, md: 200 },
                display: 'block',
                mb: { xs: 2, md: 0 }
              }}
            />
          </Grid>
          <Grid item xs={12} md={9} sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <Typography
              variant="h3"
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
              Book Recommendation Engine
            </Typography>
            <Typography variant="h6" color="text.secondary" sx={{ mb: 3 }}>
              Discover your next favorite book with our AI-powered recommendation system
            </Typography>
            <Grid container spacing={2}>
              <Grid item>
                <Button 
                  variant="contained" 
                  size="large" 
                  startIcon={<PersonIcon />}
                  onClick={() => navigate('/user-recommendations')}
                  sx={{ borderRadius: '8px' }}
                >
                  Get User Recommendations
                </Button>
              </Grid>
              <Grid item>
                <Button 
                  variant="contained" 
                  size="large" 
                  startIcon={<CompareArrowsIcon />}
                  onClick={() => navigate('/similar-books')}
                  sx={{ borderRadius: '8px' }}
                >
                  Find Similar Books
                </Button>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
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
          borderRadius: 2,
          border: (theme) => `1px solid ${theme.palette.divider}`,
          mt: 3
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <TrendingUpIcon sx={{ fontSize: 24, mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
            Popular Books
          </Typography>
        </Box>

        {loading ? (
          <Grid container spacing={3}>
            {[...Array(6)].map((_, index) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <Box sx={{ 
                  p: 2, 
                  borderRadius: 2,
                  bgcolor: 'background.paper',
                  border: (theme) => `1px solid ${theme.palette.divider}`
                }}>
                  <Box sx={{ height: 200, mb: 2, borderRadius: 2, bgcolor: 'background.paper' }} />
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Loading...
                  </Typography>
                </Box>
              </Grid>
            ))}
          </Grid>
        ) : error ? (
          <Alert severity="error" sx={{ mb: 3 }}>
            <AlertTitle>Error</AlertTitle>
            {error}
          </Alert>
        ) : (
          <Box sx={{ position: 'relative', mx: -1 }}>
            <Slider
              dots={false}
              infinite={true}
              speed={800}
              slidesToShow={4}
              slidesToScroll={1}
              autoplay={true}
              autoplaySpeed={3000}
              cssEase="linear"
              pauseOnHover={true}
              responsive={[
                {
                  breakpoint: 1200,
                  settings: {
                    slidesToShow: 3,
                    slidesToScroll: 1
                  }
                },
                {
                  breakpoint: 900,
                  settings: {
                    slidesToShow: 2,
                    slidesToScroll: 1
                  }
                },
                {
                  breakpoint: 600,
                  settings: {
                    slidesToShow: 1,
                    slidesToScroll: 1
                  }
                }
              ]}
            >
              {popularBooks.map((book) => (
                <Box key={book.book_id} sx={{ px: 1 }}>
                  <BookCard book={book} />
                </Box>
              ))}
            </Slider>
          </Box>
        )}
      </Paper>

      {/* Additional dashboard components can be added here */}
    </Container>
  );
};

export default Dashboard;
