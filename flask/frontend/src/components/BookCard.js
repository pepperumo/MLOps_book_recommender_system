import React from 'react';
import { Box, Typography, Button, Rating, Stack } from '@mui/material';
import { useNavigate } from 'react-router-dom';

/**
 * Reusable BookCard component that displays a book in a standard format
 * across the application with consistent styling
 */
const BookCard = ({ book, onGetSimilar }) => {
  const navigate = useNavigate();

  return (
    <Box 
      sx={{ 
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        p: 2,
        width: '100%',
        height: '460px', // Increased height to accommodate larger images
        borderRadius: 2,
        bgcolor: 'background.paper',
        border: '1px solid rgba(0, 0, 0, 0.12)',
        boxShadow: 1,
        transition: 'transform 0.2s',
        '&:hover': {
          transform: 'translateY(-5px)',
        }
      }}
    >
      <Box
        component="img"
        sx={{
          height: 240,
          width: 160,
          objectFit: 'contain',
          mb: 2,
          borderRadius: 1,
        }}
        src={book.image_url || "https://islandpress.org/sites/default/files/default_book_cover_2015.jpg"}
        alt={book.title}
      />
      <Typography 
        sx={{ 
          fontWeight: 600,
          textAlign: 'center',
          mb: 0.5,
          width: '100%',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          height: '48px' // Fixed height for title
        }}
        variant="subtitle1"
      >
        {book.title}
      </Typography>
      <Typography 
        variant="body2" 
        color="text.secondary"
        sx={{ 
          mb: 1, 
          textAlign: 'center',
          width: '100%',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          display: '-webkit-box',
          WebkitLineClamp: 1,
          WebkitBoxOrient: 'vertical',
          height: '20px' // Fixed height for authors
        }}
      >
        by {book.authors}
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, height: '24px' }}> {/* Fixed height for rating */}
        <Rating 
          value={parseFloat(book.average_rating || 0)} 
          precision={0.1} 
          size="small" 
          readOnly 
        />
        <Typography variant="body2" color="text.secondary" sx={{ ml: 0.5 }}>
          ({book.ratings_count ? book.ratings_count.toLocaleString() : '0'})
        </Typography>
      </Box>
      <Stack direction="row" spacing={1} sx={{ mt: 'auto', width: '100%', justifyContent: 'center' }}> {/* This pushes the buttons to the bottom */}
        <Button
          variant="outlined"
          size="small"
          onClick={() => navigate(`/similar-books?book_id=${book.book_id}`)}
          sx={{
            textTransform: 'none',
            borderRadius: '4px',
            px: 2,
          }}
        >
          Details
        </Button>
        <Button
          variant="contained"
          size="small"
          onClick={() => {
            if (onGetSimilar) {
              onGetSimilar(book);
            } else {
              // Navigate to similar books page and trigger recommendation fetch
              navigate(`/similar-books?book_id=${book.book_id}&fetch=true`);
            }
          }}
          sx={{
            bgcolor: 'primary.main',
            color: 'white',
            textTransform: 'none',
            borderRadius: '4px',
            px: 2,
            '&:hover': {
              bgcolor: 'primary.dark',
            }
          }}
        >
          Get Similar
        </Button>
      </Stack>
    </Box>
  );
};

export default BookCard;
