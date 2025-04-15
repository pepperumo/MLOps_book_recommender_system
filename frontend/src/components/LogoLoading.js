import React from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';

const LogoLoading = ({ size = 'medium', message = 'Loading...' }) => {
  // Size mapping
  const sizeMap = {
    small: { logo: 40, progress: 60, fontSize: 14 },
    medium: { logo: 60, progress: 84, fontSize: 16 },
    large: { logo: 80, progress: 108, fontSize: 18 }
  };
  
  const dimensions = sizeMap[size] || sizeMap.medium;
  
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', py: 3 }}>
      <Box sx={{ position: 'relative', display: 'inline-flex' }}>
        <CircularProgress
          size={dimensions.progress}
          thickness={3}
          sx={{
            position: 'absolute',
            color: 'primary.main',
            animationDuration: '1.5s',
          }}
        />
        <Box
          component="img"
          src="/logo192.png"
          alt="Loading"
          sx={{
            width: dimensions.logo,
            height: dimensions.logo,
            animation: 'pulse 2s infinite',
            '@keyframes pulse': {
              '0%': { opacity: 0.6, transform: 'scale(0.95)' },
              '50%': { opacity: 1, transform: 'scale(1.05)' },
              '100%': { opacity: 0.6, transform: 'scale(0.95)' },
            }
          }}
        />
      </Box>
      {message && (
        <Typography 
          variant="body1" 
          sx={{ 
            mt: 2, 
            fontSize: dimensions.fontSize,
            fontWeight: 500,
            color: 'text.secondary'
          }}
        >
          {message}
        </Typography>
      )}
    </Box>
  );
};

export default LogoLoading;
