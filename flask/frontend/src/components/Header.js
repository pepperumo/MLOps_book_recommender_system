import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar,
  Box,
  Toolbar,
  Typography,
  Switch,
  FormControlLabel,
  Button,
  Stack,
  Container,
  useScrollTrigger,
  Slide
} from '@mui/material';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import DashboardIcon from '@mui/icons-material/Dashboard';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import PersonIcon from '@mui/icons-material/Person';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import { useThemeMode } from './ThemeModeContext';

// Hide AppBar on scroll down
function HideOnScroll(props) {
  const { children } = props;
  const trigger = useScrollTrigger();

  return (
    <Slide appear={false} direction="down" in={!trigger}>
      {children}
    </Slide>
  );
}

const Header = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { darkMode, toggleDarkMode } = useThemeMode();
  
  const isActive = (path) => {
    return location.pathname === path || (path === '/dashboard' && location.pathname === '/');
  };
  
  return (
    <HideOnScroll>
      <AppBar
        position="fixed"
        elevation={0}
        sx={{
          width: '100%',
          zIndex: (theme) => theme.zIndex.drawer + 1,
          borderBottom: (theme) => `1px solid ${theme.palette.divider}`,
          backdropFilter: 'blur(20px)',
          backgroundColor: (theme) => 
            theme.palette.mode === 'light' 
              ? 'rgba(255, 255, 255, 0.8)'
              : 'rgba(0, 0, 0, 0.8)',
        }}
      >
        <Container maxWidth="xl">
          <Toolbar disableGutters sx={{ minHeight: 64 }}>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                mr: 3
              }}
            >
              <MenuBookIcon sx={{ mr: 1, color: 'primary.main', fontSize: 28 }} />
              <Typography
                variant="h6"
                noWrap
                component="div"
                sx={{ 
                  cursor: 'pointer',
                  fontWeight: 700,
                  letterSpacing: '0.5px',
                  background: (theme) => 
                    theme.palette.mode === 'light'
                      ? 'linear-gradient(45deg, #1976d2 30%, #42a5f5 90%)'
                      : 'linear-gradient(45deg, #42a5f5 30%, #1976d2 90%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  fontSize: { xs: '1.1rem', md: '1.3rem' },
                }}
                onClick={() => navigate('/')}
              >
                Book Recommender
              </Typography>
            </Box>
            
            {/* Navigation Links */}
            <Stack 
              direction="row" 
              spacing={1} 
              sx={{ 
                flexGrow: 1,
                display: { xs: 'none', md: 'flex' }
              }}
            >
              <Button 
                color="inherit" 
                startIcon={<DashboardIcon />}
                onClick={() => navigate('/dashboard')}
                sx={{ 
                  px: 2,
                  borderRadius: '8px',
                  fontWeight: 500,
                  textTransform: 'none',
                  fontSize: '0.95rem',
                  bgcolor: isActive('/dashboard') ? 'rgba(25, 118, 210, 0.08)' : 'transparent',
                  color: isActive('/dashboard') ? 'primary.main' : 'text.primary',
                  '&:hover': { 
                    bgcolor: 'rgba(25, 118, 210, 0.12)',
                    color: 'primary.main'
                  },
                  transition: 'all 0.2s ease-in-out'
                }}
              >
                Dashboard
              </Button>
              
              <Button 
                color="inherit" 
                startIcon={<PersonIcon />}
                onClick={() => navigate('/user-recommendations')}
                sx={{ 
                  px: 2,
                  borderRadius: '8px',
                  fontWeight: 500,
                  textTransform: 'none',
                  fontSize: '0.95rem',
                  bgcolor: isActive('/user-recommendations') ? 'rgba(25, 118, 210, 0.08)' : 'transparent',
                  color: isActive('/user-recommendations') ? 'primary.main' : 'text.primary',
                  '&:hover': { 
                    bgcolor: 'rgba(25, 118, 210, 0.12)',
                    color: 'primary.main'
                  },
                  transition: 'all 0.2s ease-in-out'
                }}
              >
                User Recommendations
              </Button>
              
              <Button 
                color="inherit" 
                startIcon={<CompareArrowsIcon />}
                onClick={() => navigate('/similar-books')}
                sx={{ 
                  px: 2,
                  borderRadius: '8px',
                  fontWeight: 500,
                  textTransform: 'none',
                  fontSize: '0.95rem',
                  bgcolor: isActive('/similar-books') ? 'rgba(25, 118, 210, 0.08)' : 'transparent',
                  color: isActive('/similar-books') ? 'primary.main' : 'text.primary',
                  '&:hover': { 
                    bgcolor: 'rgba(25, 118, 210, 0.12)',
                    color: 'primary.main'
                  },
                  transition: 'all 0.2s ease-in-out'
                }}
              >
                Similar Books
              </Button>
            </Stack>
            
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={darkMode}
                    onChange={toggleDarkMode}
                    color="primary"
                    size="small"
                  />
                }
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    {darkMode ? 
                      <DarkModeIcon fontSize="small" color="primary" /> : 
                      <LightModeIcon fontSize="small" color="primary" />
                    }
                  </Box>
                }
              />
            </Box>
          </Toolbar>
        </Container>
      </AppBar>
    </HideOnScroll>
  );
};

export default Header;
