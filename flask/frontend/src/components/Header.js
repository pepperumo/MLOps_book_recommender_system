import React, { useState } from 'react';
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
  Slide,
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import PersonIcon from '@mui/icons-material/Person';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import MenuIcon from '@mui/icons-material/Menu';
import { useThemeMode } from './ThemeModeContext';

// Hide AppBar on scroll down
function HideOnScroll(props) {
  const { children } = props;
  const trigger = useScrollTrigger({
    disableHysteresis: true,
    threshold: 0
  });

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
  const [anchorEl, setAnchorEl] = useState(null);
  
  const isActive = (path) => {
    return location.pathname === path || (path === '/dashboard' && location.pathname === '/');
  };
  
  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };
  
  const handleMenuClose = () => {
    setAnchorEl(null);
  };
  
  const handleNavigation = (path) => {
    navigate(path);
    handleMenuClose();
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
            {/* Mobile Menu Icon */}
            <IconButton
              size="large"
              edge="start"
              aria-label="menu"
              sx={{ 
                mr: 2, 
                display: { xs: 'flex', md: 'none' },
                color: (theme) => theme.palette.mode === 'light' ? 'primary.main' : 'white'
              }}
              onClick={handleMenuOpen}
            >
              <MenuIcon />
            </IconButton>
            
            {/* Mobile Menu */}
            <Menu
              anchorEl={anchorEl}
              open={Boolean(anchorEl)}
              onClose={handleMenuClose}
              sx={{ 
                display: { xs: 'block', md: 'none' },
                mt: 1
              }}
              PaperProps={{
                elevation: 3,
                sx: {
                  border: (theme) => `1px solid ${theme.palette.divider}`,
                  width: 250
                }
              }}
            >
              <MenuItem 
                onClick={() => handleNavigation('/dashboard')}
                sx={{
                  bgcolor: isActive('/dashboard') ? 'rgba(25, 118, 210, 0.12)' : 'transparent',
                  color: isActive('/dashboard') ? 'primary.main' : 'text.primary',
                  py: 1.5,
                  '&:hover': { bgcolor: 'rgba(25, 118, 210, 0.08)' }
                }}
              >
                <ListItemIcon>
                  <DashboardIcon fontSize="small" color={isActive('/dashboard') ? 'primary' : 'inherit'} />
                </ListItemIcon>
                <ListItemText primary="Dashboard" />
              </MenuItem>
              
              <MenuItem 
                onClick={() => handleNavigation('/user-recommendations')}
                sx={{
                  bgcolor: isActive('/user-recommendations') ? 'rgba(25, 118, 210, 0.12)' : 'transparent',
                  color: isActive('/user-recommendations') ? 'primary.main' : 'text.primary',
                  py: 1.5,
                  '&:hover': { bgcolor: 'rgba(25, 118, 210, 0.08)' }
                }}
              >
                <ListItemIcon>
                  <PersonIcon fontSize="small" color={isActive('/user-recommendations') ? 'primary' : 'inherit'} />
                </ListItemIcon>
                <ListItemText primary="User Recommendations" />
              </MenuItem>
              
              <MenuItem 
                onClick={() => handleNavigation('/similar-books')}
                sx={{
                  bgcolor: isActive('/similar-books') ? 'rgba(25, 118, 210, 0.12)' : 'transparent', 
                  color: isActive('/similar-books') ? 'primary.main' : 'text.primary',
                  py: 1.5,
                  '&:hover': { bgcolor: 'rgba(25, 118, 210, 0.08)' }
                }}
              >
                <ListItemIcon>
                  <CompareArrowsIcon fontSize="small" color={isActive('/similar-books') ? 'primary' : 'inherit'} />
                </ListItemIcon>
                <ListItemText primary="Similar Books" />
              </MenuItem>
            </Menu>
            
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                mr: 3
              }}
            >
              <Box
                component="img"
                src="/logo192.png"
                alt="Book Recommender Logo"
                sx={{ 
                  height: 40, 
                  width: 40, 
                  mr: 1 
                }}
              />
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
