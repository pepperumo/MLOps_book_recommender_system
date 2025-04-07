import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Divider,
  useTheme
} from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import PersonIcon from '@mui/icons-material/Person';
import MenuBookIcon from '@mui/icons-material/MenuBook';

const drawerWidth = 240;

const Sidebar = ({ open, toggleDrawer }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  
  const menuItems = [
    {
      text: 'Dashboard',
      path: '/dashboard',
      icon: <DashboardIcon />
    },
    {
      text: 'User Recommendations',
      path: '/user-recommendations',
      icon: <PersonIcon />
    },
    {
      text: 'Similar Books',
      path: '/similar-books',
      icon: <CompareArrowsIcon />
    }
  ];

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        [`& .MuiDrawer-paper`]: { 
          width: drawerWidth, 
          boxSizing: 'border-box',
          backgroundColor: theme.palette.background.paper,
          borderRight: `1px solid ${theme.palette.divider}`
        },
        display: { xs: 'none', sm: 'block' },
      }}
      open={open}
    >
      <Toolbar>
        <Box sx={{ display: 'flex', alignItems: 'center', pl: 1 }}>
          <MenuBookIcon sx={{ mr: 1, color: 'primary.main' }} />
          <Box component="span" sx={{ fontWeight: 'bold', fontSize: '1.2rem' }}>
            Book Recommender
          </Box>
        </Box>
      </Toolbar>
      <Divider />
      <Box sx={{ overflow: 'auto' }}>
        <List>
          {menuItems.map((item) => (
            <ListItem key={item.text} disablePadding>
              <ListItemButton 
                selected={location.pathname === item.path || (item.path === '/dashboard' && location.pathname === '/')}
                onClick={() => navigate(item.path)}
                sx={{
                  '&.Mui-selected': {
                    backgroundColor: 'primary.main',
                    color: 'white',
                    '& .MuiListItemIcon-root': {
                      color: 'white',
                    },
                    '&:hover': {
                      backgroundColor: 'primary.dark',
                    }
                  },
                }}
              >
                <ListItemIcon sx={{
                  color: location.pathname === item.path || (item.path === '/dashboard' && location.pathname === '/') ? 'white' : 'inherit',
                }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>
    </Drawer>
  );
};

export default Sidebar;
