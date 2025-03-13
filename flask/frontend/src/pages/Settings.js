import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  Grid,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
  Snackbar,
  Card,
  CardContent,
  InputAdornment,
  IconButton
} from '@mui/material';
import { useThemeMode } from '../components/ThemeModeContext';
import { useAuth } from '../components/AuthContext';
import SettingsIcon from '@mui/icons-material/Settings';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import SecurityIcon from '@mui/icons-material/Security';
import ApiIcon from '@mui/icons-material/Api';
import Visibility from '@mui/icons-material/Visibility';
import VisibilityOff from '@mui/icons-material/VisibilityOff';
import SaveIcon from '@mui/icons-material/Save';

const Settings = () => {
  const { darkMode, toggleDarkMode } = useThemeMode();
  const { user } = useAuth();
  
  const [apiUrl, setApiUrl] = useState('http://127.0.0.1:5000');
  const [maxRecommendations, setMaxRecommendations] = useState(5);
  const [modelType, setModelType] = useState('collaborative');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'success' });

  // Handle API settings save
  const handleApiSettingsSave = (event) => {
    event.preventDefault();
    // In a real app, you would save these settings to backend or localStorage
    localStorage.setItem('apiUrl', apiUrl);
    localStorage.setItem('maxRecommendations', maxRecommendations);
    localStorage.setItem('modelType', modelType);
    
    setNotification({
      open: true,
      message: 'API settings saved successfully!',
      severity: 'success'
    });
  };

  // Handle password change
  const handlePasswordChange = (event) => {
    event.preventDefault();
    
    // Validate passwords
    if (newPassword !== confirmPassword) {
      setNotification({
        open: true,
        message: 'New password and confirmation do not match!',
        severity: 'error'
      });
      return;
    }
    
    if (newPassword.length < 8) {
      setNotification({
        open: true,
        message: 'Password must be at least 8 characters long!',
        severity: 'error'
      });
      return;
    }
    
    // In a real app, you would call an API to change the password
    setNotification({
      open: true,
      message: 'Password changed successfully!',
      severity: 'success'
    });
    
    // Clear form
    setCurrentPassword('');
    setNewPassword('');
    setConfirmPassword('');
  };

  // Handle notification close
  const handleNotificationClose = () => {
    setNotification({ ...notification, open: false });
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom component="div">
        Settings
      </Typography>
      
      <Grid container spacing={3}>
        {/* Account Settings */}
        <Grid item xs={12} md={6}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <SettingsIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h5">Account Settings</Typography>
              </Box>
              <Divider sx={{ mb: 3 }} />
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  User Information
                </Typography>
                <Typography variant="body1">
                  Username: {user?.username || 'Guest'}
                </Typography>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  App Preferences
                </Typography>
                <FormControlLabel
                  control={
                    <Switch
                      checked={darkMode}
                      onChange={toggleDarkMode}
                      color="primary"
                    />
                  }
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <DarkModeIcon sx={{ mr: 1, fontSize: 20 }} />
                      <Typography variant="body1">Dark Mode</Typography>
                    </Box>
                  }
                />
              </Box>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <SecurityIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h5">Security</Typography>
              </Box>
              <Divider sx={{ mb: 3 }} />
              
              <form onSubmit={handlePasswordChange}>
                <TextField
                  fullWidth
                  margin="normal"
                  label="Current Password"
                  type={showPassword ? 'text' : 'password'}
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  InputProps={{
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          onClick={() => setShowPassword(!showPassword)}
                          edge="end"
                        >
                          {showPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    )
                  }}
                />
                <TextField
                  fullWidth
                  margin="normal"
                  label="New Password"
                  type={showPassword ? 'text' : 'password'}
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  helperText="Password must be at least 8 characters long"
                />
                <TextField
                  fullWidth
                  margin="normal"
                  label="Confirm New Password"
                  type={showPassword ? 'text' : 'password'}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                />
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  sx={{ mt: 2 }}
                  startIcon={<SaveIcon />}
                >
                  Change Password
                </Button>
              </form>
            </CardContent>
          </Card>
        </Grid>
        
        {/* API Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <ApiIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h5">API Configuration</Typography>
              </Box>
              <Divider sx={{ mb: 3 }} />
              
              <Alert severity="info" sx={{ mb: 3 }}>
                These settings control how the recommendation system behaves.
              </Alert>
              
              <form onSubmit={handleApiSettingsSave}>
                <TextField
                  fullWidth
                  margin="normal"
                  label="API URL"
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                  helperText="The base URL for the recommender API"
                />
                <TextField
                  fullWidth
                  margin="normal"
                  label="Maximum Recommendations"
                  type="number"
                  value={maxRecommendations}
                  onChange={(e) => setMaxRecommendations(e.target.value)}
                  inputProps={{ min: 1, max: 20 }}
                  helperText="Number of recommendations to show (1-20)"
                />
                <TextField
                  fullWidth
                  margin="normal"
                  label="Model Type"
                  select
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                  SelectProps={{
                    native: true,
                  }}
                  helperText="The type of recommendation model to use"
                >
                  <option value="collaborative">Collaborative Filtering</option>
                  <option value="content">Content-Based</option>
                  <option value="hybrid">Hybrid</option>
                </TextField>
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  sx={{ mt: 2 }}
                  startIcon={<SaveIcon />}
                >
                  Save API Settings
                </Button>
              </form>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Notification Snackbar */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleNotificationClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleNotificationClose} 
          severity={notification.severity}
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Settings;
