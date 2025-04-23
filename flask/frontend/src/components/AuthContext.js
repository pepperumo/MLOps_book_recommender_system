import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Check if user is already logged in on mount
  useEffect(() => {
    const checkAuth = () => {
      const token = localStorage.getItem('authToken');
      console.log('Initial auth check:', token ? 'Token found' : 'No token found');
      
      if (token) {
        setIsAuthenticated(true);
        setUser({ username: localStorage.getItem('username') || 'User' });
      }
      setLoading(false);
    };
    
    checkAuth();
  }, []);

  // Login function
  const login = async (username, password) => {
    console.log('Login attempt with:', username);
    
    // For demo purposes - hardcoded credentials
    if (username === 'demo' && password === 'password') {
      console.log('Login successful');
      localStorage.setItem('authToken', 'demo-token');
      localStorage.setItem('username', username);
      setIsAuthenticated(true);
      setUser({ username });
      return true;
    }
    
    console.log('Login failed - invalid credentials');
    throw new Error('Invalid credentials');
  };

  // Logout function
  const logout = () => {
    console.log('Logging out');
    localStorage.removeItem('authToken');
    localStorage.removeItem('username');
    setIsAuthenticated(false);
    setUser(null);
  };

  const value = {
    isAuthenticated,
    user,
    loading,
    login,
    logout
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  return useContext(AuthContext);
};
