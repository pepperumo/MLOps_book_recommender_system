import React, { createContext, useContext } from 'react';

// Create context
const AuthContext = createContext();

// Custom hook to use the auth context
export const useAuth = () => useContext(AuthContext);

// Provider component - simplified stub version with no actual authentication
export const AuthProvider = ({ children }) => {
  // No user authentication as per requirements
  const user = null;
  const isAuthenticated = false;
  
  // Empty stub functions
  const login = () => {
    console.log('Login functionality removed as per requirements');
  };
  
  const logout = () => {
    console.log('Logout functionality removed as per requirements');
  };
  
  const register = () => {
    console.log('Register functionality removed as per requirements');
  };

  return (
    <AuthContext.Provider value={{ 
      user, 
      isAuthenticated, 
      login, 
      logout, 
      register 
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;
