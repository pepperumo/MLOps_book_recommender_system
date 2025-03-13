import React, { createContext, useContext } from 'react';

// Create context for theme mode
const ThemeModeContext = createContext();

// Custom hook to use theme mode context
export const useThemeMode = () => useContext(ThemeModeContext);

export const ThemeModeProvider = ({ value, children }) => {
  return (
    <ThemeModeContext.Provider value={value}>
      {children}
    </ThemeModeContext.Provider>
  );
};
