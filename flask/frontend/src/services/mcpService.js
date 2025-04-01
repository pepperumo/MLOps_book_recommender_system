/**
 * MCP Service - Client for interacting with the Book Recommender MCP Server
 */

import axios from 'axios';

// Base URL for the MCP server
const MCP_BASE_URL = '/mcp';

/**
 * Call an MCP tool with the given parameters
 * @param {string} toolName - Name of the MCP tool to call
 * @param {object} params - Parameters to pass to the tool
 * @returns {Promise<any>} - Response from the MCP server
 */
export const callMcpTool = async (toolName, params) => {
  try {
    const response = await axios.post(
      `${MCP_BASE_URL}/tools/${toolName}`,
      { params }
    );
    return response.data;
  } catch (error) {
    console.error(`Error calling MCP tool ${toolName}:`, error);
    throw error;
  }
};

/**
 * Access an MCP resource
 * @param {string} resourceUri - URI of the MCP resource to access
 * @returns {Promise<any>} - Response from the MCP server
 */
export const accessMcpResource = async (resourceUri) => {
  try {
    const response = await axios.get(
      `${MCP_BASE_URL}/resources/read?uri=${encodeURIComponent(resourceUri)}`
    );
    return response.data;
  } catch (error) {
    console.error(`Error accessing MCP resource ${resourceUri}:`, error);
    throw error;
  }
};

/**
 * Check if MCP is available on the server
 * @returns {Promise<boolean>} - Whether MCP is available
 */
export const checkMcpAvailability = async () => {
  try {
    const response = await axios.get('/mcp-docs');
    return response.data.is_available === true;
  } catch (error) {
    console.error('Error checking MCP availability:', error);
    return false;
  }
};

/**
 * Get book recommendations for a user using MCP
 * @param {number} userId - ID of the user to get recommendations for
 * @param {number} count - Number of recommendations to return
 * @returns {Promise<any>} - Book recommendations
 */
export const getBookRecommendationsViaMcp = async (userId, count = 5) => {
  return callMcpTool('recommend_books_for_user', { user_id: userId, n: count });
};

/**
 * Find similar books using MCP
 * @param {number} bookId - ID of the book to find similar books for
 * @param {number} count - Number of similar books to return
 * @returns {Promise<any>} - Similar books
 */
export const getSimilarBooksViaMcp = async (bookId, count = 5) => {
  return callMcpTool('find_similar_books', { book_id: bookId, n: count });
};

/**
 * Get top books using MCP
 * @param {number} count - Number of top books to return
 * @param {number} minRatings - Minimum number of ratings a book must have
 * @returns {Promise<any>} - Top books
 */
export const getTopBooksViaMcp = async (count = 10, minRatings = 50) => {
  return callMcpTool('get_top_books', { n: count, min_ratings: minRatings });
};

/**
 * Get book metadata using MCP
 * @param {number} bookId - ID of the book to get metadata for
 * @returns {Promise<any>} - Book metadata
 */
export const getBookMetadataViaMcp = async (bookId) => {
  return accessMcpResource(`books://metadata/${bookId}`);
};

/**
 * Get user ratings using MCP
 * @param {number} userId - ID of the user to get ratings for
 * @returns {Promise<any>} - User ratings
 */
export const getUserRatingsViaMcp = async (userId) => {
  return accessMcpResource(`users://ratings/${userId}`);
};

export default {
  callMcpTool,
  accessMcpResource,
  checkMcpAvailability,
  getBookRecommendationsViaMcp,
  getSimilarBooksViaMcp,
  getTopBooksViaMcp,
  getBookMetadataViaMcp,
  getUserRatingsViaMcp
};
