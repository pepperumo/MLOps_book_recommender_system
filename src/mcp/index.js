/**
 * Node.js module entry point for Book Recommendation MCP Server
 */

const path = require('path');
const { spawn } = require('cross-spawn');

/**
 * Start the Book Recommendation MCP Server
 * @param {Object} options - Server configuration options
 * @param {string} [options.host='0.0.0.0'] - Host to bind the server to
 * @param {number} [options.port=8080] - Port to run the server on
 * @param {string} [options.dataDir] - Path to the data directory
 * @param {string} [options.modelsDir] - Path to the models directory
 * @returns {Object} The spawned process
 */
function startServer(options = {}) {
  const host = options.host || '0.0.0.0';
  const port = options.port || 8080;
  const dataDir = options.dataDir || null;
  const modelsDir = options.modelsDir || null;
  
  // Set environment variables
  const env = { ...process.env };
  env.MCP_HOST = host;
  env.MCP_PORT = port.toString();
  
  if (dataDir) {
    env.MCP_DATA_DIR = dataDir;
  }
  
  if (modelsDir) {
    env.MCP_MODELS_DIR = modelsDir;
  }
  
  // Get the path to the Python script
  const scriptPath = path.join(__dirname, 'mcp_server.py');
  
  // Determine Python command based on platform
  const pythonCommand = process.platform === 'win32' ? 'python' : 'python3';
  
  // Start the server process
  const serverProcess = spawn(pythonCommand, [scriptPath], {
    env,
    stdio: 'inherit'
  });
  
  // Return the process for control
  return serverProcess;
}

module.exports = {
  startServer
};