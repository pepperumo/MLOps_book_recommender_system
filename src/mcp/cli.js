#!/usr/bin/env node

const { program } = require('commander');
const spawn = require('cross-spawn');
const path = require('path');
const fs = require('fs');

// Define the program version and description
program
  .version('1.0.0')
  .description('Book Recommendation MCP Server');

// Command to run the MCP server
program
  .command('serve')
  .description('Start the Book Recommendation MCP server')
  .option('-p, --port <port>', 'Port to run the server on', '8080')
  .option('-h, --host <host>', 'Host to bind the server to', '0.0.0.0')
  .option('-d, --data-dir <path>', 'Path to the data directory')
  .option('-m, --models-dir <path>', 'Path to the models directory')
  .option('--python <path>', 'Path to Python interpreter', process.env.VIRTUAL_ENV ? 
    path.join(process.env.VIRTUAL_ENV, 'Scripts', 'python.exe') : 'python')
  .action((options) => {
    console.log('Starting Book Recommendation MCP Server...');
    
    // Get the path to the Python script
    const scriptPath = path.join(__dirname, 'mcp_server.py');
    
    // Use Python from virtual environment if available
    const pythonCommand = options.python || 
                          (process.env.VIRTUAL_ENV ? 
                            path.join(process.env.VIRTUAL_ENV, 'Scripts', 'python.exe') : 
                            (process.platform === 'win32' ? 'python' : 'python3'));
    
    console.log(`Using Python interpreter: ${pythonCommand}`);
    
    // Build arguments for the Python script
    const args = [scriptPath];
    
    // Add optional arguments
    if (options.port) {
      process.env.MCP_PORT = options.port;
    }
    
    if (options.host) {
      process.env.MCP_HOST = options.host;
    }
    
    if (options.dataDir) {
      process.env.MCP_DATA_DIR = options.dataDir;
    }
    
    if (options.modelsDir) {
      process.env.MCP_MODELS_DIR = options.modelsDir;
    }
    
    // Spawn the Python process
    const child = spawn(pythonCommand, args, { 
      stdio: 'inherit',
      env: process.env
    });
    
    // Handle process exit
    child.on('close', (code) => {
      if (code !== 0) {
        console.error(`MCP server exited with code ${code}`);
        process.exit(code);
      }
    });
    
    // Handle process errors
    child.on('error', (err) => {
      console.error('Failed to start MCP server:', err);
      process.exit(1);
    });
  });

// Parse command line arguments
program.parse(process.argv);

// If no arguments provided, show help
if (!process.argv.slice(2).length) {
  program.outputHelp();
}