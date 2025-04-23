const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:8000',  // Your FastAPI server
      changeOrigin: true,
      logLevel: 'debug'  // Add this to see detailed logs
    })
  );
};
