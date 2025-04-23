module.exports = function(app) {
  const { createProxyMiddleware } = require('http-proxy-middleware');
  
  // Proxy settings for API requests
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:5000',
      changeOrigin: true,
    })
  );
};
