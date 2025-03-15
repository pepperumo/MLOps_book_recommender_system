module.exports = function(app) {
  app.use(
    '/api',
    require('http-proxy-middleware').createProxyMiddleware({
      target: 'http://localhost:5000',
      changeOrigin: true,
    })
  );
};
