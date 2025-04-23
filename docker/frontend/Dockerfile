# Stage 1: Build react application
FROM node:18-alpine AS build

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy all frontend files
COPY frontend/ ./

# Build the application
RUN npm run build

# Stage 2: Serve the application with Nginx
FROM nginx:stable-alpine

# Copy built assets from the build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY docker/frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Add non-root user and set ownership
RUN adduser -D -H -u 1000 -s /sbin/nologin -G www-data www-data && \
    chown -R www-data:www-data /usr/share/nginx/html && \
    chown -R www-data:www-data /var/cache/nginx && \
    touch /var/run/nginx.pid && \
    chown -R www-data:www-data /var/run/nginx.pid

# Set user
USER www-data

# Expose port 80
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]