FROM node:16-alpine as build

WORKDIR /app

# Copy package files first
COPY frontend/package.json ./
COPY frontend/package-lock.json* ./

# Install dependencies with legacy peer deps
RUN npm install --legacy-peer-deps

# Copy the rest of the frontend code
COPY frontend/ ./

# Set environment variables for build
ENV DISABLE_ESLINT_PLUGIN=true
ENV CI=false

# Create a direct symlink to react-scripts in node_modules
RUN ln -sf /app/node_modules/.bin/react-scripts /usr/local/bin/react-scripts

# Build the React app
RUN node /app/node_modules/react-scripts/bin/react-scripts.js build

FROM nginx:alpine
# Copy built files from the build stage
COPY --from=build /app/build /usr/share/nginx/html
# Copy nginx configuration
COPY docker/frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]