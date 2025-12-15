FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Expose port
EXPOSE 3001

# Start dev server (accessible from outside)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3001"]
