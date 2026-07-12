# Synapse UI image — multi-stage: build the Vite/React bundle, then serve it from nginx.
# The browser talks to a single origin; nginx reverse-proxies /api and /ws to the api
# container (mirrors the Vite dev proxy in vite.config.ts). No build-time API URL needed.

# --- build stage ---------------------------------------------------------------
FROM node:22-slim AS build
WORKDIR /app

# Install deps first so this layer caches across source-only changes.
COPY package.json package-lock.json ./
RUN npm ci

# Source + build (tsc -b && vite build → /app/dist).
COPY . .
RUN npm run build

# --- serve stage ---------------------------------------------------------------
FROM nginx:1.27-alpine AS serve
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
