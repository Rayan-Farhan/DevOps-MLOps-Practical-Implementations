// Get API URL: Priority 1) Build-time env var, 2) Runtime detection, 3) Fallback
const getApiUrl = () => {
  // Build-time variable (set during Docker build)
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }

  const hostname = window.location.hostname;
  const protocol = window.location.protocol;
  const port = window.location.port;

  // Local development: use localhost:8000 (browser can't resolve Docker service names)
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'http://localhost:8000';
  }

  // Production/Cloud: use same origin (API gateway or reverse proxy routes /api/*)
  if (hostname.includes('yourdomain.com') || hostname.includes('amazonaws.com')) {
    return `${protocol}//${hostname}${port ? ':' + port : ''}`;
  }

  // Fallback: construct from current origin
  return `${protocol}//${hostname}${port ? ':' + port : ''}`;
};

const getEnvironment = () => {
  return import.meta.env.MODE || 'development';
};

const isProduction = () => {
  return getEnvironment() === 'production';
};

const isDevelopment = () => {
  return getEnvironment() === 'development';
};

export const config = {
  API_URL: getApiUrl(),
  ENVIRONMENT: getEnvironment(),
  isProduction: isProduction(),
  isDevelopment: isDevelopment(),
};

export const API_URL = config.API_URL;

if (isDevelopment()) {
  console.log('Frontend Configuration:', {
    API_URL: config.API_URL,
    ENVIRONMENT: config.ENVIRONMENT,
    hostname: window.location.hostname,
    origin: window.location.origin,
  });
}

export default config;

