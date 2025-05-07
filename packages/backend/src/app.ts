/**
 * @fileoverview Express application configuration for Minos-AI backend
 * @description Configures the Express application with middleware, routes, and error handlers
 * @author Minos-AI Engineering Team <engineering@minos-ai.io>
 * @copyright 2024 Minos-AI Labs, Inc.
 * @license MIT
 */

import express, { Application, NextFunction, Request, Response } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import bodyParser from 'body-parser';
import { rateLimit, RateLimitRequestHandler } from 'express-rate-limit';
import 'express-async-errors'; // Enables async error handling
import { expressjwt } from 'express-jwt';
import swaggerJSDoc from 'swagger-jsdoc';
import swaggerUI from 'swagger-ui-express';
import { createLogger } from './utils/logger';
import { config } from './config';
import { errorHandler } from './middlewares/error-handler.middleware';
import { requestIdMiddleware } from './middlewares/request-id.middleware';
import { loggerMiddleware } from './middlewares/logger.middleware';
import { metricsMiddleware } from './middlewares/metrics.middleware';
import { securityMiddleware } from './middlewares/security.middleware';
import { correlationIdMiddleware } from './middlewares/correlation-id.middleware';
import { sanitizationMiddleware } from './middlewares/sanitization.middleware';
import { validationMiddleware } from './middlewares/validation.middleware';
import { healthRouter } from './routes/health.routes';
import { vaultRoutes } from './routes/vaults.routes';
import { authRoutes } from './routes/auth.routes';
import { strategyRoutes } from './routes/strategies.routes';
import { adminRoutes } from './routes/admin.routes';
import { userRoutes } from './routes/users.routes';
import { apiKeyRoutes } from './routes/api-keys.routes';
import { transactionRoutes } from './routes/transactions.routes';
import { webhookRoutes } from './routes/webhooks.routes';
import { documentationRoutes } from './routes/documentation.routes';
import { HttpError } from './utils/errors';
import { join } from 'path';

// Initialize logger instance
const logger = createLogger('app');

// Create Express application
const app: Application = express();

// Configure Swagger API documentation
const swaggerOptions = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'Minos-AI DeFi Strategy Platform API',
      version: process.env.npm_package_version || '0.1.0',
      description: 'API documentation for the Minos-AI DeFi Strategy Platform',
      license: {
        name: 'MIT',
        url: 'https://opensource.org/licenses/MIT',
      },
      contact: {
        name: 'Minos-AI Support',
        url: 'https://minos-ai.io/support',
        email: 'support@minos-ai.io',
      },
    },
    servers: [
      {
        url: `http://localhost:${config.server.port}`,
        description: 'Development server',
      },
      {
        url: 'https://api.staging.minos-ai.io',
        description: 'Staging server',
      },
      {
        url: 'https://api.minos-ai.io',
        description: 'Production server',
      },
    ],
    components: {
      securitySchemes: {
        BearerAuth: {
          type: 'http',
          scheme: 'bearer',
          bearerFormat: 'JWT',
        },
        ApiKeyAuth: {
          type: 'apiKey',
          in: 'header',
          name: 'X-API-KEY',
        },
      },
    },
  },
  apis: ['./src/routes/*.ts', './src/models/*.ts', './src/dtos/*.ts'],
};

const swaggerSpec = swaggerJSDoc(swaggerOptions);

// Configure rate limiting
const apiLimiter: RateLimitRequestHandler = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: config.security.rateLimit.maxRequests, // Limit each IP to X requests per windowMs
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    status: 429,
    message: 'Too many requests, please try again later.',
  },
  skip: (req: Request) => {
    // Skip rate limiting for trusted sources or internal requests
    const clientIp = req.ip || req.socket.remoteAddress || '';
    return config.security.rateLimit.whitelist.includes(clientIp);
  },
});

// Apply global middleware
app.set('trust proxy', config.server.trustProxy);
app.use(requestIdMiddleware());
app.use(correlationIdMiddleware());
app.use(helmet()); // Security headers
app.use(securityMiddleware());
app.use(compression()); // Compress responses
app.use(bodyParser.json({ limit: '2mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '2mb' }));
app.use(morgan(config.isDevelopment ? 'dev' : 'combined', { 
  stream: { 
    write: (message: string) => logger.http(message.trim()) 
  } 
}));
app.use(loggerMiddleware());
app.use(sanitizationMiddleware());

// CORS configuration
app.use(cors({
  origin: config.security.cors.origins,
  methods: config.security.cors.methods,
  allowedHeaders: config.security.cors.allowedHeaders,
  exposedHeaders: config.security.cors.exposedHeaders,
  credentials: config.security.cors.credentials,
  maxAge: config.security.cors.maxAge,
}));

// Metrics collection for monitoring
if (config.monitoring.enabled) {
  app.use(metricsMiddleware());
}

// Static files
app.use('/static', express.static(join(__dirname, '../public')));

// API Documentation
app.use('/api-docs', swaggerUI.serve, swaggerUI.setup(swaggerSpec, {
  customCss: '.swagger-ui .topbar { display: none }',
  customSiteTitle: 'Minos-AI API Documentation',
}));

// API Routes
app.use('/api/v1/health', healthRouter);

// Apply rate limiting to API routes
if (config.security.rateLimit.enabled) {
  app.use('/api/v1', apiLimiter);
}

// Authentication middleware for protected routes
const jwtMiddleware = expressjwt({
  secret: config.security.jwt.secret,
  algorithms: ['HS256'],
  requestProperty: 'user',
  getToken: (req) => {
    if (req.headers.authorization && req.headers.authorization.split(' ')[0] === 'Bearer') {
      return req.headers.authorization.split(' ')[1];
    }
    return null;
  },
}).unless({
  path: [
    '/api/v1/health',
    '/api/v1/auth/login',
    '/api/v1/auth/register',
    '/api/v1/auth/verify',
    '/api/v1/auth/forgot-password',
    '/api/v1/auth/reset-password',
    '/metrics',
    '/api-docs',
    /\/webhooks\/.*/,
  ],
});

// Apply validation middleware globally
app.use(validationMiddleware());

// Apply JWT authentication
app.use(jwtMiddleware);

// Register API routes
app.use('/api/v1/auth', authRoutes);
app.use('/api/v1/vaults', vaultRoutes);
app.use('/api/v1/strategies', strategyRoutes);
app.use('/api/v1/users', userRoutes);
app.use('/api/v1/admin', adminRoutes);
app.use('/api/v1/api-keys', apiKeyRoutes);
app.use('/api/v1/transactions', transactionRoutes);
app.use('/api/v1/webhooks', webhookRoutes);
app.use('/api/v1/docs', documentationRoutes);

// Error handling middleware
app.use((req: Request, res: Response, next: NextFunction) => {
  next(new HttpError(404, 'Not Found', 'The requested resource was not found'));
});

// Global error handler
app.use(errorHandler);

// Health check endpoint
app.get('/healthz', (req: Request, res: Response) => {
  res.status(200).json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    version: process.env.npm_package_version || 'unknown',
    environment: config.nodeEnv,
  });
});

// Metrics endpoint for Prometheus
if (config.monitoring.enabled) {
  const { collectDefaultMetrics, register } = require('prom-client');
  collectDefaultMetrics({ register });
  
  app.get('/metrics', async (req: Request, res: Response) => {
    res.set('Content-Type', register.contentType);
    res.end(await register.metrics());
  });
}

// Initialize application event listeners
app.on('error', (error: Error) => {
  logger.error({ err: error }, 'Application error occurred');
});

// Exports
export { app };