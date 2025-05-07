/**
 * @fileoverview Main entry point for the Minos-AI DeFi Strategy Platform backend service
 * @description Initializes and starts the Express server with configured middleware and routes
 * @author Minos-AI Engineering Team <engineering@minos-ai.io>
 * @copyright 2024 Minos-AI Labs, Inc.
 * @license MIT
 */

import 'reflect-metadata'; // Required for TypeORM/class-validator decorators
import { createServer, Server as HttpServer } from 'http';
import { WebSocketServer } from 'ws';
import { config, loadConfigFromEnvironment } from './config';
import { app } from './app';
import { connectDatabase } from './database';
import cluster from 'cluster';
import os from 'os';
import { join } from 'path';
import { createLogger } from './utils/logger';
import { gracefulShutdown } from './utils/process';
import { initializeRedisClient } from './services/cache.service';
import { initializeMetricsCollector } from './monitoring/metrics';
import { initializeWebSocketHandlers } from './websocket';
import { initializeEventEmitters } from './events';
import { connectSolanaCluster } from './utils/solana';
import { initializeTaskScheduler } from './scheduler';

// Initialize the logger
const logger = createLogger('server');

/**
 * Bootstrap the application - run all necessary initialization tasks
 * @returns {Promise<void>}
 */
async function bootstrap(): Promise<void> {
  try {
    // Load configuration from environment variables
    loadConfigFromEnvironment();
    logger.info(`Starting Minos-AI backend service in ${config.nodeEnv} environment`);
    
    // Create HTTP server
    const httpServer: HttpServer = createServer(app);
    
    // Initialize WebSocket server if enabled
    let wss: WebSocketServer | null = null;
    if (config.websocket.enabled) {
      wss = new WebSocketServer({ server: httpServer });
      await initializeWebSocketHandlers(wss);
      logger.info('WebSocket server initialized successfully');
    }
    
    // Connect to database
    await connectDatabase();
    logger.info('Database connection established successfully');
    
    // Initialize Redis for caching if enabled
    if (config.redis.enabled) {
      await initializeRedisClient();
      logger.info('Redis connection established successfully');
    }
    
    // Connect to Solana cluster
    await connectSolanaCluster();
    logger.info(`Connected to Solana ${config.solana.network} cluster at ${config.solana.rpcEndpoint}`);
    
    // Initialize event emitters
    initializeEventEmitters();
    logger.info('Event emitters initialized successfully');
    
    // Initialize metrics collector for monitoring
    if (config.monitoring.enabled) {
      await initializeMetricsCollector();
      logger.info('Metrics collector initialized successfully');
    }
    
    // Initialize task scheduler for background jobs
    if (config.scheduler.enabled) {
      await initializeTaskScheduler();
      logger.info('Task scheduler initialized successfully');
    }
    
    // Start the server
    httpServer.listen(config.server.port, () => {
      logger.info(`Minos-AI backend service listening on port ${config.server.port}`);
      logger.info(`API documentation available at: http://localhost:${config.server.port}/api-docs`);
      
      // Log application info
      logger.info('Application information:');
      logger.info(`- Environment: ${config.nodeEnv}`);
      logger.info(`- Version: ${process.env.npm_package_version || 'unknown'}`);
      logger.info(`- Node.js: ${process.version}`);
      logger.info(`- Process ID: ${process.pid}`);
      
      // Register signal handlers for graceful shutdown
      process.on('SIGTERM', () => gracefulShutdown(httpServer, wss));
      process.on('SIGINT', () => gracefulShutdown(httpServer, wss));
    });
    
    // Handle uncaught exceptions
    process.on('uncaughtException', (error: Error) => {
      logger.fatal({ err: error }, 'Uncaught Exception');
      // Perform a graceful shutdown
      gracefulShutdown(httpServer, wss, 1);
    });
    
    // Handle unhandled promise rejections
    process.on('unhandledRejection', (reason: unknown) => {
      logger.fatal({ err: reason }, 'Unhandled Promise Rejection');
      // Log but don't exit, as per Node.js best practices
    });
    
  } catch (error) {
    logger.fatal({ err: error }, 'Failed to initialize application');
    process.exit(1);
  }
}

/**
 * Main application entry point with clustering support for production
 */
(async function main() {
  // Use clustering in production for horizontal scaling
  if (config.server.cluster && config.nodeEnv === 'production' && cluster.isPrimary) {
    const numCPUs = config.server.workers || os.cpus().length;
    logger.info(`Primary ${process.pid} is running`);
    logger.info(`Starting ${numCPUs} workers...`);
    
    // Fork workers
    for (let i = 0; i < numCPUs; i++) {
      cluster.fork();
    }
    
    // Log when a worker dies and create a replacement
    cluster.on('exit', (worker, code, signal) => {
      logger.warn(`Worker ${worker.process.pid} died with code ${code} and signal ${signal}`);
      logger.info('Starting a new worker...');
      cluster.fork();
    });
  } else {
    // Worker process or single-threaded mode
    await bootstrap();
  }
})().catch((error) => {
  console.error('Fatal error during application startup:', error);
  process.exit(1);
});

// Export for testing purposes
export { app };