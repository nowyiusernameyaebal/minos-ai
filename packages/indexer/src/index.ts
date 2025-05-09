/**
 * Minos-AI Indexer
 * 
 * Main entry point for the Solana blockchain indexer for the Minos-AI platform.
 * This service monitors on-chain activities related to AI models, trading signals,
 * strategies, and vaults, then stores them in a structured database for efficient
 * querying and analysis.
 */

import 'reflect-metadata';
import { config } from 'dotenv';
import { Connection } from '@solana/web3.js';
import { initializeDatabase } from './db/connection';
import logger from './utils/logger';
import { createProcessors } from './processors';
import { startServer } from './api/server';
import { monitorHealthCheck } from './utils/monitor';

// Load environment variables
config();

// Connection configs
const SOLANA_RPC_URL = process.env.SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com';
const API_PORT = parseInt(process.env.API_PORT || '3000', 10);
const COMMITMENT = process.env.COMMITMENT || 'confirmed';

/**
 * Startup sequence
 */
async function bootstrap() {
  try {
    // Initialize Solana connection
    logger.info(`Connecting to Solana network: ${SOLANA_RPC_URL}`);
    const connection = new Connection(SOLANA_RPC_URL, COMMITMENT as any);

    // Check connection
    const version = await connection.getVersion();
    logger.info(`Connected to Solana, version: ${version['solana-core']}`);

    // Initialize database connection
    logger.info('Initializing database connection');
    const dataSource = await initializeDatabase();
    logger.info(`Database connected: ${dataSource.options.database}`);

    // Initialize and start processors
    logger.info('Starting blockchain processors');
    const processorManager = createProcessors(connection, dataSource);
    await processorManager.startAll();
    logger.info('All processors started successfully');

    // Start API server
    logger.info(`Starting API server on port: ${API_PORT}`);
    await startServer(API_PORT, dataSource);

    // Start health monitoring
    monitorHealthCheck(processorManager, connection);
    
    logger.info('Minos-AI Indexer fully initialized and running');
    
    // Handle shutdown signals
    process.on('SIGINT', async () => {
      logger.info('Received SIGINT, gracefully shutting down...');
      await processorManager.stopAll();
      await dataSource.destroy();
      process.exit(0);
    });
    
    process.on('SIGTERM', async () => {
      logger.info('Received SIGTERM, gracefully shutting down...');
      await processorManager.stopAll();
      await dataSource.destroy();
      process.exit(0);
    });
    
  } catch (error) {
    logger.error('Failed to start indexer service:', error);
    process.exit(1);
  }
}

// Start the application
bootstrap();