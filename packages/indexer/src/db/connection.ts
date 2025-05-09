/**
 * Database Connection Module
 * 
 * Handles the connection to the PostgreSQL database using TypeORM.
 * This module provides functionality to:
 * - Initialize the database connection
 * - Run migrations
 * - Provide access to repositories
 */

import { DataSource } from 'typeorm';
import { join } from 'path';
import { config } from 'dotenv';
import logger from '../utils/logger';

// Load environment variables
config();

// Database connection parameters from environment variables
const DB_HOST = process.env.DB_HOST || 'localhost';
const DB_PORT = parseInt(process.env.DB_PORT || '5432', 10);
const DB_USERNAME = process.env.DB_USERNAME || 'postgres';
const DB_PASSWORD = process.env.DB_PASSWORD || 'postgres';
const DB_DATABASE = process.env.DB_DATABASE || 'minos_ai';
const DB_SCHEMA = process.env.DB_SCHEMA || 'public';
const DB_SSL = process.env.DB_SSL === 'true';

// Entity and migration paths relative to dist folder
const ENTITIES_PATH = join(__dirname, '../entities/**/*.js');
const MIGRATIONS_PATH = join(__dirname, '../db/migrations/**/*.js');

/**
 * TypeORM data source configuration
 */
export const AppDataSource = new DataSource({
  type: 'postgres',
  host: DB_HOST,
  port: DB_PORT,
  username: DB_USERNAME,
  password: DB_PASSWORD,
  database: DB_DATABASE,
  schema: DB_SCHEMA,
  synchronize: false, // Never set to true in production!
  logging: process.env.NODE_ENV === 'development',
  entities: [ENTITIES_PATH],
  migrations: [MIGRATIONS_PATH],
  migrationsTableName: 'minos_ai_migrations',
  ssl: DB_SSL ? { rejectUnauthorized: false } : false,
  connectTimeoutMS: 10000, // 10 seconds
  extra: {
    // Connection pool configuration
    max: parseInt(process.env.DB_POOL_MAX || '10', 10),
    min: parseInt(process.env.DB_POOL_MIN || '2', 10),
    idleTimeoutMillis: 30000, // 30 seconds
    connectionTimeoutMillis: 10000, // 10 seconds
  },
});

/**
 * Initialize database connection
 */
export async function initializeDatabase(): Promise<DataSource> {
  try {
    // Initialize connection
    if (!AppDataSource.isInitialized) {
      logger.info('Initializing database connection...');
      await AppDataSource.initialize();
      logger.info(`Connected to database: ${DB_DATABASE} on ${DB_HOST}:${DB_PORT}`);
    }
    
    // Run pending migrations
    if (process.env.AUTO_RUN_MIGRATIONS === 'true') {
      logger.info('Running database migrations...');
      const migrations = await AppDataSource.runMigrations();
      logger.info(`Successfully ran ${migrations.length} migrations`);
    }
    
    return AppDataSource;
  } catch (error) {
    logger.error('Error initializing database connection:', error);
    throw error;
  }
}

/**
 * Check if database connection is healthy
 */
export async function checkDatabaseHealth(): Promise<boolean> {
  try {
    if (!AppDataSource.isInitialized) {
      return false;
    }
    
    // Try to query the database
    await AppDataSource.query('SELECT 1');
    return true;
  } catch (error) {
    logger.error('Database health check failed:', error);
    return false;
  }
}

/**
 * Close database connection
 */
export async function closeDatabase(): Promise<void> {
  try {
    if (AppDataSource.isInitialized) {
      await AppDataSource.destroy();
      logger.info('Database connection closed');
    }
  } catch (error) {
    logger.error('Error closing database connection:', error);
    throw error;
  }
}

// Export the data source as default
export default AppDataSource;