/**
 * @fileoverview Configuration management for Minos-AI backend service
 * @description Centralizes all configuration variables and provides validation
 * @author Minos-AI Engineering Team <engineering@minos-ai.io>
 * @copyright 2024 Minos-AI Labs, Inc.
 * @license MIT
 */

import dotenv from 'dotenv';
import path from 'path';
import Joi from 'joi';
import * as fs from 'fs';
import { LogLevel } from 'pino';

// Load environment variables from .env file
const envFile = process.env.NODE_ENV === 'test' ? '.env.test' : '.env';
const envPath = path.resolve(process.cwd(), envFile);

if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath });
} else {
  dotenv.config(); // Fallback to default .env file
}

/**
 * Environment variable validation schema
 */
const envVarsSchema = Joi.object()
  .keys({
    // Server configuration
    NODE_ENV: Joi.string()
      .valid('development', 'production', 'test', 'staging')
      .default('development'),
    PORT: Joi.number().default(3000),
    HOST: Joi.string().default('localhost'),
    TRUST_PROXY: Joi.boolean().default(false),
    API_PREFIX: Joi.string().default('/api/v1'),
    ENABLE_CLUSTER: Joi.boolean().default(false),
    CLUSTER_WORKERS: Joi.number().default(0),
    
    // Logging configuration
    LOG_LEVEL: Joi.string()
      .valid('fatal', 'error', 'warn', 'info', 'debug', 'trace', 'silent')
      .default('info'),
    LOG_FORMAT: Joi.string().valid('json', 'pretty').default('json'),
    LOG_ENABLE_CONSOLE: Joi.boolean().default(true),
    LOG_ENABLE_FILE: Joi.boolean().default(false),
    LOG_FILE_PATH: Joi.string().default('./logs'),
    LOG_FILE_MAX_SIZE: Joi.string().default('10m'),
    LOG_FILE_MAX_FILES: Joi.number().default(7),
    LOG_DISABLE_COLORS: Joi.boolean().default(false),
    LOG_REDACT_FIELDS: Joi.string().default('password,secretKey,token,apiKey'),
    
    // Security configuration
    JWT_SECRET: Joi.string().required().min(32),
    JWT_EXPIRES_IN: Joi.string().default('1d'),
    JWT_REFRESH_EXPIRES_IN: Joi.string().default('7d'),
    CORS_ORIGINS: Joi.string().default('*'),
    CORS_METHODS: Joi.string().default('GET,POST,PUT,DELETE,PATCH,OPTIONS'),
    CORS_ALLOWED_HEADERS: Joi.string().default('Content-Type,Authorization,X-Requested-With,X-API-KEY'),
    CORS_EXPOSED_HEADERS: Joi.string().default('Content-Range,X-Total-Count'),
    CORS_CREDENTIALS: Joi.boolean().default(true),
    CORS_MAX_AGE: Joi.number().default(86400),
    RATE_LIMIT_ENABLED: Joi.boolean().default(true),
    RATE_LIMIT_MAX_REQUESTS: Joi.number().default(100),
    RATE_LIMIT_WINDOW_MS: Joi.number().default(900000),
    RATE_LIMIT_WHITELIST: Joi.string().default('127.0.0.1,::1'),
    ENABLE_CSRF: Joi.boolean().default(true),
    CSRF_TOKEN_SECRET: Joi.string().when('ENABLE_CSRF', {
      is: true,
      then: Joi.string().required().min(32),
      otherwise: Joi.string().optional(),
    }),
    
    // Solana configuration
    SOLANA_RPC_ENDPOINT: Joi.string().required(),
    SOLANA_NETWORK: Joi.string().valid('mainnet-beta', 'testnet', 'devnet', 'localnet').default('devnet'),
    SOLANA_COMMITMENT: Joi.string()
      .valid('processed', 'confirmed', 'finalized', 'recent', 'single', 'singleGossip', 'root', 'max')
      .default('confirmed'),
    SOLANA_WALLET_PRIVATE_KEY: Joi.string().required(),
    SOLANA_CONNECTION_TIMEOUT_MS: Joi.number().default(30000),
    
    // Database configuration
    DB_TYPE: Joi.string().valid('postgres', 'mongodb', 'mysql').default('mongodb'),
    DB_HOST: Joi.string().required(),
    DB_PORT: Joi.number().default(27017),
    DB_NAME: Joi.string().required(),
    DB_USER: Joi.string().optional(),
    DB_PASSWORD: Joi.string().optional(),
    DB_AUTH_SOURCE: Joi.string().optional(),
    DB_SSL: Joi.boolean().default(false),
    DB_URI: Joi.string().optional(),
    DB_POOL_SIZE: Joi.number().default(10),
    DB_KEEP_ALIVE: Joi.boolean().default(true),
    DB_CONNECT_TIMEOUT_MS: Joi.number().default(30000),
    DB_RETRY_WRITES: Joi.boolean().default(true),
    DB_RETRY_READS: Joi.boolean().default(true),
    
    // Redis configuration (for caching and rate limiting)
    REDIS_ENABLED: Joi.boolean().default(false),
    REDIS_HOST: Joi.string().when('REDIS_ENABLED', {
      is: true,
      then: Joi.string().required(),
      otherwise: Joi.string().optional(),
    }),
    REDIS_PORT: Joi.number().default(6379),
    REDIS_PASSWORD: Joi.string().optional().allow(''),
    REDIS_DB: Joi.number().default(0),
    REDIS_CONNECT_TIMEOUT: Joi.number().default(10000),
    REDIS_TTL: Joi.number().default(86400),
    REDIS_CACHE_PREFIX: Joi.string().default('minos-ai:'),
    
    // Monitoring configuration
    MONITORING_ENABLED: Joi.boolean().default(false),
    SENTRY_DSN: Joi.string().optional(),
    SENTRY_ENVIRONMENT: Joi.string().optional(),
    SENTRY_RELEASE: Joi.string().optional(),
    SENTRY_TRACES_SAMPLE_RATE: Joi.number().default(0.1),
    ENABLE_PROMETHEUS: Joi.boolean().default(false),
    NEW_RELIC_LICENSE_KEY: Joi.string().optional(),
    NEW_RELIC_APP_NAME: Joi.string().optional(),
    
    // WebSocket configuration
    WEBSOCKET_ENABLED: Joi.boolean().default(false),
    WEBSOCKET_PATH: Joi.string().default('/ws'),
    WEBSOCKET_PING_INTERVAL: Joi.number().default(30000),
    
    // Task Scheduler configuration
    SCHEDULER_ENABLED: Joi.boolean().default(true),
    SCHEDULER_TIMEZONE: Joi.string().default('UTC'),
    
    // File storage configuration
    STORAGE_TYPE: Joi.string().valid('local', 's3', 'gcs').default('local'),
    STORAGE_LOCAL_PATH: Joi.string().default('./uploads'),
    AWS_S3_BUCKET: Joi.string().optional(),
    AWS_S3_REGION: Joi.string().optional(),
    AWS_ACCESS_KEY_ID: Joi.string().optional(),
    AWS_SECRET_ACCESS_KEY: Joi.string().optional(),
    GCS_BUCKET: Joi.string().optional(),
    GCS_PROJECT_ID: Joi.string().optional(),
    GCS_CLIENT_EMAIL: Joi.string().optional(),
    GCS_PRIVATE_KEY: Joi.string().optional(),
    
    // Vault program configuration
    VAULT_PROGRAM_ID: Joi.string().required(),
    STRATEGY_EXECUTION_INTERVAL: Joi.number().default(3600000), // 1 hour in milliseconds
    MAX_STRATEGY_EXECUTION_TIME: Joi.number().default(300000), // 5 minutes in milliseconds
    
    // API documentation
    ENABLE_SWAGGER: Joi.boolean().default(true),
    
    // Advanced features
    FEATURE_FLAG_NEW_STRATEGY_ENGINE: Joi.boolean().default(false),
    FEATURE_FLAG_ADVANCED_RISK_METRICS: Joi.boolean().default(false),
    FEATURE_FLAG_MULTI_CHAIN_SUPPORT: Joi.boolean().default(false),
  })
  .unknown();

// Validate environment variables
const { value: envVars, error } = envVarsSchema.prefs({ errors: { label: 'key' } }).validate(process.env);

if (error) {
  throw new Error(`Environment validation error: ${error.message}`);
}

/**
 * Configuration interface for strongly typed access
 */
interface IConfig {
  isDevelopment: boolean;
  isProduction: boolean;
  isTest: boolean;
  isStaging: boolean;
  nodeEnv: string;
  logLevel: LogLevel;
  
  server: {
    port: number;
    host: string;
    apiPrefix: string;
    trustProxy: boolean;
    cluster: boolean;
    workers: number;
  };
  
  logging: {
    level: LogLevel;
    format: 'json' | 'pretty';
    enableConsole: boolean;
    enableFile: boolean;
    filePath: string;
    fileMaxSize: string;
    fileMaxFiles: number;
    disableColors: boolean;
    redactFields: string[];
  };
  
  security: {
    jwt: {
      secret: string;
      expiresIn: string;
      refreshExpiresIn: string;
    };
    cors: {
      origins: string[];
      methods: string[];
      allowedHeaders: string[];
      exposedHeaders: string[];
      credentials: boolean;
      maxAge: number;
    };
    rateLimit: {
      enabled: boolean;
      maxRequests: number;
      windowMs: number;
      whitelist: string[];
    };
    csrf: {
      enabled: boolean;
      tokenSecret?: string;
    };
  };
  
  solana: {
    rpcEndpoint: string;
    network: 'mainnet-beta' | 'testnet' | 'devnet' | 'localnet';
    commitment: string;
    walletPrivateKey: string;
    connectionTimeoutMs: number;
  };
  
  database: {
    type: 'postgres' | 'mongodb' | 'mysql';
    host: string;
    port: number;
    name: string;
    user?: string;
    password?: string;
    authSource?: string;
    ssl: boolean;
    uri?: string;
    poolSize: number;
    keepAlive: boolean;
    connectTimeoutMs: number;
    retryWrites: boolean;
    retryReads: boolean;
  };
  
  redis: {
    enabled: boolean;
    host?: string;
    port: number;
    password?: string;
    db: number;
    connectTimeout: number;
    ttl: number;
    cachePrefix: string;
  };
  
  monitoring: {
    enabled: boolean;
    sentry?: {
      dsn?: string;
      environment?: string;
      release?: string;
      tracesSampleRate: number;
    };
    prometheus: boolean;
    newRelic?: {
      licenseKey?: string;
      appName?: string;
    };
  };
  
  websocket: {
    enabled: boolean;
    path: string;
    pingInterval: number;
  };
  
  scheduler: {
    enabled: boolean;
    timezone: string;
  };
  
  storage: {
    type: 'local' | 's3' | 'gcs';
    localPath: string;
    s3?: {
      bucket?: string;
      region?: string;
      accessKeyId?: string;
      secretAccessKey?: string;
    };
    gcs?: {
      bucket?: string;
      projectId?: string;
      clientEmail?: string;
      privateKey?: string;
    };
  };
  
  vault: {
    programId: string;
    strategyExecutionInterval: number;
    maxStrategyExecutionTime: number;
  };
  
  swagger: {
    enabled: boolean;
  };
  
  featureFlags: {
    newStrategyEngine: boolean;
    advancedRiskMetrics: boolean;
    multiChainSupport: boolean;
  };
}

/**
 * Application configuration object
 */
export const config: IConfig = {
  isDevelopment: envVars.NODE_ENV === 'development',
  isProduction: envVars.NODE_ENV === 'production',
  isTest: envVars.NODE_ENV === 'test',
  isStaging: envVars.NODE_ENV === 'staging',
  nodeEnv: envVars.NODE_ENV,
  logLevel: envVars.LOG_LEVEL as LogLevel,
  
  server: {
    port: envVars.PORT,
    host: envVars.HOST,
    apiPrefix: envVars.API_PREFIX,
    trustProxy: envVars.TRUST_PROXY,
    cluster: envVars.ENABLE_CLUSTER,
    workers: envVars.CLUSTER_WORKERS,
  },
  
  logging: {
    level: envVars.LOG_LEVEL as LogLevel,
    format: envVars.LOG_FORMAT as 'json' | 'pretty',
    enableConsole: envVars.LOG_ENABLE_CONSOLE,
    enableFile: envVars.LOG_ENABLE_FILE,
    filePath: envVars.LOG_FILE_PATH,
    fileMaxSize: envVars.LOG_FILE_MAX_SIZE,
    fileMaxFiles: envVars.LOG_FILE_MAX_FILES,
    disableColors: envVars.LOG_DISABLE_COLORS,
    redactFields: envVars.LOG_REDACT_FIELDS.split(','),
  },
  
  security: {
    jwt: {
      secret: envVars.JWT_SECRET,
      expiresIn: envVars.JWT_EXPIRES_IN,
      refreshExpiresIn: envVars.JWT_REFRESH_EXPIRES_IN,
    },
    cors: {
      origins: envVars.CORS_ORIGINS.split(','),
      methods: envVars.CORS_METHODS.split(','),
      allowedHeaders: envVars.CORS_ALLOWED_HEADERS.split(','),
      exposedHeaders: envVars.CORS_EXPOSED_HEADERS.split(','),
      credentials: envVars.CORS_CREDENTIALS,
      maxAge: envVars.CORS_MAX_AGE,
    },
    rateLimit: {
      enabled: envVars.RATE_LIMIT_ENABLED,
      maxRequests: envVars.RATE_LIMIT_MAX_REQUESTS,
      windowMs: envVars.RATE_LIMIT_WINDOW_MS,
      whitelist: envVars.RATE_LIMIT_WHITELIST.split(','),
    },
    csrf: {
      enabled: envVars.ENABLE_CSRF,
      tokenSecret: envVars.CSRF_TOKEN_SECRET,
    },
  },
  
  solana: {
    rpcEndpoint: envVars.SOLANA_RPC_ENDPOINT,
    network: envVars.SOLANA_NETWORK as 'mainnet-beta' | 'testnet' | 'devnet' | 'localnet',
    commitment: envVars.SOLANA_COMMITMENT,
    walletPrivateKey: envVars.SOLANA_WALLET_PRIVATE_KEY,
    connectionTimeoutMs: envVars.SOLANA_CONNECTION_TIMEOUT_MS,
  },
  
  database: {
    type: envVars.DB_TYPE as 'postgres' | 'mongodb' | 'mysql',
    host: envVars.DB_HOST,
    port: envVars.DB_PORT,
    name: envVars.DB_NAME,
    user: envVars.DB_USER,
    password: envVars.DB_PASSWORD,
    authSource: envVars.DB_AUTH_SOURCE,
    ssl: envVars.DB_SSL,
    uri: envVars.DB_URI,
    poolSize: envVars.DB_POOL_SIZE,
    keepAlive: envVars.DB_KEEP_ALIVE,
    connectTimeoutMs: envVars.DB_CONNECT_TIMEOUT_MS,
    retryWrites: envVars.DB_RETRY_WRITES,
    retryReads: envVars.DB_RETRY_READS,
  },
  
  redis: {
    enabled: envVars.REDIS_ENABLED,
    host: envVars.REDIS_HOST,
    port: envVars.REDIS_PORT,
    password: envVars.REDIS_PASSWORD,
    db: envVars.REDIS_DB,
    connectTimeout: envVars.REDIS_CONNECT_TIMEOUT,
    ttl: envVars.REDIS_TTL,
    cachePrefix: envVars.REDIS_CACHE_PREFIX,
  },
  
  monitoring: {
    enabled: envVars.MONITORING_ENABLED,
    sentry: {
      dsn: envVars.SENTRY_DSN,
      environment: envVars.SENTRY_ENVIRONMENT,
      release: envVars.SENTRY_RELEASE,
      tracesSampleRate: envVars.SENTRY_TRACES_SAMPLE_RATE,
    },
    prometheus: envVars.ENABLE_PROMETHEUS,
    newRelic: {
      licenseKey: envVars.NEW_RELIC_LICENSE_KEY,
      appName: envVars.NEW_RELIC_APP_NAME,
    },
  },
  
  websocket: {
    enabled: envVars.WEBSOCKET_ENABLED,
    path: envVars.WEBSOCKET_PATH,
    pingInterval: envVars.WEBSOCKET_PING_INTERVAL,
  },
  
  scheduler: {
    enabled: envVars.SCHEDULER_ENABLED,
    timezone: envVars.SCHEDULER_TIMEZONE,
  },
  
  storage: {
    type: envVars.STORAGE_TYPE as 'local' | 's3' | 'gcs',
    localPath: envVars.STORAGE_LOCAL_PATH,
    s3: {
      bucket: envVars.AWS_S3_BUCKET,
      region: envVars.AWS_S3_REGION,
      accessKeyId: envVars.AWS_ACCESS_KEY_ID,
      secretAccessKey: envVars.AWS_SECRET_ACCESS_KEY,
    },
    gcs: {
      bucket: envVars.GCS_BUCKET,
      projectId: envVars.GCS_PROJECT_ID,
      clientEmail: envVars.GCS_CLIENT_EMAIL,
      privateKey: envVars.GCS_PRIVATE_KEY,
    },
  },
  
  vault: {
    programId: envVars.VAULT_PROGRAM_ID,
    strategyExecutionInterval: envVars.STRATEGY_EXECUTION_INTERVAL,
    maxStrategyExecutionTime: envVars.MAX_STRATEGY_EXECUTION_TIME,
  },
  
  swagger: {
    enabled: envVars.ENABLE_SWAGGER,
  },
  
  featureFlags: {
    newStrategyEngine: envVars.FEATURE_FLAG_NEW_STRATEGY_ENGINE,
    advancedRiskMetrics: envVars.FEATURE_FLAG_ADVANCED_RISK_METRICS,
    multiChainSupport: envVars.FEATURE_FLAG_MULTI_CHAIN_SUPPORT,
  },
};

/**
 * Load configuration from environment variables
 * Call this function to reload configuration at runtime
 */
export function loadConfigFromEnvironment(): void {
  dotenv.config();
  const { value } = envVarsSchema.prefs({ errors: { label: 'key' } }).validate(process.env);
  
  // Update configuration with new values
  Object.assign(config, {
    isDevelopment: value.NODE_ENV === 'development',
    isProduction: value.NODE_ENV === 'production',
    isTest: value.NODE_ENV === 'test',
    isStaging: value.NODE_ENV === 'staging',
    nodeEnv: value.NODE_ENV,
    logLevel: value.LOG_LEVEL,
    
    // Update nested objects
    server: {
      port: value.PORT,
      host: value.HOST,
      apiPrefix: value.API_PREFIX,
      trustProxy: value.TRUST_PROXY,
      cluster: value.ENABLE_CLUSTER,
      workers: value.CLUSTER_WORKERS,
    },
    
    // Additional nested objects would be updated here...
  });
}

/**
 * Get database connection URI
 * @returns {string} Database connection URI
 */
export function getDatabaseUri(): string {
  if (config.database.uri) {
    return config.database.uri;
  }
  
  switch (config.database.type) {
    case 'mongodb':
      const credentials = config.database.user && config.database.password
        ? `${config.database.user}:${config.database.password}@`
        : '';
      const authSource = config.database.authSource
        ? `?authSource=${config.database.authSource}`
        : '';
      return `mongodb://${credentials}${config.database.host}:${config.database.port}/${config.database.name}${authSource}`;
    
    case 'postgres':
      return `postgresql://${config.database.user}:${config.database.password}@${config.database.host}:${config.database.port}/${config.database.name}`;
    
    case 'mysql':
      return `mysql://${config.database.user}:${config.database.password}@${config.database.host}:${config.database.port}/${config.database.name}`;
    
    default:
      throw new Error(`Unsupported database type: ${config.database.type}`);
  }
}

export default config;