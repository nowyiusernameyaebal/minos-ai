-- Minos-AI Initial Database Migration
-- Creates the core schema for the indexer service

BEGIN;

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Set up schema
SET search_path TO public;

-- Create enum types
CREATE TYPE signal_type AS ENUM ('buy', 'sell', 'hold');
CREATE TYPE trade_status AS ENUM ('profit', 'loss', 'neutral');
CREATE TYPE vault_transaction_type AS ENUM ('deposit', 'withdrawal', 'performance_fee', 'protocol_fee');
CREATE TYPE model_type AS ENUM ('ariadne', 'androgeus', 'deucalion');

-- Create users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    address VARCHAR(44) NOT NULL UNIQUE,
    name VARCHAR(100),
    email VARCHAR(100),
    first_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    total_deposits BIGINT DEFAULT 0,
    total_withdrawals BIGINT DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create models table
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    version VARCHAR(20) NOT NULL,
    model_type model_type NOT NULL,
    creator_address VARCHAR(44) NOT NULL,
    submitter_address VARCHAR(44) NOT NULL,
    total_signals BIGINT DEFAULT 0,
    successful_signals BIGINT DEFAULT 0,
    win_rate DECIMAL(5,2) DEFAULT 0,
    sharpe_ratio DECIMAL(10,4),
    sortino_ratio DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    verification_key TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    registration_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    last_update_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_models_creator
        FOREIGN KEY (creator_address)
        REFERENCES users (address)
        ON DELETE SET NULL
);

-- Create strategies table
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    model_id VARCHAR(20) NOT NULL,
    owner_address VARCHAR(44) NOT NULL,
    risk_level SMALLINT NOT NULL,
    max_position_size_pct SMALLINT NOT NULL,
    max_drawdown_pct SMALLINT NOT NULL,
    stop_loss_pct SMALLINT,
    take_profit_pct SMALLINT,
    max_leverage SMALLINT,
    performance_fee_bps SMALLINT NOT NULL,
    rebalance_period BIGINT,
    allowed_assets BIGINT,
    vault_address VARCHAR(44),
    token_mint VARCHAR(44),
    total_deposits BIGINT DEFAULT 0,
    total_withdrawals BIGINT DEFAULT 0,
    total_fees_collected BIGINT DEFAULT 0,
    total_profit BIGINT DEFAULT 0,
    current_balance BIGINT DEFAULT 0,
    total_trades BIGINT DEFAULT 0,
    profitable_trades BIGINT DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    creation_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    last_update_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_strategies_model
        FOREIGN KEY (model_id)
        REFERENCES models (model_id)
        ON DELETE CASCADE,
        
    CONSTRAINT fk_strategies_owner
        FOREIGN KEY (owner_address)
        REFERENCES users (address)
        ON DELETE SET NULL
);

-- Create signals table
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(20) NOT NULL,
    signal_id VARCHAR(20) NOT NULL,
    strategy_id VARCHAR(20) NOT NULL,
    submitter VARCHAR(44) NOT NULL,
    asset_pair VARCHAR(20),
    signal_type signal_type,
    confidence_score SMALLINT,
    price DECIMAL(24,8),
    target_price DECIMAL(24,8),
    stop_loss_price DECIMAL(24,8),
    leverage DECIMAL(5,2),
    position_size DECIMAL(10,4),
    signal_data JSONB,
    executed BOOLEAN DEFAULT FALSE,
    verified BOOLEAN DEFAULT FALSE,
    execution_result BIGINT,
    verification_hash VARCHAR(64),
    transaction_signature VARCHAR(88) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    expiry_timestamp TIMESTAMP WITH TIME ZONE,
    execution_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT signals_unique_composite
        UNIQUE (model_id, signal_id),
        
    CONSTRAINT fk_signals_model
        FOREIGN KEY (model_id)
        REFERENCES models (model_id)
        ON DELETE CASCADE,
        
    CONSTRAINT fk_signals_strategy
        FOREIGN KEY (strategy_id)
        REFERENCES strategies (strategy_id)
        ON DELETE CASCADE,
        
    CONSTRAINT fk_signals_submitter
        FOREIGN KEY (submitter)
        REFERENCES users (address)
        ON DELETE SET NULL
);

-- Create trades table
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(20) NOT NULL,
    signal_id VARCHAR(20) NOT NULL,
    strategy_id VARCHAR(20) NOT NULL,
    executor VARCHAR(44) NOT NULL,
    asset_pair VARCHAR(20),
    position_size DECIMAL(10,4),
    entry_price DECIMAL(24,8),
    execution_price DECIMAL(24,8),
    exit_price DECIMAL(24,8),
    leverage DECIMAL(5,2),
    execution_result BIGINT,
    status trade_status NOT NULL,
    transaction_signature VARCHAR(88) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    exit_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_trades_signal
        FOREIGN KEY (model_id, signal_id)
        REFERENCES signals (model_id, signal_id)
        ON DELETE CASCADE,
        
    CONSTRAINT fk_trades_strategy
        FOREIGN KEY (strategy_id)
        REFERENCES strategies (strategy_id)
        ON DELETE CASCADE,
        
    CONSTRAINT fk_trades_executor
        FOREIGN KEY (executor)
        REFERENCES users (address)
        ON DELETE SET NULL
);

-- Create vault transactions table
CREATE TABLE vault_transactions (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(20) NOT NULL,
    user_address VARCHAR(44) NOT NULL,
    amount BIGINT NOT NULL,
    transaction_type vault_transaction_type NOT NULL,
    transaction_signature VARCHAR(88) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_vault_txn_strategy
        FOREIGN KEY (strategy_id)
        REFERENCES strategies (strategy_id)
        ON DELETE CASCADE,
        
    CONSTRAINT fk_vault_txn_user
        FOREIGN KEY (user_address)
        REFERENCES users (address)
        ON DELETE SET NULL
);

-- Create performance records table
CREATE TABLE performance_records (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(10) NOT NULL, -- 'model' or 'strategy'
    entity_id VARCHAR(20) NOT NULL,
    total_signals INTEGER,
    successful_signals INTEGER,
    win_rate DECIMAL(5,2),
    profit_factor DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    sortino_ratio DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    avg_trade_duration DECIMAL(10,4),
    avg_profit_per_trade DECIMAL(24,8),
    volatility DECIMAL(10,4),
    performance_data JSONB,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT performance_records_unique_composite
        UNIQUE (entity_type, entity_id, timestamp)
);

-- Create system status table
CREATE TABLE system_status (
    id SERIAL PRIMARY KEY,
    component VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    last_block_processed BIGINT,
    last_signature_processed VARCHAR(88),
    last_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    message TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT system_status_component_unique
        UNIQUE (component)
);

-- Create indexes
CREATE INDEX idx_models_model_id ON models (model_id);
CREATE INDEX idx_models_creator ON models (creator_address);
CREATE INDEX idx_models_active ON models (is_active);
CREATE INDEX idx_models_type ON models (model_type);

CREATE INDEX idx_strategies_strategy_id ON strategies (strategy_id);
CREATE INDEX idx_strategies_model_id ON strategies (model_id);
CREATE INDEX idx_strategies_owner ON strategies (owner_address);
CREATE INDEX idx_strategies_active ON strategies (is_active);
CREATE INDEX idx_strategies_risk ON strategies (risk_level);
CREATE INDEX idx_strategies_vault ON strategies (vault_address);

CREATE INDEX idx_signals_composite ON signals (model_id, signal_id);
CREATE INDEX idx_signals_strategy ON signals (strategy_id);
CREATE INDEX idx_signals_type ON signals (signal_type);
CREATE INDEX idx_signals_executed ON signals (executed);
CREATE INDEX idx_signals_timestamp ON signals (timestamp);
CREATE INDEX idx_signals_transaction ON signals (transaction_signature);

CREATE INDEX idx_trades_signal ON trades (model_id, signal_id);
CREATE INDEX idx_trades_strategy ON trades (strategy_id);
CREATE INDEX idx_trades_status ON trades (status);
CREATE INDEX idx_trades_timestamp ON trades (timestamp);
CREATE INDEX idx_trades_transaction ON trades (transaction_signature);

CREATE INDEX idx_vault_txn_strategy ON vault_transactions (strategy_id);
CREATE INDEX idx_vault_txn_user ON vault_transactions (user_address);
CREATE INDEX idx_vault_txn_type ON vault_transactions (transaction_type);
CREATE INDEX idx_vault_txn_timestamp ON vault_transactions (timestamp);
CREATE INDEX idx_vault_txn_transaction ON vault_transactions (transaction_signature);

CREATE INDEX idx_performance_entity ON performance_records (entity_type, entity_id);
CREATE INDEX idx_performance_timestamp ON performance_records (timestamp);

-- Create JSONB indexes for query performance
CREATE INDEX idx_signals_data_gin ON signals USING GIN (signal_data);
CREATE INDEX idx_performance_data_gin ON performance_records USING GIN (performance_data);

-- Create text search indexes
CREATE INDEX idx_strategies_name_trgm ON strategies USING GIN (name gin_trgm_ops);
CREATE INDEX idx_strategies_description_trgm ON strategies USING GIN (description gin_trgm_ops);
CREATE INDEX idx_models_name_trgm ON models USING GIN (name gin_trgm_ops);
CREATE INDEX idx_models_description_trgm ON models USING GIN (description gin_trgm_ops);

-- Initial system status data
INSERT INTO system_status (component, status, last_block_processed, last_timestamp, created_at, updated_at)
VALUES
    ('trade_processor', 'initialized', 0, NOW(), NOW(), NOW()),
    ('vault_processor', 'initialized', 0, NOW(), NOW(), NOW()),
    ('market_processor', 'initialized', 0, NOW(), NOW(), NOW());

COMMIT;