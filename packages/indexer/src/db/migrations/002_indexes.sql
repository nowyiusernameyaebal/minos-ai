-- =====================================================
-- Minos-AI DeFi Indexer - Database Indexes Migration
-- Version: 002
-- Description: Create comprehensive indexes for optimal query performance
-- =====================================================

-- Drop existing indexes if they exist (for migration rollback/re-run)
DROP INDEX IF EXISTS idx_blocks_number;
DROP INDEX IF EXISTS idx_blocks_timestamp;
DROP INDEX IF EXISTS idx_blocks_hash;

DROP INDEX IF EXISTS idx_transactions_hash;
DROP INDEX IF EXISTS idx_transactions_block_number;
DROP INDEX IF EXISTS idx_transactions_from;
DROP INDEX IF EXISTS idx_transactions_to;
DROP INDEX IF EXISTS idx_transactions_timestamp;
DROP INDEX IF EXISTS idx_transactions_status;

DROP INDEX IF EXISTS idx_logs_transaction_hash;
DROP INDEX IF EXISTS idx_logs_address;
DROP INDEX IF EXISTS idx_logs_topic0;
DROP INDEX IF EXISTS idx_logs_block_number;
DROP INDEX IF EXISTS idx_logs_log_index;

DROP INDEX IF EXISTS idx_protocol_events_protocol;
DROP INDEX IF EXISTS idx_protocol_events_event_type;
DROP INDEX IF EXISTS idx_protocol_events_block_number;
DROP INDEX IF EXISTS idx_protocol_events_timestamp;
DROP INDEX IF EXISTS idx_protocol_events_user_address;

DROP INDEX IF EXISTS idx_token_transfers_token_address;
DROP INDEX IF EXISTS idx_token_transfers_from;
DROP INDEX IF EXISTS idx_token_transfers_to;
DROP INDEX IF EXISTS idx_token_transfers_amount;
DROP INDEX IF EXISTS idx_token_transfers_timestamp;

DROP INDEX IF EXISTS idx_defi_positions_user_address;
DROP INDEX IF EXISTS idx_defi_positions_protocol;
DROP INDEX IF EXISTS idx_defi_positions_pool_address;
DROP INDEX IF EXISTS idx_defi_positions_timestamp;
DROP INDEX IF EXISTS idx_defi_positions_is_active;

DROP INDEX IF EXISTS idx_protocol_metrics_protocol;
DROP INDEX IF EXISTS idx_protocol_metrics_timestamp;
DROP INDEX IF EXISTS idx_protocol_metrics_tvl;

DROP INDEX IF EXISTS idx_user_portfolios_user_address;
DROP INDEX IF EXISTS idx_user_portfolios_created_at;
DROP INDEX IF EXISTS idx_user_portfolios_is_active;

DROP INDEX IF EXISTS idx_alerts_user_address;
DROP INDEX IF EXISTS idx_alerts_is_triggered;
DROP INDEX IF EXISTS idx_alerts_created_at;

-- =====================================================
-- PRIMARY INDEXES FOR CORE TABLES
-- =====================================================

-- Blocks table indexes
CREATE INDEX idx_blocks_number ON blocks(block_number);
CREATE INDEX idx_blocks_timestamp ON blocks(timestamp);
CREATE INDEX idx_blocks_hash ON blocks(block_hash);
CREATE INDEX idx_blocks_composite ON blocks(block_number, timestamp);

-- Transactions table indexes
CREATE INDEX idx_transactions_hash ON transactions(transaction_hash);
CREATE INDEX idx_transactions_block_number ON transactions(block_number);
CREATE INDEX idx_transactions_from ON transactions(from_address);
CREATE INDEX idx_transactions_to ON transactions(to_address);
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX idx_transactions_status ON transactions(status);
CREATE INDEX idx_transactions_composite_from ON transactions(from_address, block_number);
CREATE INDEX idx_transactions_composite_to ON transactions(to_address, block_number);

-- Logs table indexes
CREATE INDEX idx_logs_transaction_hash ON logs(transaction_hash);
CREATE INDEX idx_logs_address ON logs(address);
CREATE INDEX idx_logs_topic0 ON logs(topic0);
CREATE INDEX idx_logs_block_number ON logs(block_number);
CREATE INDEX idx_logs_log_index ON logs(log_index);
CREATE INDEX idx_logs_composite_addr_topic ON logs(address, topic0);
CREATE INDEX idx_logs_composite_tx_log_idx ON logs(transaction_hash, log_index);

-- =====================================================
-- PROTOCOL-SPECIFIC INDEXES
-- =====================================================

-- Protocol events indexes
CREATE INDEX idx_protocol_events_protocol ON protocol_events(protocol);
CREATE INDEX idx_protocol_events_event_type ON protocol_events(event_type);
CREATE INDEX idx_protocol_events_block_number ON protocol_events(block_number);
CREATE INDEX idx_protocol_events_timestamp ON protocol_events(timestamp);
CREATE INDEX idx_protocol_events_user_address ON protocol_events(user_address);
CREATE INDEX idx_protocol_events_composite_protocol_time ON protocol_events(protocol, timestamp);
CREATE INDEX idx_protocol_events_composite_user_protocol ON protocol_events(user_address, protocol);
CREATE INDEX idx_protocol_events_composite_event_time ON protocol_events(event_type, timestamp);

-- Token transfers indexes
CREATE INDEX idx_token_transfers_token_address ON token_transfers(token_address);
CREATE INDEX idx_token_transfers_from ON token_transfers(from_address);
CREATE INDEX idx_token_transfers_to ON token_transfers(to_address);
CREATE INDEX idx_token_transfers_amount ON token_transfers(amount);
CREATE INDEX idx_token_transfers_timestamp ON token_transfers(timestamp);
CREATE INDEX idx_token_transfers_composite_token_from ON token_transfers(token_address, from_address);
CREATE INDEX idx_token_transfers_composite_token_to ON token_transfers(token_address, to_address);
CREATE INDEX idx_token_transfers_composite_token_time ON token_transfers(token_address, timestamp);

-- DeFi positions indexes
CREATE INDEX idx_defi_positions_user_address ON defi_positions(user_address);
CREATE INDEX idx_defi_positions_protocol ON defi_positions(protocol);
CREATE INDEX idx_defi_positions_pool_address ON defi_positions(pool_address);
CREATE INDEX idx_defi_positions_timestamp ON defi_positions(timestamp);
CREATE INDEX idx_defi_positions_is_active ON defi_positions(is_active);
CREATE INDEX idx_defi_positions_composite_user_protocol ON defi_positions(user_address, protocol);
CREATE INDEX idx_defi_positions_composite_user_active ON defi_positions(user_address, is_active);
CREATE INDEX idx_defi_positions_composite_protocol_pool ON defi_positions(protocol, pool_address);

-- =====================================================
-- ANALYTICAL INDEXES FOR AGGREGATIONS
-- =====================================================

-- Protocol metrics indexes for time-series queries
CREATE INDEX idx_protocol_metrics_protocol ON protocol_metrics(protocol);
CREATE INDEX idx_protocol_metrics_timestamp ON protocol_metrics(timestamp);
CREATE INDEX idx_protocol_metrics_tvl ON protocol_metrics(tvl);
CREATE INDEX idx_protocol_metrics_composite_protocol_time ON protocol_metrics(protocol, timestamp);
CREATE INDEX idx_protocol_metrics_composite_time_tvl ON protocol_metrics(timestamp, tvl);

-- User portfolios indexes
CREATE INDEX idx_user_portfolios_user_address ON user_portfolios(user_address);
CREATE INDEX idx_user_portfolios_created_at ON user_portfolios(created_at);
CREATE INDEX idx_user_portfolios_updated_at ON user_portfolios(updated_at);
CREATE INDEX idx_user_portfolios_is_active ON user_portfolios(is_active);
CREATE INDEX idx_user_portfolios_composite_user_active ON user_portfolios(user_address, is_active);

-- Alerts indexes
CREATE INDEX idx_alerts_user_address ON alerts(user_address);
CREATE INDEX idx_alerts_is_triggered ON alerts(is_triggered);
CREATE INDEX idx_alerts_created_at ON alerts(created_at);
CREATE INDEX idx_alerts_alert_type ON alerts(alert_type);
CREATE INDEX idx_alerts_composite_user_type ON alerts(user_address, alert_type);
CREATE INDEX idx_alerts_composite_user_triggered ON alerts(user_address, is_triggered);

-- =====================================================
-- SPECIALIZED INDEXES FOR COMPLEX QUERIES
-- =====================================================

-- Covering indexes for frequent read patterns
CREATE INDEX idx_transactions_covering_from ON transactions(from_address) 
  INCLUDE (transaction_hash, block_number, timestamp, value, gas_used);

CREATE INDEX idx_transactions_covering_to ON transactions(to_address) 
  INCLUDE (transaction_hash, block_number, timestamp, value, gas_used);

CREATE INDEX idx_logs_covering_address ON logs(address) 
  INCLUDE (transaction_hash, topic0, topic1, topic2, topic3, data);

-- Partial indexes for active records only
CREATE INDEX idx_defi_positions_active_only ON defi_positions(user_address, protocol) 
  WHERE is_active = true;

CREATE INDEX idx_user_portfolios_active_only ON user_portfolios(user_address) 
  WHERE is_active = true;

CREATE INDEX idx_alerts_untriggered_only ON alerts(user_address, alert_type) 
  WHERE is_triggered = false;

-- =====================================================
-- FUNCTIONAL INDEXES FOR SPECIAL QUERIES
-- =====================================================

-- Index for case-insensitive address searches
CREATE INDEX idx_transactions_from_lower ON transactions(LOWER(from_address));
CREATE INDEX idx_transactions_to_lower ON transactions(LOWER(to_address));

-- Index for JSON data queries (if using JSONB columns)
CREATE INDEX IF NOT EXISTS idx_protocol_events_metadata_gin ON protocol_events 
  USING gin(metadata) WHERE metadata IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_defi_positions_position_data_gin ON defi_positions 
  USING gin(position_data) WHERE position_data IS NOT NULL;

-- Expression indexes for common calculations
CREATE INDEX idx_token_transfers_amount_usd ON token_transfers((amount * price_usd)) 
  WHERE price_usd IS NOT NULL;

-- =====================================================
-- TIME-SERIES SPECIFIC INDEXES
-- =====================================================

-- BRIN indexes for time-series data (more efficient for large sequential data)
CREATE INDEX idx_blocks_timestamp_brin ON blocks USING brin(timestamp);
CREATE INDEX idx_transactions_timestamp_brin ON transactions USING brin(timestamp);
CREATE INDEX idx_logs_timestamp_brin ON logs USING brin(timestamp);
CREATE INDEX idx_protocol_events_timestamp_brin ON protocol_events USING brin(timestamp);

-- =====================================================
-- INDEXES FOR SOCIAL SENTIMENT DATA
-- =====================================================

-- Social sentiment indexes
CREATE INDEX IF NOT EXISTS idx_social_posts_protocol ON social_posts(protocol);
CREATE INDEX IF NOT EXISTS idx_social_posts_timestamp ON social_posts(timestamp);
CREATE INDEX IF NOT EXISTS idx_social_posts_sentiment ON social_posts(sentiment_score);
CREATE INDEX IF NOT EXISTS idx_social_posts_composite_protocol_time ON social_posts(protocol, timestamp);

CREATE INDEX IF NOT EXISTS idx_social_metrics_protocol ON social_metrics(protocol);
CREATE INDEX IF NOT EXISTS idx_social_metrics_timestamp ON social_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_social_metrics_composite ON social_metrics(protocol, timestamp);

-- =====================================================
-- FOREIGN KEY INDEXES (for referential integrity performance)
-- =====================================================

-- Ensure all foreign key columns have indexes
CREATE INDEX IF NOT EXISTS idx_transactions_fk_block ON transactions(block_number);
CREATE INDEX IF NOT EXISTS idx_logs_fk_transaction ON logs(transaction_hash);
CREATE INDEX IF NOT EXISTS idx_protocol_events_fk_transaction ON protocol_events(transaction_hash);
CREATE INDEX IF NOT EXISTS idx_token_transfers_fk_transaction ON token_transfers(transaction_hash);

-- =====================================================
-- MAINTENANCE AND STATISTICS
-- =====================================================

-- Update table statistics for query planner
ANALYZE blocks;
ANALYZE transactions;
ANALYZE logs;
ANALYZE protocol_events;
ANALYZE token_transfers;
ANALYZE defi_positions;
ANALYZE protocol_metrics;
ANALYZE user_portfolios;
ANALYZE alerts;

-- =====================================================
-- INDEX USAGE MONITORING VIEWS
-- =====================================================

-- View to monitor index usage (PostgreSQL specific)
CREATE OR REPLACE VIEW index_usage_stats AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_fetch,
    idx_tup_read,
    CASE WHEN idx_tup_read > 0 
        THEN (idx_tup_fetch::float / idx_tup_read::float * 100)::int 
        ELSE 0 
    END AS hit_rate_pct
FROM pg_stat_user_indexes
ORDER BY schemaname, tablename, indexname;

-- View to find unused indexes
CREATE OR REPLACE VIEW unused_indexes AS
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
AND idx_tup_read = 0
AND idx_tup_fetch = 0
ORDER BY pg_relation_size(indexrelid) DESC;

-- =====================================================
-- CUSTOM FUNCTIONS FOR INDEX MAINTENANCE
-- =====================================================

-- Function to rebuild all indexes concurrently
CREATE OR REPLACE FUNCTION rebuild_all_indexes_concurrently()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    rec record;
    sql_cmd text;
BEGIN
    -- Get all user-defined indexes
    FOR rec IN 
        SELECT schemaname, tablename, indexname
        FROM pg_indexes 
        WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
        AND indexname NOT LIKE '%_pkey'
    LOOP
        sql_cmd := 'REINDEX INDEX CONCURRENTLY ' || quote_ident(rec.schemaname) || '.' || quote_ident(rec.indexname);
        
        BEGIN
            EXECUTE sql_cmd;
            RAISE NOTICE 'Rebuilt index: %', rec.indexname;
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'Failed to rebuild index %: %', rec.indexname, SQLERRM;
        END;
    END LOOP;
END;
$$;

-- Function to get index bloat information
CREATE OR REPLACE FUNCTION get_index_bloat_info()
RETURNS TABLE(
    schema_name text,
    table_name text,
    index_name text,
    bloat_pct numeric,
    real_size text,
    extra_size text
)
LANGUAGE sql
AS $$
    SELECT 
        schemaname::text,
        tablename::text,
        indexname::text,
        CASE WHEN pg_relation_size(indexrelname::regclass) > 0
            THEN round(((pg_relation_size(indexrelname::regclass) - pg_relation_size(tablename::regclass::text || '_' || indexname || '_idx')) * 100.0) / pg_relation_size(indexrelname::regclass), 2)
            ELSE 0
        END,
        pg_size_pretty(pg_relation_size(indexrelname::regclass)),
        pg_size_pretty(pg_relation_size(indexrelname::regclass) - pg_relation_size(tablename::regclass::text || '_' || indexname || '_idx'))
    FROM pg_stat_user_indexes
    WHERE pg_relation_size(indexrelname::regclass) > 0
    ORDER BY pg_relation_size(indexrelname::regclass) DESC;
$$;

-- =====================================================
-- COMMENTS FOR DOCUMENTATION
-- =====================================================

COMMENT ON INDEX idx_blocks_composite IS 'Composite index for block queries by number and timestamp';
COMMENT ON INDEX idx_transactions_composite_from IS 'Composite index for transaction queries from specific address';
COMMENT ON INDEX idx_logs_composite_addr_topic IS 'Composite index for log queries by address and topic';
COMMENT ON INDEX idx_protocol_events_composite_protocol_time IS 'Composite index for protocol events time-series queries';

-- =====================================================
-- Migration completion log
-- =====================================================

INSERT INTO schema_migrations (version, description, executed_at)
VALUES (
    '002',
    'Created comprehensive indexes for optimal query performance',
    NOW()
)
ON CONFLICT (version) DO UPDATE SET
    description = EXCLUDED.description,
    executed_at = EXCLUDED.executed_at;