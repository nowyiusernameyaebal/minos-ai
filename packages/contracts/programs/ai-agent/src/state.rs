// Minos-AI Agent - State Definitions
//
// This module defines the account state structures stored on-chain for the AI agent system.
// Each state struct represents a different type of account in the system, with their
// respective fields and associated metadata.

use anchor_lang::prelude::*;
use crate::models::{ModelMetadata, Signal, PerformanceMetrics};
use crate::constants::*;

/// Main agent account that manages the entire system
#[account]
#[derive(Default)]
pub struct AgentAccount {
    /// Authority allowed to administer the agent
    pub authority: Pubkey,
    
    /// Bump seed for PDA derivation
    pub bump: u8,
    
    /// Whether the agent is currently paused (emergency mode)
    pub paused: bool,
    
    /// Protocol fee basis points (e.g., 50 = 0.5%)
    pub protocol_fee_bps: u16,
    
    /// Address where protocol fees are sent
    pub fee_recipient: Pubkey,
    
    /// Minimum confidence threshold for signals (0-100)
    pub min_confidence_threshold: u8,
    
    /// Maximum signal validity period in seconds
    pub max_signal_validity_period: i64,
    
    /// Counter for model IDs
    pub next_model_id: u64,
    
    /// Counter for strategy IDs
    pub next_strategy_id: u64,
    
    /// Total number of active strategies
    pub strategies_count: u32,
    
    /// Total number of models registered
    pub models_count: u32,
    
    /// Unix timestamp of last fee collection
    pub last_fee_collection: i64,
    
    /// Maximum number of strategies allowed per model
    pub max_strategies_per_model: u16,
    
    /// Minimum amount required for strategy creation
    pub min_strategy_deposit: u64,
    
    /// Last time the agent configuration was updated
    pub last_update_time: i64,
    
    /// Version of the agent program
    pub version: u32,
    
    /// Reserved space for future extensions
    pub reserved: [u8; 64],
}

impl AgentAccount {
    pub const LEN: usize = 8 + // discriminator
        32 + // authority
        1 +  // bump
        1 +  // paused
        2 +  // protocol_fee_bps
        32 + // fee_recipient
        1 +  // min_confidence_threshold
        8 +  // max_signal_validity_period
        8 +  // next_model_id
        8 +  // next_strategy_id
        4 +  // strategies_count
        4 +  // models_count
        8 +  // last_fee_collection
        2 +  // max_strategies_per_model
        8 +  // min_strategy_deposit
        8 +  // last_update_time
        4 +  // version
        64;  // reserved
}

/// Parameters for initializing or updating the agent
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Default, Debug, PartialEq)]
pub struct AgentParams {
    /// Protocol fee basis points (e.g., 50 = 0.5%)
    pub protocol_fee_bps: u16,
    
    /// Address where protocol fees are sent
    pub fee_recipient: Pubkey,
    
    /// Minimum confidence threshold for signals (0-100)
    pub min_confidence_threshold: u8,
    
    /// Maximum signal validity period in seconds
    pub max_signal_validity_period: i64,
    
    /// Maximum number of strategies allowed per model
    pub max_strategies_per_model: u16,
    
    /// Minimum amount required for strategy creation
    pub min_strategy_deposit: u64,
}

/// AI model account storing metadata and verification info
#[account]
#[derive(Default)]
pub struct ModelAccount {
    /// Unique identifier for the model
    pub model_id: u64,
    
    /// Bump seed for PDA derivation
    pub bump: u8,
    
    /// Whether the model is currently active
    pub active: bool,
    
    /// Authority that registered the model
    pub authority: Pubkey,
    
    /// Address authorized to submit signals from this model
    pub authorized_submitter: Pubkey,
    
    /// Model metadata
    pub metadata: ModelMetadata,
    
    /// Number of strategies using this model
    pub strategies_count: u16,
    
    /// Counter for signal IDs
    pub next_signal_id: u64,
    
    /// Total number of signals submitted
    pub total_signals: u64,
    
    /// Total number of successful signals (profitable)
    pub successful_signals: u64,
    
    /// Registration timestamp
    pub registration_time: i64,
    
    /// Last update timestamp
    pub last_update_time: i64,
    
    /// Verification key for signal validation
    pub verification_key: [u8; 64],
    
    /// Latest performance metrics
    pub performance: PerformanceMetrics,
    
    /// Reserved space for future extensions
    pub reserved: [u8; 64],
}

impl ModelAccount {
    pub const LEN: usize = 8 + // discriminator
        8 +  // model_id
        1 +  // bump
        1 +  // active
        32 + // authority
        32 + // authorized_submitter
        ModelMetadata::LEN + // metadata
        2 +  // strategies_count
        8 +  // next_signal_id
        8 +  // total_signals
        8 +  // successful_signals
        8 +  // registration_time
        8 +  // last_update_time
        64 + // verification_key
        PerformanceMetrics::LEN + // performance
        64;  // reserved
}

/// Trading strategy account
#[account]
#[derive(Default)]
pub struct Strategy {
    /// Unique identifier for the strategy
    pub strategy_id: u64,
    
    /// Bump seed for PDA derivation
    pub bump: u8,
    
    /// Bump seed for vault PDA derivation
    pub vault_bump: u8,
    
    /// Whether the strategy is currently active
    pub active: bool,
    
    /// Owner of the strategy
    pub owner: Pubkey,
    
    /// Model ID associated with this strategy
    pub model_id: u64,
    
    /// Strategy parameters
    pub params: StrategyParams,
    
    /// Last update timestamp
    pub last_update_time: i64,
    
    /// Creation timestamp
    pub creation_time: i64,
    
    /// Mint of the token used in the strategy
    pub token_mint: Pubkey,
    
    /// Total fees collected
    pub total_fees_collected: u64,
    
    /// Initial deposit amount
    pub initial_deposit: u64,
    
    /// Total profit realized
    pub total_profit: i64,
    
    /// Performance fee basis points (e.g., 2000 = 20%)
    pub performance_fee_bps: u16,
    
    /// Last profit calculation timestamp
    pub last_profit_calculation: i64,
    
    /// High water mark for performance fees
    pub high_water_mark: u64,
    
    /// Total number of trades executed
    pub total_trades: u64,
    
    /// Total number of profitable trades
    pub profitable_trades: u64,
    
    /// Reserved space for future extensions
    pub reserved: [u8; 64],
}

impl Strategy {
    pub const LEN: usize = 8 + // discriminator
        8 +  // strategy_id
        1 +  // bump
        1 +  // vault_bump
        1 +  // active
        32 + // owner
        8 +  // model_id
        StrategyParams::LEN + // params
        8 +  // last_update_time
        8 +  // creation_time
        32 + // token_mint
        8 +  // total_fees_collected
        8 +  // initial_deposit
        8 +  // total_profit
        2 +  // performance_fee_bps
        8 +  // last_profit_calculation
        8 +  // high_water_mark
        8 +  // total_trades
        8 +  // profitable_trades
        64;  // reserved
}

/// Parameters for creating or updating a strategy
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Default, Debug, PartialEq)]
pub struct StrategyParams {
    /// Name of the strategy (max 32 bytes)
    pub name: [u8; 32],
    
    /// Risk level (1-5, where 1 is lowest risk)
    pub risk_level: u8,
    
    /// Maximum position size as percentage of portfolio (e.g., 50 = 50%)
    pub max_position_size_pct: u8,
    
    /// Maximum drawdown allowed as percentage (e.g., 20 = 20%)
    pub max_drawdown_pct: u8,
    
    /// ID of the model used for this strategy
    pub model_id: u64,
    
    /// Description (max 128 bytes)
    pub description: [u8; 128],
    
    /// Stop loss percentage (e.g., 10 = 10%)
    pub stop_loss_pct: u8,
    
    /// Take profit percentage (e.g., 20 = 20%)
    pub take_profit_pct: u8,
    
    /// Maximum leverage allowed (e.g., 2 = 2x)
    pub max_leverage: u8,
    
    /// Performance fee basis points (e.g., 2000 = 20%)
    pub performance_fee_bps: u16,
    
    /// Rebalance period in seconds
    pub rebalance_period: i64,
    
    /// Assets allowed for trading (bitmap)
    pub allowed_assets: u64,
    
    /// Reserved space for future extensions
    pub reserved: [u8; 32],
}

impl StrategyParams {
    pub const LEN: usize = 32 + // name
        1 +  // risk_level
        1 +  // max_position_size_pct
        1 +  // max_drawdown_pct
        8 +  // model_id
        128 + // description
        1 +  // stop_loss_pct
        1 +  // take_profit_pct
        1 +  // max_leverage
        2 +  // performance_fee_bps
        8 +  // rebalance_period
        8 +  // allowed_assets
        32;  // reserved
}

/// Signal account storing trading signals
#[account]
#[derive(Default)]
pub struct SignalAccount {
    /// Model ID that generated the signal
    pub model_id: u64,
    
    /// Signal ID
    pub signal_id: u64,
    
    /// Bump seed for PDA derivation
    pub bump: u8,
    
    /// Strategy ID this signal is for
    pub strategy_id: u64,
    
    /// Timestamp when the signal was created
    pub creation_time: i64,
    
    /// Timestamp when the signal expires
    pub expiry_time: i64,
    
    /// Whether the signal has been verified
    pub verified: bool,
    
    /// Whether the signal has been executed
    pub executed: bool,
    
    /// Result of execution (profit/loss amount)
    pub execution_result: i64,
    
    /// Timestamp of execution
    pub execution_time: i64,
    
    /// Address that submitted the signal
    pub submitter: Pubkey,
    
    /// Verification hash
    pub verification_hash: [u8; 32],
    
    /// Actual signal data
    pub signal: Signal,
    
    /// Reserved space for future extensions
    pub reserved: [u8; 64],
}

impl SignalAccount {
    pub const LEN: usize = 8 + // discriminator
        8 +  // model_id
        8 +  // signal_id
        1 +  // bump
        8 +  // strategy_id
        8 +  // creation_time
        8 +  // expiry_time
        1 +  // verified
        1 +  // executed
        8 +  // execution_result
        8 +  // execution_time
        32 + // submitter
        32 + // verification_hash
        Signal::LEN + // signal
        64;  // reserved
}

/// Performance metrics for a model or strategy
#[account]
#[derive(Default)]
pub struct PerformanceRecord {
    /// ID of the model or strategy
    pub id: u64,
    
    /// Bump seed for PDA derivation
    pub bump: u8,
    
    /// Type (0 = model, 1 = strategy)
    pub record_type: u8,
    
    /// Timestamp of the record
    pub timestamp: i64,
    
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    
    /// Reserved space for future extensions
    pub reserved: [u8; 32],
}

impl PerformanceRecord {
    pub const LEN: usize = 8 + // discriminator
        8 +  // id
        1 +  // bump
        1 +  // record_type
        8 +  // timestamp
        PerformanceMetrics::LEN + // metrics
        32;  // reserved
}