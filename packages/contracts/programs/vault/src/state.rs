//! State definitions for the Minos Vault program
//!
//! This module contains all the account structures that persist data 
//! for the vault program.

use anchor_lang::prelude::*;
use solana_program::pubkey::Pubkey;
use std::collections::VecDeque;

/// Maximum number of strategies a vault can have
pub const MAX_STRATEGIES: usize = 10;
/// Maximum number of assets in a single strategy
pub const MAX_ASSETS_PER_STRATEGY: usize = 20;
/// Maximum length for vault name
pub const MAX_VAULT_NAME_LENGTH: usize = 32;
/// Maximum length for vault description
pub const MAX_VAULT_DESCRIPTION_LENGTH: usize = 200;
/// Maximum length for strategy description
pub const MAX_STRATEGY_DESCRIPTION_LENGTH: usize = 100;
/// Maximum history entries to keep
pub const MAX_HISTORY_ENTRIES: usize = 30;
/// Maximum length for user tags
pub const MAX_USER_TAG_LENGTH: usize = 20;
/// Maximum number of user tags
pub const MAX_USER_TAGS: usize = 5;

/// Main vault account that stores the vault's configuration and state
#[account]
#[derive(Debug)]
pub struct Vault {
    /// The authority that can manage the vault configuration
    pub authority: Pubkey,
    
    /// The strategy manager that can execute strategies
    pub strategy_manager: Pubkey,
    
    /// The recipient of fees
    pub fee_recipient: Pubkey,
    
    /// The mint of the asset controlled by this vault
    pub asset_mint: Pubkey,
    
    /// The mint for the vault shares
    pub share_mint: Pubkey,
    
    /// The name of the vault
    pub vault_name: String,
    
    /// Description of vault purpose and strategy
    pub description: String,
    
    /// Fee configuration for the vault
    pub fee_config: FeeConfig,
    
    /// Risk profile (1-5, where 1 is lowest risk)
    pub risk_profile: u8,
    
    /// Total value locked at last update
    pub tvl: u64,
    
    /// Maximum capacity of the vault (0 means unlimited)
    pub capacity: u64,
    
    /// Strategy execution frequency in seconds
    pub strategy_execution_frequency: u64,
    
    /// Last strategy execution timestamp
    pub last_strategy_execution: i64,
    
    /// Minimum deposit amount
    pub min_deposit: u64,
    
    /// Maximum deposit amount per user (0 means unlimited)
    pub max_deposit_per_user: u64,
    
    /// Lockup period in seconds (0 means no lockup)
    pub lockup_period: u64,
    
    /// Is the vault paused?
    pub is_paused: bool,
    
    /// Vault creation timestamp
    pub created_at: i64,
    
    /// PDA bump for the vault
    pub bump: u8,
    
    /// PDA bump for the treasury
    pub treasury_bump: u8,
    
    /// PDA bump for the share mint
    pub share_mint_bump: u8,
    
    /// Current active strategy ID
    pub active_strategy_id: u64,
    
    /// Historical performance data
    pub performance_history: VaultPerformanceHistory,
    
    /// Cumulative fees collected
    pub cumulative_fees: u64,
    
    /// Total deposits ever made
    pub total_deposits: u64,
    
    /// Total withdrawals ever made
    pub total_withdrawals: u64,
    
    /// Total unique users
    pub total_users: u32,
    
    /// Total number of strategy executions
    pub total_strategy_executions: u32,
    
    /// Marks if AI optimization is enabled
    pub ai_optimization_enabled: bool,
    
    /// ID of the AI model being used (0 if none)
    pub ai_model_id: u16,
    
    /// How often to update AI model recommendations (in seconds)
    pub ai_update_frequency: u64,
    
    /// Last time AI recommendations were updated
    pub last_ai_update: i64,
    
    /// Reserved space for future upgrades
    pub reserved: [u8; 64],
}

impl Vault {
    /// Calculate the space required for the vault account
    pub fn space() -> usize {
        8 +  // discriminator
        32 + // authority
        32 + // strategy_manager
        32 + // fee_recipient
        32 + // asset_mint
        32 + // share_mint
        4 + MAX_VAULT_NAME_LENGTH + // vault_name (string with len prefix)
        4 + MAX_VAULT_DESCRIPTION_LENGTH + // description (string with len prefix)
        FeeConfig::space() + // fee_config
        1 + // risk_profile
        8 + // tvl
        8 + // capacity
        8 + // strategy_execution_frequency
        8 + // last_strategy_execution
        8 + // min_deposit
        8 + // max_deposit_per_user
        8 + // lockup_period
        1 + // is_paused
        8 + // created_at
        1 + // bump
        1 + // treasury_bump
        1 + // share_mint_bump
        8 + // active_strategy_id
        VaultPerformanceHistory::space() + // performance_history
        8 + // cumulative_fees
        8 + // total_deposits
        8 + // total_withdrawals
        4 + // total_users
        4 + // total_strategy_executions
        1 + // ai_optimization_enabled
        2 + // ai_model_id
        8 + // ai_update_frequency
        8 + // last_ai_update
        64   // reserved
    }
}

/// Fee configuration for the vault
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Default, PartialEq)]
pub struct FeeConfig {
    /// Management fee in basis points (annual, applied continuously)
    pub management_fee_bps: u16,
    
    /// Performance fee in basis points (applied on profit)
    pub performance_fee_bps: u16,
    
    /// Withdrawal fee in basis points
    pub withdrawal_fee_bps: u16,
    
    /// Deposit fee in basis points
    pub deposit_fee_bps: u16,
    
    /// Threshold for high-water mark (in basis points over previous)
    pub hwm_threshold_bps: u16,
    
    /// Last time management fees were collected
    pub last_management_fee_collection: i64,
    
    /// High water mark value per share (for performance fee calculation)
    pub high_water_mark: u64,
}

impl FeeConfig {
    /// Calculate the space required for the fee config
    pub fn space() -> usize {
        2 + // management_fee_bps
        2 + // performance_fee_bps
        2 + // withdrawal_fee_bps
        2 + // deposit_fee_bps
        2 + // hwm_threshold_bps
        8 + // last_management_fee_collection
        8   // high_water_mark
    }
}

/// Historical performance of the vault
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Default, PartialEq)]
pub struct VaultPerformanceHistory {
    /// Historical NAV values
    pub nav_history: VecDeque<HistoryEntry>,
    
    /// Returns in basis points by time period
    pub returns: Returns,
    
    /// Volatility measures
    pub volatility: Volatility,
    
    /// Drawdown information
    pub max_drawdown_bps: u16,
    
    /// Sharpe ratio (multiplied by X100 for precision)
    pub sharpe_ratio_x100: i32,
}

impl VaultPerformanceHistory {
    /// Calculate the space required for the performance history
    pub fn space() -> usize {
        4 + (HistoryEntry::space() * MAX_HISTORY_ENTRIES) + // nav_history (vec with len prefix)
        Returns::space() + // returns
        Volatility::space() + // volatility
        2 + // max_drawdown_bps
        4   // sharpe_ratio_x100
    }
}

/// A single historical entry
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Default, PartialEq)]
pub struct HistoryEntry {
    /// Timestamp of the entry
    pub timestamp: i64,
    
    /// NAV value in the asset's smallest unit
    pub nav: u64,
    
    /// Share supply at this point
    pub share_supply: u64,
    
    /// TVL at this point
    pub tvl: u64,
}

impl HistoryEntry {
    /// Calculate the space required for a history entry
    pub fn space() -> usize {
        8 + // timestamp
        8 + // nav
        8 + // share_supply
        8   // tvl
    }
}

/// Performance returns by time period
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Default, PartialEq)]
pub struct Returns {
    /// Daily return in basis points
    pub daily_bps: i16,
    
    /// Weekly return in basis points
    pub weekly_bps: i16,
    
    /// Monthly return in basis points
    pub monthly_bps: i16,
    
    /// Quarterly return in basis points
    pub quarterly_bps: i16,
    
    /// Yearly return in basis points
    pub yearly_bps: i16,
    
    /// Return since inception in basis points
    pub inception_bps: i16,
}

impl Returns {
    /// Calculate the space required for returns
    pub fn space() -> usize {
        2 + // daily_bps
        2 + // weekly_bps
        2 + // monthly_bps
        2 + // quarterly_bps
        2 + // yearly_bps
        2   // inception_bps
    }
}

/// Volatility measures
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Default, PartialEq)]
pub struct Volatility {
    /// Daily volatility in basis points
    pub daily_bps: u16,
    
    /// Weekly volatility in basis points
    pub weekly_bps: u16,
    
    /// Monthly volatility in basis points
    pub monthly_bps: u16,
    
    /// Annualized volatility in basis points
    pub annualized_bps: u16,
}

impl Volatility {
    /// Calculate the space required for volatility
    pub fn space() -> usize {
        2 + // daily_bps
        2 + // weekly_bps
        2 + // monthly_bps
        2   // annualized_bps
    }
}

/// Strategy information
#[account]
#[derive(Debug)]
pub struct Strategy {
    /// Vault this strategy belongs to
    pub vault: Pubkey,
    
    /// Unique identifier for this strategy
    pub strategy_id: u64,
    
    /// Description of the strategy
    pub description: String,
    
    /// Strategy type identifier
    pub strategy_type: u8,
    
    /// Risk profile (1-5, where 1 is lowest risk)
    pub risk_profile: u8,
    
    /// Is this strategy active?
    pub is_active: bool,
    
    /// Creation timestamp
    pub created_at: i64,
    
    /// Last update timestamp
    pub updated_at: i64,
    
    /// Target assets for the strategy
    pub target_assets: Vec<Pubkey>,
    
    /// Allocation percentages (out of 10000 for precision)
    pub allocations: Vec<u16>,
    
    /// Any specific parameters for the strategy
    pub extra_params: Vec<u8>,
    
    /// Performance statistics specific to this strategy
    pub performance: StrategyPerformance,
    
    /// Number of times this strategy was executed
    pub execution_count: u32,
    
    /// PDA bump
    pub bump: u8,
    
    /// Reserved space for future upgrades
    pub reserved: [u8; 32],
}

impl Strategy {
    /// Calculate the space required for a strategy
    pub fn space() -> usize {
        8 +  // discriminator
        32 + // vault
        8 +  // strategy_id
        4 + MAX_STRATEGY_DESCRIPTION_LENGTH + // description (string with len prefix)
        1 +  // strategy_type
        1 +  // risk_profile
        1 +  // is_active
        8 +  // created_at
        8 +  // updated_at
        4 + (32 * MAX_ASSETS_PER_STRATEGY) + // target_assets (vec with len prefix)
        4 + (2 * MAX_ASSETS_PER_STRATEGY) +  // allocations (vec with len prefix)
        4 + 128 + // extra_params (variable length with max 128 bytes)
        StrategyPerformance::space() + // performance
        4 +  // execution_count
        1 +  // bump
        32   // reserved
    }
}

/// Strategy performance metrics
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Default, PartialEq)]
pub struct StrategyPerformance {
    /// Return in basis points
    pub return_bps: i16,
    
    /// Volatility in basis points
    pub volatility_bps: u16,
    
    /// Sharpe ratio (multiplied by X100 for precision)
    pub sharpe_ratio_x100: i32,
    
    /// Maximum drawdown in basis points
    pub max_drawdown_bps: u16,
    
    /// Correlation with market (multiplied by X100 for precision)
    pub market_correlation_x100: i16,
    
    /// Best performing asset index
    pub best_asset_index: u8,
    
    /// Worst performing asset index
    pub worst_asset_index: u8,
}

impl StrategyPerformance {
    /// Calculate the space required for strategy performance
    pub fn space() -> usize {
        2 + // return_bps
        2 + // volatility_bps
        4 + // sharpe_ratio_x100
        2 + // max_drawdown_bps
        2 + // market_correlation_x100
        1 + // best_asset_index
        1   // worst_asset_index
    }
}

/// User deposit information
#[account]
#[derive(Debug)]
pub struct UserDeposit {
    /// The owner of this deposit
    pub owner: Pubkey,
    
    /// The vault this deposit belongs to
    pub vault: Pubkey,
    
    /// Total amount deposited by the user
    pub total_deposited: u64,
    
    /// Current share balance
    pub share_balance: u64,
    
    /// Timestamp of first deposit
    pub first_deposit_time: i64,
    
    /// Timestamp of last deposit
    pub last_deposit_time: i64,
    
    /// Timestamp of last withdrawal
    pub last_withdrawal_time: i64,
    
    /// Cumulative realized profit
    pub realized_profit: i64,
    
    /// Cumulative fees paid
    pub fees_paid: u64,
    
    /// PDA bump
    pub bump: u8,
    
    /// Reserved space for future upgrades
    pub reserved: [u8; 32],
}

impl UserDeposit {
    /// Calculate the space required for user deposit
    pub fn space() -> usize {
        8 +  // discriminator
        32 + // owner
        32 + // vault
        8 +  // total_deposited
        8 +  // share_balance
        8 +  // first_deposit_time
        8 +  // last_deposit_time
        8 +  // last_withdrawal_time
        8 +  // realized_profit
        8 +  // fees_paid
        1 +  // bump
        32   // reserved
    }
}

/// User profile for AI strategy customization
#[account]
#[derive(Debug)]
pub struct UserProfile {
    /// The owner of this profile
    pub owner: Pubkey,
    
    /// User's risk tolerance (1-10)
    pub risk_tolerance: u8,
    
    /// User's investment time horizon in days
    pub time_horizon: u32,
    
    /// User's preferred investment categories (bitmask)
    pub preferences: u32,
    
    /// User-provided tags
    pub tags: Vec<String>,
    
    /// AI-generated profile hash (can be used for clustering)
    pub profile_hash: [u8; 32],
    
    /// Timestamp of creation
    pub created_at: i64,
    
    /// Timestamp of last update
    pub updated_at: i64,
    
    /// PDA bump
    pub bump: u8,
    
    /// Reserved space for future upgrades
    pub reserved: [u8; 32],
}

impl UserProfile {
    /// Calculate the space required for user profile
    pub fn space() -> usize {
        8 +  // discriminator
        32 + // owner
        1 +  // risk_tolerance
        4 +  // time_horizon
        4 +  // preferences
        4 + (4 + MAX_USER_TAG_LENGTH) * MAX_USER_TAGS + // tags (vec with strings with len prefixes)
        32 + // profile_hash
        8 +  // created_at
        8 +  // updated_at
        1 +  // bump
        32   // reserved
    }
}

/// Rewards distribution configuration
#[account]
#[derive(Debug)]
pub struct RewardsConfig {
    /// The vault this rewards config belongs to
    pub vault: Pubkey,
    
    /// The token mint for rewards
    pub rewards_mint: Pubkey,
    
    /// Rewards emission rate per second
    pub emission_rate: u64,
    
    /// Start timestamp of rewards
    pub start_time: i64,
    
    /// End timestamp of rewards
    pub end_time: i64,
    
    /// Last update timestamp
    pub last_update_time: i64,
    
    /// Cumulative reward per share (scaled for precision)
    pub reward_per_share_x64: u128,
    
    /// Total rewards distributed
    pub total_rewards_distributed: u64,
    
    /// Total rewards claimed
    pub total_rewards_claimed: u64,
    
    /// PDA bump
    pub bump: u8,
    
    /// Reserved space for future upgrades
    pub reserved: [u8; 32],
}

impl RewardsConfig {
    /// Calculate the space required for rewards config
    pub fn space() -> usize {
        8 +   // discriminator
        32 +  // vault
        32 +  // rewards_mint
        8 +   // emission_rate
        8 +   // start_time
        8 +   // end_time
        8 +   // last_update_time
        16 +  // reward_per_share_x64
        8 +   // total_rewards_distributed
        8 +   // total_rewards_claimed
        1 +   // bump
        32    // reserved
    }
}

/// User rewards information
#[account]
#[derive(Debug)]
pub struct UserRewards {
    /// The owner of these rewards
    pub owner: Pubkey,
    
    /// The vault these rewards belong to
    pub vault: Pubkey,
    
    /// The rewards configuration these rewards are from
    pub rewards_config: Pubkey,
    
    /// The reward per share value when the user last claimed
    pub reward_debt_per_share_x64: u128,
    
    /// Pending rewards ready to claim
    pub pending_rewards: u64,
    
    /// Timestamp of last reward claim
    pub last_claim_time: i64,
    
    /// Total rewards claimed
    pub total_claimed: u64,
    
    /// PDA bump
    pub bump: u8,
    
    /// Reserved space for future upgrades
    pub reserved: [u8; 16],
}

impl UserRewards {
    /// Calculate the space required for user rewards
    pub fn space() -> usize {
        8 +   // discriminator
        32 +  // owner
        32 +  // vault
        32 +  // rewards_config
        16 +  // reward_debt_per_share_x64
        8 +   // pending_rewards
        8 +   // last_claim_time
        8 +   // total_claimed
        1 +   // bump
        16    // reserved
    }
}