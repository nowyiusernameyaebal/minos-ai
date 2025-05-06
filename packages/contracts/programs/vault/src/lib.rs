//! Minos-AI Vault Program
//!
//! This program implements an AI-powered DeFi strategy vault on Solana
//! that uses machine learning algorithms to optimize yield and risk management.
//!
//! The vault handles asset deposits, withdrawals, strategy execution,
//! fee collection, and performance tracking.

use anchor_lang::prelude::*;
use anchor_spl::{
    associated_token::AssociatedToken,
    token::{Mint, Token, TokenAccount, Transfer},
};
use solana_program::{
    program_error::ProgramError,
    pubkey::Pubkey,
    sysvar::{clock::Clock, rent::Rent},
};

// Import modules
pub mod errors;
pub mod events;
pub mod instructions;
pub mod state;
pub mod utils;

// Re-export key components for convenience
pub use errors::VaultError;
pub use events::*;
pub use instructions::*;
pub use state::*;

// Security measure
solana_security_txt::security_txt! {
    name: "Minos Vault Program",
    project_url: "https://minos.ai",
    contacts: "email:security@minos.ai,discord:https://discord.gg/minos,twitter:@MinosAI",
    policy: "https://github.com/minos-ai/minos-platform/blob/main/SECURITY.md",
    preferred_languages: "en",
    source_code: "https://github.com/minos-ai/minos-platform",
    auditors: ""
}

// Program ID
declare_id!("Vauu5LGc3GUVxcMgp12wonnV4HaJXWgy5vwAHZmJftx");

/// Main program entry point
#[program]
pub mod minos_vault {
    use super::*;

    /// Initialize a new vault with the specified parameters
    pub fn initialize_vault(
        ctx: Context<InitializeVault>,
        params: VaultInitParams,
    ) -> Result<()> {
        instructions::initialize_vault::handler(ctx, params)
    }

    /// Deposit assets into the vault
    pub fn deposit(ctx: Context<Deposit>, amount: u64) -> Result<()> {
        instructions::deposit::handler(ctx, amount)
    }

    /// Withdraw assets from the vault
    pub fn withdraw(ctx: Context<Withdraw>, amount: u64) -> Result<()> {
        instructions::withdraw::handler(ctx, amount)
    }

    /// Execute a strategy on behalf of the vault
    pub fn execute_strategy(ctx: Context<ExecuteStrategy>, params: StrategyParams) -> Result<()> {
        instructions::execute_strategy::handler(ctx, params)
    }

    /// Update vault parameters and configuration
    pub fn update_vault_config(
        ctx: Context<UpdateVaultConfig>,
        params: VaultConfigParams,
    ) -> Result<()> {
        instructions::update_vault_config::handler(ctx, params)
    }

    /// Collect performance fees for the vault
    pub fn collect_fees(ctx: Context<CollectFees>) -> Result<()> {
        instructions::collect_fees::handler(ctx)
    }

    /// Add a new strategy to the vault
    pub fn add_strategy(ctx: Context<AddStrategy>, params: AddStrategyParams) -> Result<()> {
        instructions::add_strategy::handler(ctx, params)
    }

    /// Update an existing strategy's parameters
    pub fn update_strategy(
        ctx: Context<UpdateStrategy>,
        strategy_id: u64,
        params: UpdateStrategyParams,
    ) -> Result<()> {
        instructions::update_strategy::handler(ctx, strategy_id, params)
    }

    /// Pause a vault's operations
    pub fn pause_vault(ctx: Context<PauseVault>) -> Result<()> {
        instructions::pause_vault::handler(ctx)
    }

    /// Resume a vault's operations
    pub fn resume_vault(ctx: Context<ResumeVault>) -> Result<()> {
        instructions::resume_vault::handler(ctx)
    }

    /// Emergency withdraw all funds in case of critical issues
    pub fn emergency_withdraw(ctx: Context<EmergencyWithdraw>) -> Result<()> {
        instructions::emergency_withdraw::handler(ctx)
    }

    /// Rebalance the vault's holdings according to the current strategy
    pub fn rebalance(ctx: Context<Rebalance>) -> Result<()> {
        instructions::rebalance::handler(ctx)
    }
    
    /// Update the oracle data for vault pricing
    pub fn update_oracle(ctx: Context<UpdateOracle>) -> Result<()> {
        instructions::update_oracle::handler(ctx)
    }
    
    /// Initialize the reward distribution system for vault stakers
    pub fn initialize_rewards(
        ctx: Context<InitializeRewards>,
        params: RewardsParams,
    ) -> Result<()> {
        instructions::initialize_rewards::handler(ctx, params)
    }
    
    /// Claim accrued rewards for a vault participant
    pub fn claim_rewards(ctx: Context<ClaimRewards>) -> Result<()> {
        instructions::claim_rewards::handler(ctx)
    }
    
    /// Register a user's analytics profile for AI strategy customization
    pub fn register_user_profile(
        ctx: Context<RegisterUserProfile>,
        params: UserProfileParams,
    ) -> Result<()> {
        instructions::register_user_profile::handler(ctx, params)
    }
}

// Include instruction contexts
pub mod contexts {
    use super::*;

    /// Context for initializing a new vault
    #[derive(Accounts)]
    #[instruction(params: VaultInitParams)]
    pub struct InitializeVault<'info> {
        #[account(mut)]
        pub authority: Signer<'info>,
        
        #[account(
            init,
            payer = authority,
            space = Vault::space(),
            seeds = [b"vault", params.vault_name.as_bytes(), authority.key().as_ref()],
            bump
        )]
        pub vault: Account<'info, Vault>,
        
        #[account(
            init,
            payer = authority,
            seeds = [b"vault_treasury", vault.key().as_ref()],
            bump,
            token::mint = asset_mint,
            token::authority = vault,
        )]
        pub vault_treasury: Account<'info, TokenAccount>,
        
        pub asset_mint: Account<'info, Mint>,
        
        #[account(
            init,
            payer = authority,
            seeds = [b"vault_share_mint", vault.key().as_ref()],
            bump,
        )]
        pub vault_share_mint: Account<'info, Mint>,
        
        pub fee_recipient: SystemAccount<'info>,
        
        pub system_program: Program<'info, System>,
        pub token_program: Program<'info, Token>,
        pub associated_token_program: Program<'info, AssociatedToken>,
        pub rent: Sysvar<'info, Rent>,
    }

    /// Context for depositing into a vault
    #[derive(Accounts)]
    pub struct Deposit<'info> {
        #[account(mut)]
        pub user: Signer<'info>,
        
        #[account(
            mut,
            seeds = [b"vault", vault.vault_name.as_bytes(), vault.authority.as_ref()],
            bump = vault.bump,
        )]
        pub vault: Account<'info, Vault>,
        
        #[account(
            mut,
            seeds = [b"vault_treasury", vault.key().as_ref()],
            bump = vault.treasury_bump,
        )]
        pub vault_treasury: Account<'info, TokenAccount>,
        
        #[account(
            mut,
            constraint = user_token_account.mint == vault.asset_mint,
            constraint = user_token_account.owner == user.key(),
        )]
        pub user_token_account: Account<'info, TokenAccount>,
        
        #[account(
            mut,
            seeds = [b"vault_share_mint", vault.key().as_ref()],
            bump = vault.share_mint_bump,
        )]
        pub vault_share_mint: Account<'info, Mint>,
        
        #[account(
            init_if_needed,
            payer = user,
            associated_token::mint = vault_share_mint,
            associated_token::authority = user,
        )]
        pub user_share_token_account: Account<'info, TokenAccount>,
        
        pub token_program: Program<'info, Token>,
        pub associated_token_program: Program<'info, AssociatedToken>,
        pub system_program: Program<'info, System>,
        pub rent: Sysvar<'info, Rent>,
    }

    /// Context for withdrawing from a vault
    #[derive(Accounts)]
    pub struct Withdraw<'info> {
        #[account(mut)]
        pub user: Signer<'info>,
        
        #[account(
            mut,
            seeds = [b"vault", vault.vault_name.as_bytes(), vault.authority.as_ref()],
            bump = vault.bump,
        )]
        pub vault: Account<'info, Vault>,
        
        #[account(
            mut,
            seeds = [b"vault_treasury", vault.key().as_ref()],
            bump = vault.treasury_bump,
        )]
        pub vault_treasury: Account<'info, TokenAccount>,
        
        #[account(
            mut,
            constraint = user_token_account.mint == vault.asset_mint,
            constraint = user_token_account.owner == user.key(),
        )]
        pub user_token_account: Account<'info, TokenAccount>,
        
        #[account(
            mut,
            seeds = [b"vault_share_mint", vault.key().as_ref()],
            bump = vault.share_mint_bump,
        )]
        pub vault_share_mint: Account<'info, Mint>,
        
        #[account(
            mut,
            associated_token::mint = vault_share_mint,
            associated_token::authority = user,
        )]
        pub user_share_token_account: Account<'info, TokenAccount>,
        
        pub token_program: Program<'info, Token>,
        pub system_program: Program<'info, System>,
    }

    /// Context for executing strategies
    #[derive(Accounts)]
    pub struct ExecuteStrategy<'info> {
        #[account(
            constraint = strategy_manager.key() == vault.strategy_manager || vault.authority == strategy_manager.key(),
        )]
        pub strategy_manager: Signer<'info>,
        
        #[account(
            mut,
            seeds = [b"vault", vault.vault_name.as_bytes(), vault.authority.as_ref()],
            bump = vault.bump,
            constraint = !vault.is_paused @ VaultError::VaultIsPaused,
        )]
        pub vault: Account<'info, Vault>,
        
        #[account(
            mut,
            seeds = [b"vault_treasury", vault.key().as_ref()],
            bump = vault.treasury_bump,
        )]
        pub vault_treasury: Account<'info, TokenAccount>,
        
        // Additional accounts will be passed in dynamically based on the strategy
        // This is the base context only
        
        pub token_program: Program<'info, Token>,
        pub system_program: Program<'info, System>,
        pub clock: Sysvar<'info, Clock>,
    }

    // Additional contexts would be defined here for the remaining instructions
    // This is a simplified version for brevity
}

// Include helper types
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq)]
pub struct VaultInitParams {
    /// Name of the vault
    pub vault_name: String,
    /// Description of vault purpose and strategy
    pub description: String,
    /// Fee configuration for the vault
    pub fee_config: FeeConfig,
    /// Risk profile (1-5, where 1 is lowest risk)
    pub risk_profile: u8,
    /// Maximum capacity of the vault
    pub capacity: Option<u64>,
    /// Strategy execution frequency in seconds
    pub strategy_execution_frequency: u64,
    /// Minimum deposit amount
    pub min_deposit: u64,
    /// Maximum deposit amount per user
    pub max_deposit_per_user: Option<u64>,
    /// Optional lockup period in seconds
    pub lockup_period: Option<u64>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq)]
pub struct StrategyParams {
    /// Unique identifier for the strategy type
    pub strategy_type: u8,
    /// Target assets for the strategy
    pub target_assets: Vec<Pubkey>,
    /// Allocation percentages (out of 10000 for precision)
    pub allocations: Vec<u16>,
    /// Any specific parameters for the strategy
    pub extra_params: Vec<u8>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq)]
pub struct VaultConfigParams {
    /// Optional new description
    pub description: Option<String>,
    /// Optional new fee configuration
    pub fee_config: Option<FeeConfig>,
    /// Optional new risk profile
    pub risk_profile: Option<u8>,
    /// Optional new capacity
    pub capacity: Option<u64>,
    /// Optional new strategy execution frequency
    pub strategy_execution_frequency: Option<u64>,
    /// Optional new strategy manager
    pub strategy_manager: Option<Pubkey>,
    /// Optional new minimum deposit amount
    pub min_deposit: Option<u64>,
    /// Optional new maximum deposit per user
    pub max_deposit_per_user: Option<u64>,
    /// Optional new lockup period
    pub lockup_period: Option<u64>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq)]
pub struct AddStrategyParams {
    /// Unique identifier for this strategy
    pub strategy_id: u64,
    /// Description of the strategy
    pub description: String,
    /// Strategy type identifier
    pub strategy_type: u8,
    /// Risk profile (1-5, where 1 is lowest risk)
    pub risk_profile: u8,
    /// Target assets for the strategy
    pub target_assets: Vec<Pubkey>,
    /// Allocation percentages (out of 10000 for precision)
    pub allocations: Vec<u16>,
    /// Any specific parameters for the strategy
    pub extra_params: Vec<u8>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq)]
pub struct UpdateStrategyParams {
    /// Optional new description
    pub description: Option<String>,
    /// Optional new risk profile
    pub risk_profile: Option<u8>,
    /// Optional new target assets
    pub target_assets: Option<Vec<Pubkey>>,
    /// Optional new allocations
    pub allocations: Option<Vec<u16>>,
    /// Optional new extra parameters
    pub extra_params: Option<Vec<u8>>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq)]
pub struct RewardsParams {
    /// The token mint for rewards
    pub rewards_mint: Pubkey,
    /// Rewards emission rate per second
    pub emission_rate: u64,
    /// Duration of rewards program in seconds
    pub duration: u64,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq)]
pub struct UserProfileParams {
    /// User's risk tolerance (1-10)
    pub risk_tolerance: u8,
    /// User's investment time horizon in days
    pub time_horizon: u32,
    /// User's preferred investment categories (bitmask)
    pub preferences: u32,
    /// Optional user-provided tags
    pub tags: Vec<String>,
}