// Minos-AI Agent - Solana Program for AI-powered trading and investment strategies
// 
// This program implements an on-chain AI agent system that can:
// - Store and manage AI model metadata and verification
// - Execute trading strategies based on AI signals
// - Authorize and verify AI-generated signals with cryptographic proofs
// - Manage strategy vaults and handle asset allocation
// - Integrate with other Minos-AI components via cross-program invocation

use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};
use solana_program::{program::invoke_signed, pubkey::Pubkey, system_instruction};
use std::convert::TryFrom;

// Security imports
use solana_security_txt::security_txt;

// Import modules
pub mod errors;
pub mod events;
pub mod instructions;
pub mod models;
pub mod state;
pub mod utils;
pub mod constants;
pub mod validation;

// Re-export key components
pub use errors::AgentError;
pub use state::{AgentAccount, ModelAccount, SignalAccount, Strategy, StrategyParams};
pub use models::{ModelMetadata, Signal, SignalType, SignalVerification};
pub use constants::*;

// Security.txt metadata for vulnerability reporting
security_txt! {
    name: "Minos-AI Agent",
    project_url: "https://minos-ai.com",
    contacts: "email:security@minos-ai.com,discord:minos-security,github:https://github.com/minos-ai/minos-ai/security",
    policy: "https://minos-ai.com/security-policy",
    preferred_languages: "en",
    source_code: "https://github.com/minos-ai/minos-ai",
    encryption: "https://minos-ai.com/pgp-key.txt",
    acknowledgements: "https://minos-ai.com/hall-of-fame",
    expiry: "2025-12-31T00:00:00Z"
}

// Program ID
declare_id!("Agen5PFLmvq6M7K7x9qAKVoVj9YZZCNz1VUKNdKBuUM");

#[program]
pub mod minos_ai_agent {
    use super::*;

    /// Initialize a new AI agent with admin authority and configuration
    pub fn initialize_agent(
        ctx: Context<InitializeAgent>,
        params: state::AgentParams,
    ) -> Result<()> {
        instructions::agent::initialize_agent(ctx, params)
    }

    /// Register a new AI model for signals generation
    pub fn register_model(
        ctx: Context<RegisterModel>,
        metadata: models::ModelMetadata,
    ) -> Result<()> {
        instructions::model::register_model(ctx, metadata)
    }

    /// Update existing AI model metadata
    pub fn update_model(
        ctx: Context<UpdateModel>,
        metadata: models::ModelMetadata,
    ) -> Result<()> {
        instructions::model::update_model(ctx, metadata)
    }

    /// Deactivate an AI model
    pub fn deactivate_model(ctx: Context<DeactivateModel>) -> Result<()> {
        instructions::model::deactivate_model(ctx)
    }

    /// Create a new trading strategy with parameters and initial allocation
    pub fn create_strategy(
        ctx: Context<CreateStrategy>,
        params: state::StrategyParams,
    ) -> Result<()> {
        instructions::strategy::create_strategy(ctx, params)
    }

    /// Update an existing strategy parameters
    pub fn update_strategy(
        ctx: Context<UpdateStrategy>,
        params: state::StrategyParams,
    ) -> Result<()> {
        instructions::strategy::update_strategy(ctx, params)
    }

    /// Submit a new trading signal from an AI model
    pub fn submit_signal(
        ctx: Context<SubmitSignal>,
        signal: models::Signal,
        verification: models::SignalVerification,
    ) -> Result<()> {
        instructions::signal::submit_signal(ctx, signal, verification)
    }

    /// Execute a validated trading signal
    pub fn execute_signal(
        ctx: Context<ExecuteSignal>,
        signal_id: u64,
        execution_params: models::ExecutionParams,
    ) -> Result<()> {
        instructions::signal::execute_signal(ctx, signal_id, execution_params)
    }

    /// Deposit funds into a strategy vault
    pub fn deposit_funds(
        ctx: Context<DepositFunds>,
        amount: u64,
    ) -> Result<()> {
        instructions::vault::deposit_funds(ctx, amount)
    }

    /// Withdraw funds from a strategy vault
    pub fn withdraw_funds(
        ctx: Context<WithdrawFunds>,
        amount: u64,
    ) -> Result<()> {
        instructions::vault::withdraw_funds(ctx, amount)
    }

    /// Update agent configuration
    pub fn update_agent_config(
        ctx: Context<UpdateAgentConfig>,
        params: state::AgentParams,
    ) -> Result<()> {
        instructions::agent::update_agent_config(ctx, params)
    }

    /// Pause all operations of the agent (emergency function)
    pub fn pause_agent(ctx: Context<PauseAgent>) -> Result<()> {
        instructions::agent::pause_agent(ctx)
    }

    /// Resume operations after pause
    pub fn resume_agent(ctx: Context<ResumeAgent>) -> Result<()> {
        instructions::agent::resume_agent(ctx)
    }

    /// Record model performance metrics
    pub fn record_performance(
        ctx: Context<RecordPerformance>,
        metrics: models::PerformanceMetrics,
    ) -> Result<()> {
        instructions::model::record_performance(ctx, metrics)
    }

    /// Collect fees from strategy profits
    pub fn collect_fees(
        ctx: Context<CollectFees>,
        strategy_id: u64,
    ) -> Result<()> {
        instructions::vault::collect_fees(ctx, strategy_id)
    }

    /// Update agent authority (ownership transfer)
    pub fn update_authority(
        ctx: Context<UpdateAuthority>,
        new_authority: Pubkey,
    ) -> Result<()> {
        instructions::agent::update_authority(ctx, new_authority)
    }
}

#[derive(Accounts)]
pub struct InitializeAgent<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        init,
        payer = authority,
        space = AgentAccount::LEN,
        seeds = [AGENT_SEED.as_bytes()],
        bump
    )]
    pub agent_account: Account<'info, AgentAccount>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct RegisterModel<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        seeds = [AGENT_SEED.as_bytes()],
        bump = agent_account.bump,
        constraint = agent_account.authority == authority.key() @ AgentError::InvalidAuthority,
        constraint = !agent_account.paused @ AgentError::AgentPaused
    )]
    pub agent_account: Account<'info, AgentAccount>,

    #[account(
        init,
        payer = authority,
        space = ModelAccount::LEN,
        seeds = [MODEL_SEED.as_bytes(), agent_account.next_model_id.to_le_bytes().as_ref()],
        bump
    )]
    pub model_account: Account<'info, ModelAccount>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct UpdateModel<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        seeds = [AGENT_SEED.as_bytes()],
        bump = agent_account.bump,
        constraint = agent_account.authority == authority.key() @ AgentError::InvalidAuthority,
        constraint = !agent_account.paused @ AgentError::AgentPaused
    )]
    pub agent_account: Account<'info, AgentAccount>,

    #[account(
        mut,
        seeds = [MODEL_SEED.as_bytes(), model_account.model_id.to_le_bytes().as_ref()],
        bump = model_account.bump,
        constraint = model_account.active @ AgentError::ModelNotActive
    )]
    pub model_account: Account<'info, ModelAccount>,
}

#[derive(Accounts)]
pub struct DeactivateModel<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        seeds = [AGENT_SEED.as_bytes()],
        bump = agent_account.bump,
        constraint = agent_account.authority == authority.key() @ AgentError::InvalidAuthority
    )]
    pub agent_account: Account<'info, AgentAccount>,

    #[account(
        mut,
        seeds = [MODEL_SEED.as_bytes(), model_account.model_id.to_le_bytes().as_ref()],
        bump = model_account.bump,
        constraint = model_account.active @ AgentError::ModelNotActive
    )]
    pub model_account: Account<'info, ModelAccount>,
}

#[derive(Accounts)]
pub struct CreateStrategy<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        seeds = [AGENT_SEED.as_bytes()],
        bump = agent_account.bump,
        constraint = agent_account.authority == authority.key() @ AgentError::InvalidAuthority,
        constraint = !agent_account.paused @ AgentError::AgentPaused,
        constraint = agent_account.strategies_count < MAX_STRATEGIES @ AgentError::MaxStrategiesReached
    )]
    pub agent_account: Account<'info, AgentAccount>,

    #[account(
        init,
        payer = authority,
        space = Strategy::LEN,
        seeds = [STRATEGY_SEED.as_bytes(), agent_account.next_strategy_id.to_le_bytes().as_ref()],
        bump
    )]
    pub strategy_account: Account<'info, Strategy>,

    #[account(
        seeds = [MODEL_SEED.as_bytes(), strategy_params.model_id.to_le_bytes().as_ref()],
        bump,
        constraint = model_account.active @ AgentError::ModelNotActive
    )]
    pub model_account: Account<'info, ModelAccount>,

    #[account(
        init,
        payer = authority,
        seeds = [VAULT_SEED.as_bytes(), agent_account.next_strategy_id.to_le_bytes().as_ref()],
        bump,
        token::mint = token_mint,
        token::authority = strategy_account
    )]
    pub strategy_vault: Account<'info, TokenAccount>,

    pub token_mint: Account<'info, token::Mint>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct UpdateStrategy<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        seeds = [AGENT_SEED.as_bytes()],
        bump = agent_account.bump,
        constraint = agent_account.authority == authority.key() @ AgentError::InvalidAuthority,
        constraint = !agent_account.paused @ AgentError::AgentPaused
    )]
    pub agent_account: Account<'info, AgentAccount>,

    #[account(
        mut,
        seeds = [STRATEGY_SEED.as_bytes(), strategy_account.strategy_id.to_le_bytes().as_ref()],
        bump = strategy_account.bump,
        constraint = strategy_account.active @ AgentError::StrategyNotActive
    )]
    pub strategy_account: Account<'info, Strategy>,

    #[account(
        seeds = [MODEL_SEED.as_bytes(), strategy_account.model_id.to_le_bytes().as_ref()],
        bump,
        constraint = model_account.active @ AgentError::ModelNotActive
    )]
    pub model_account: Account<'info, ModelAccount>,
}

#[derive(Accounts)]
pub struct SubmitSignal<'info> {
    #[account(mut)]
    pub submitter: Signer<'info>,

    #[account(
        seeds = [AGENT_SEED.as_bytes()],
        bump = agent_account.bump,
        constraint = !agent_account.paused @ AgentError::AgentPaused
    )]
    pub agent_account: Account<'info, AgentAccount>,

    #[account(
        seeds = [MODEL_SEED.as_bytes(), model_account.model_id.to_le_bytes().as_ref()],
        bump = model_account.bump,
        constraint = model_account.active @ AgentError::ModelNotActive,
        constraint = model_account.authorized_submitter == submitter.key() @ AgentError::UnauthorizedSubmitter
    )]
    pub model_account: Account<'info, ModelAccount>,

    #[account(
        seeds = [STRATEGY_SEED.as_bytes(), strategy_account.strategy_id.to_le_bytes().as_ref()],
        bump = strategy_account.bump,
        constraint = strategy_account.active @ AgentError::StrategyNotActive,
        constraint = strategy_account.model_id == model_account.model_id @ AgentError::InvalidModelForStrategy
    )]
    pub strategy_account: Account<'info, Strategy>,

    #[account(
        init,
        payer = submitter,
        space = SignalAccount::LEN,
        seeds = [SIGNAL_SEED.as_bytes(), model_account.model_id.to_le_bytes().as_ref(), model_account.next_signal_id.to_le_bytes().as_ref()],
        bump
    )]
    pub signal_account: Account<'info, SignalAccount>,

    pub system_program: Program<'info, System>,
    pub clock: Sysvar<'info, Clock>,
}

#[derive(Accounts)]
pub struct ExecuteSignal<'info> {
    #[account(mut)]
    pub executor: Signer<'info>,

    #[account(
        seeds = [AGENT_SEED.as_bytes()],
        bump = agent_account.bump,
        constraint = !agent_account.paused @ AgentError::AgentPaused
    )]
    pub agent_account: Account<'info, AgentAccount>,

    #[account(
        mut,
        seeds = [SIGNAL_SEED.as_bytes(), signal_account.model_id.to_le_bytes().as_ref(), signal_account.signal_id.to_le_bytes().as_ref()],
        bump = signal_account.bump,
        constraint = signal_account.verified @ AgentError::SignalNotVerified,
        constraint = !signal_account.executed @ AgentError::SignalAlreadyExecuted,
        constraint = signal_account.expiry_time > clock.unix_timestamp @ AgentError::SignalExpired
    )]
    pub signal_account: Account<'info, SignalAccount>,

    #[account(
        mut,
        seeds = [STRATEGY_SEED.as_bytes(), strategy_account.strategy_id.to_le_bytes().as_ref()],
        bump = strategy_account.bump,
        constraint = strategy_account.active @ AgentError::StrategyNotActive,
        constraint = strategy_account.model_id == signal_account.model_id @ AgentError::InvalidModelForStrategy
    )]
    pub strategy_account: Account<'info, Strategy>,

    #[account(
        mut,
        seeds = [VAULT_SEED.as_bytes(), strategy_account.strategy_id.to_le_bytes().as_ref()],
        bump = strategy_account.vault_bump
    )]
    pub strategy_vault: Account<'info, TokenAccount>,

    // Additional accounts for trade execution will be provided through remaining_accounts
    pub token_program: Program<'info, Token>,
    pub clock: Sysvar<'info, Clock>,
}

#[derive(Accounts)]
pub struct DepositFunds<'info> {
    #[account(mut)]
    pub depositor: Signer<'info>,

    #[account(
        seeds = [AGENT_SEED.as_bytes()],
        bump = agent_account.bump,
        constraint = !agent_account.paused @ AgentError::AgentPaused
    )]
    pub agent_account: Account<'info, AgentAccount>,

    #[account(
        seeds = [STRATEGY_SEED.as_bytes(), strategy_account.strategy_id.to_le_bytes().as_ref()],
        bump = strategy_account.bump,
        constraint = strategy_account.active @ AgentError::StrategyNotActive
    )]
    pub strategy_account: Account<'info, Strategy>,

    #[account(
        mut,
        seeds = [VAULT_SEED.as_bytes(), strategy_account.strategy_id.to_le_bytes().as_ref()],
        bump = strategy_account.vault_bump
    )]
    pub strategy_vault: Account<'info, TokenAccount>,

    #[account(
        mut,
        constraint = depositor_token_account.mint == strategy_vault.mint,
        constraint = depositor_token_account.owner == depositor.key()
    )]
    pub depositor_token_account: Account<'info, TokenAccount>,

    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct WithdrawFunds<'info> {
    #[account(mut)]
    pub owner: Signer<'info>,

    #[account(
        seeds = [AGENT_SEED.as_bytes()],
        bump = agent_account.bump,
        constraint = !agent_account.paused @ AgentError::AgentPaused
    )]
    pub agent_account: Account<'info, AgentAccount>,

    #[account(
        mut,
        seeds = [STRATEGY_SEED.as_bytes(), strategy_account.strategy_id.to_le_bytes().as_ref()],
        bump = strategy_account.bump,
        constraint = strategy_account.active @ AgentError::StrategyNotActive,
        constraint = strategy_account.owner == owner.key() @ AgentError::NotStrategyOwner
    )]
    pub strategy_account: Account<'info, Strategy>,

    #[account(
        mut,
        seeds = [VAULT_SEED.as_bytes(), strategy_account.strategy_id.to_le_bytes().as_ref()],
        bump = strategy_account.vault_bump
    )]
    pub strategy_vault: Account<'info, TokenAccount>,

    #[account(
        mut,
        constraint = recipient_token_account.mint == strategy_vault.mint,
        constraint = recipient_token_account.owner == owner.key()
    )]
    pub recipient_token_account: Account<'info, TokenAccount>,

    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct UpdateAgentConfig<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        mut,
        seeds = [AGENT_SEED.as_bytes()],
        bump = agent_account.bump,
        constraint = agent_account.authority == authority.key() @ AgentError::InvalidAuthority
    )]
    pub agent_account: Account<'info, AgentAccount>,
}

#[derive(Accounts)]
pub struct PauseAgent<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        mut,
        seeds = [AGENT_SEED.as_bytes()],
        bump = agent_account.bump,
        constraint = agent_account.authority == authority.key() @ AgentError::InvalidAuthority,
        constraint = !agent_account.paused @ AgentError::AgentAlreadyPaused
    )]
    pub agent_account: Account<'info, AgentAccount>,
}

#[derive(Accounts)]
pub struct ResumeAgent<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        mut,
        seeds = [AGENT_SEED.as_bytes()],
        bump = agent_account.bump,
        constraint = agent_account.authority == authority.key() @ AgentError::InvalidAuthority,
        constraint = agent_account.paused @ AgentError::AgentNotPaused
    )]
    pub agent_account: Account<'info, AgentAccount>,
}

#[derive(Accounts)]
pub struct RecordPerformance<'info> {
    #[account(mut)]
    pub authorized_submitter: Signer<'info>,

    #[account(
        seeds = [AGENT_SEED.as_bytes()],
        bump = agent_account.bump,
        constraint = !agent_account.paused @ AgentError::AgentPaused
    )]
    pub agent_account: Account<'info, AgentAccount>,

    #[account(
        mut,
        seeds = [MODEL_SEED.as_bytes(), model_account.model_id.to_le_bytes().as_ref()],
        bump = model_account.bump,
        constraint = model_account.active @ AgentError::ModelNotActive,
        constraint = model_account.authorized_submitter == authorized_submitter.key() @ AgentError::UnauthorizedSubmitter
    )]
    pub model_account: Account<'info, ModelAccount>,

    pub clock: Sysvar<'info, Clock>,
}

#[derive(Accounts)]
pub struct CollectFees<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        seeds = [AGENT_SEED.as_bytes()],
        bump = agent_account.bump,
        constraint = agent_account.authority == authority.key() @ AgentError::InvalidAuthority,
        constraint = !agent_account.paused @ AgentError::AgentPaused
    )]
    pub agent_account: Account<'info, AgentAccount>,

    #[account(
        mut,
        seeds = [STRATEGY_SEED.as_bytes(), strategy_id.to_le_bytes().as_ref()],
        bump = strategy_account.bump,
        constraint = strategy_account.active @ AgentError::StrategyNotActive
    )]
    pub strategy_account: Account<'info, Strategy>,

    #[account(
        mut,
        seeds = [VAULT_SEED.as_bytes(), strategy_id.to_le_bytes().as_ref()],
        bump = strategy_account.vault_bump
    )]
    pub strategy_vault: Account<'info, TokenAccount>,

    #[account(
        mut,
        constraint = fee_recipient_account.mint == strategy_vault.mint
    )]
    pub fee_recipient_account: Account<'info, TokenAccount>,

    pub token_program: Program<'info, Token>,
    pub clock: Sysvar<'info, Clock>,
}

#[derive(Accounts)]
pub struct UpdateAuthority<'info> {
    #[account(mut)]
    pub current_authority: Signer<'info>,

    /// CHECK: We're just reading the public key
    pub new_authority: UncheckedAccount<'info>,

    #[account(
        mut,
        seeds = [AGENT_SEED.as_bytes()],
        bump = agent_account.bump,
        constraint = agent_account.authority == current_authority.key() @ AgentError::InvalidAuthority
    )]
    pub agent_account: Account<'info, AgentAccount>,
}