// Minos-AI Agent - Instructions Module
//
// This module provides implementations for all instructions defined in the AI agent program.
// It's organized into submodules for better code organization, with each submodule handling
// a specific area of functionality.

use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};
use solana_program::{program::invoke_signed, pubkey::Pubkey, system_instruction};

use crate::{
    errors::AgentError,
    events,
    models::{ModelMetadata, Signal, SignalType, SignalVerification, PerformanceMetrics, ExecutionParams},
    state::{AgentAccount, ModelAccount, SignalAccount, Strategy, StrategyParams, AgentParams},
    constants::*,
    utils::{verify_signal, get_current_price, calculate_fees},
    validation,
};

/// Agent management instructions
pub mod agent {
    use super::*;

    /// Initialize a new AI agent with admin authority and configuration
    pub fn initialize_agent(
        ctx: Context<crate::InitializeAgent>,
        params: AgentParams,
    ) -> Result<()> {
        let agent_account = &mut ctx.accounts.agent_account;
        let authority = &ctx.accounts.authority;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Validate parameters
        if params.protocol_fee_bps > MAX_PROTOCOL_FEE_BPS {
            return Err(AgentError::ProtocolFeeTooHigh.into());
        }
        
        if params.min_confidence_threshold < MIN_CONFIDENCE_THRESHOLD || params.min_confidence_threshold > 100 {
            return Err(AgentError::InvalidConfidenceThreshold.into());
        }
        
        if params.max_signal_validity_period > MAX_SIGNAL_VALIDITY_PERIOD {
            return Err(AgentError::SignalValidityTooLong.into());
        }
        
        // Set initial agent state
        agent_account.authority = authority.key();
        agent_account.bump = *ctx.bumps.get("agent_account").unwrap();
        agent_account.paused = false;
        agent_account.protocol_fee_bps = params.protocol_fee_bps;
        agent_account.fee_recipient = params.fee_recipient;
        agent_account.min_confidence_threshold = params.min_confidence_threshold;
        agent_account.max_signal_validity_period = params.max_signal_validity_period;
        agent_account.next_model_id = 1;
        agent_account.next_strategy_id = 1;
        agent_account.strategies_count = 0;
        agent_account.models_count = 0;
        agent_account.last_fee_collection = current_time;
        agent_account.max_strategies_per_model = params.max_strategies_per_model;
        agent_account.min_strategy_deposit = params.min_strategy_deposit;
        agent_account.last_update_time = current_time;
        agent_account.version = CURRENT_VERSION;

        // Emit initialization event
        emit!(events::AgentInitialized {
            authority: authority.key(),
            timestamp: current_time,
            protocol_fee_bps: params.protocol_fee_bps,
            version: CURRENT_VERSION,
        });
        
        Ok(())
    }

    /// Update agent configuration
    pub fn update_agent_config(
        ctx: Context<crate::UpdateAgentConfig>,
        params: AgentParams,
    ) -> Result<()> {
        let agent_account = &mut ctx.accounts.agent_account;
        let authority = &ctx.accounts.authority;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Validate parameters
        if params.protocol_fee_bps > MAX_PROTOCOL_FEE_BPS {
            return Err(AgentError::ProtocolFeeTooHigh.into());
        }
        
        if params.min_confidence_threshold < MIN_CONFIDENCE_THRESHOLD || params.min_confidence_threshold > 100 {
            return Err(AgentError::InvalidConfidenceThreshold.into());
        }
        
        if params.max_signal_validity_period > MAX_SIGNAL_VALIDITY_PERIOD {
            return Err(AgentError::SignalValidityTooLong.into());
        }
        
        // Update agent configuration
        agent_account.protocol_fee_bps = params.protocol_fee_bps;
        agent_account.fee_recipient = params.fee_recipient;
        agent_account.min_confidence_threshold = params.min_confidence_threshold;
        agent_account.max_signal_validity_period = params.max_signal_validity_period;
        agent_account.max_strategies_per_model = params.max_strategies_per_model;
        agent_account.min_strategy_deposit = params.min_strategy_deposit;
        agent_account.last_update_time = current_time;
        
        // Emit update event
        emit!(events::AgentConfigUpdated {
            authority: authority.key(),
            timestamp: current_time,
            protocol_fee_bps: params.protocol_fee_bps,
        });
        
        Ok(())
    }

    /// Pause all operations of the agent (emergency function)
    pub fn pause_agent(ctx: Context<crate::PauseAgent>) -> Result<()> {
        let agent_account = &mut ctx.accounts.agent_account;
        let authority = &ctx.accounts.authority;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Set pause flag
        agent_account.paused = true;
        
        // Emit pause event
        emit!(events::AgentPaused {
            authority: authority.key(),
            timestamp: current_time,
        });
        
        Ok(())
    }

    /// Resume operations after pause
    pub fn resume_agent(ctx: Context<crate::ResumeAgent>) -> Result<()> {
        let agent_account = &mut ctx.accounts.agent_account;
        let authority = &ctx.accounts.authority;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Clear pause flag
        agent_account.paused = false;
        
        // Emit resume event
        emit!(events::AgentResumed {
            authority: authority.key(),
            timestamp: current_time,
        });
        
        Ok(())
    }

    /// Update agent authority (ownership transfer)
    pub fn update_authority(
        ctx: Context<crate::UpdateAuthority>,
        new_authority: Pubkey,
    ) -> Result<()> {
        let agent_account = &mut ctx.accounts.agent_account;
        let current_authority = &ctx.accounts.current_authority;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Update authority
        let old_authority = agent_account.authority;
        agent_account.authority = new_authority;
        agent_account.last_update_time = current_time;
        
        // Emit authority update event
        emit!(events::AuthorityUpdated {
            old_authority,
            new_authority,
            timestamp: current_time,
        });
        
        Ok(())
    }
}

/// AI model management instructions
pub mod model {
    use super::*;

    /// Register a new AI model for signals generation
    pub fn register_model(
        ctx: Context<crate::RegisterModel>,
        metadata: ModelMetadata,
    ) -> Result<()> {
        let agent_account = &mut ctx.accounts.agent_account;
        let model_account = &mut ctx.accounts.model_account;
        let authority = &ctx.accounts.authority;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Validate metadata
        validation::validate_model_metadata(&metadata)?;
        
        // Set model account state
        model_account.model_id = agent_account.next_model_id;
        model_account.bump = *ctx.bumps.get("model_account").unwrap();
        model_account.active = true;
        model_account.authority = authority.key();
        model_account.authorized_submitter = metadata.submitter;
        model_account.metadata = metadata;
        model_account.strategies_count = 0;
        model_account.next_signal_id = 1;
        model_account.total_signals = 0;
        model_account.successful_signals = 0;
        model_account.registration_time = current_time;
        model_account.last_update_time = current_time;
        model_account.verification_key = [0; 64]; // Should be updated with actual verification key
        
        // Initialize performance metrics to default values
        model_account.performance = PerformanceMetrics::default();
        
        // Update agent state
        agent_account.next_model_id += 1;
        agent_account.models_count += 1;
        
        // Emit model registration event
        emit!(events::ModelRegistered {
            model_id: model_account.model_id,
            authority: authority.key(),
            timestamp: current_time,
            name: model_account.metadata.name,
            version: model_account.metadata.version,
        });
        
        Ok(())
    }

    /// Update existing AI model metadata
    pub fn update_model(
        ctx: Context<crate::UpdateModel>,
        metadata: ModelMetadata,
    ) -> Result<()> {
        let model_account = &mut ctx.accounts.model_account;
        let authority = &ctx.accounts.authority;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Validate metadata
        validation::validate_model_metadata(&metadata)?;
        
        // Update model account state
        model_account.authorized_submitter = metadata.submitter;
        model_account.metadata = metadata;
        model_account.last_update_time = current_time;
        
        // Emit model update event
        emit!(events::ModelUpdated {
            model_id: model_account.model_id,
            authority: authority.key(),
            timestamp: current_time,
            name: model_account.metadata.name,
            version: model_account.metadata.version,
        });
        
        Ok(())
    }

    /// Deactivate an AI model
    pub fn deactivate_model(ctx: Context<crate::DeactivateModel>) -> Result<()> {
        let model_account = &mut ctx.accounts.model_account;
        let authority = &ctx.accounts.authority;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Deactivate model
        model_account.active = false;
        model_account.last_update_time = current_time;
        
        // Emit model deactivation event
        emit!(events::ModelDeactivated {
            model_id: model_account.model_id,
            authority: authority.key(),
            timestamp: current_time,
        });
        
        Ok(())
    }

    /// Record model performance metrics
    pub fn record_performance(
        ctx: Context<crate::RecordPerformance>,
        metrics: PerformanceMetrics,
    ) -> Result<()> {
        let model_account = &mut ctx.accounts.model_account;
        let authorized_submitter = &ctx.accounts.authorized_submitter;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Update model performance metrics
        model_account.performance = metrics;
        model_account.last_update_time = current_time;
        
        // Emit performance record event
        emit!(events::PerformanceRecorded {
            model_id: model_account.model_id,
            submitter: authorized_submitter.key(),
            timestamp: current_time,
            win_rate: metrics.win_rate,
            sharpe_ratio: metrics.sharpe_ratio,
        });
        
        Ok(())
    }
}

/// Strategy management instructions
pub mod strategy {
    use super::*;

    /// Create a new trading strategy with parameters and initial allocation
    pub fn create_strategy(
        ctx: Context<crate::CreateStrategy>,
        params: StrategyParams,
    ) -> Result<()> {
        let agent_account = &mut ctx.accounts.agent_account;
        let strategy_account = &mut ctx.accounts.strategy_account;
        let model_account = &ctx.accounts.model_account;
        let authority = &ctx.accounts.authority;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Validate parameters
        validation::validate_strategy_params(&params)?;
        
        // Check if model has reached maximum strategies
        if model_account.strategies_count >= agent_account.max_strategies_per_model {
            return Err(AgentError::MaxStrategiesPerModelReached.into());
        }
        
        // Set strategy account state
        strategy_account.strategy_id = agent_account.next_strategy_id;
        strategy_account.bump = *ctx.bumps.get("strategy_account").unwrap();
        strategy_account.vault_bump = *ctx.bumps.get("strategy_vault").unwrap();
        strategy_account.active = true;
        strategy_account.owner = authority.key();
        strategy_account.model_id = params.model_id;
        strategy_account.params = params;
        strategy_account.last_update_time = current_time;
        strategy_account.creation_time = current_time;
        strategy_account.token_mint = ctx.accounts.token_mint.key();
        strategy_account.total_fees_collected = 0;
        strategy_account.initial_deposit = 0; // Will be updated when funds are deposited
        strategy_account.total_profit = 0;
        strategy_account.performance_fee_bps = params.performance_fee_bps;
        strategy_account.last_profit_calculation = current_time;
        strategy_account.high_water_mark = 0;
        strategy_account.total_trades = 0;
        strategy_account.profitable_trades = 0;
        
        // Update agent state
        agent_account.next_strategy_id += 1;
        agent_account.strategies_count += 1;
        
        // Emit strategy creation event
        emit!(events::StrategyCreated {
            strategy_id: strategy_account.strategy_id,
            owner: authority.key(),
            model_id: params.model_id,
            timestamp: current_time,
            name: params.name,
            risk_level: params.risk_level,
        });
        
        Ok(())
    }

    /// Update an existing strategy parameters
    pub fn update_strategy(
        ctx: Context<crate::UpdateStrategy>,
        params: StrategyParams,
    ) -> Result<()> {
        let strategy_account = &mut ctx.accounts.strategy_account;
        let model_account = &ctx.accounts.model_account;
        let authority = &ctx.accounts.authority;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Validate parameters
        validation::validate_strategy_params(&params)?;
        
        // Ensure model ID matches
        if params.model_id != model_account.model_id {
            return Err(AgentError::InvalidModelForStrategy.into());
        }
        
        // Update strategy parameters
        strategy_account.params = params;
        strategy_account.last_update_time = current_time;
        
        // Emit strategy update event
        emit!(events::StrategyUpdated {
            strategy_id: strategy_account.strategy_id,
            owner: authority.key(),
            timestamp: current_time,
            name: params.name,
            risk_level: params.risk_level,
        });
        
        Ok(())
    }
}

/// Signal submission and execution instructions
pub mod signal {
    use super::*;

    /// Submit a new trading signal from an AI model
    pub fn submit_signal(
        ctx: Context<crate::SubmitSignal>,
        signal: Signal,
        verification: SignalVerification,
    ) -> Result<()> {
        let agent_account = &ctx.accounts.agent_account;
        let model_account = &mut ctx.accounts.model_account;
        let strategy_account = &ctx.accounts.strategy_account;
        let signal_account = &mut ctx.accounts.signal_account;
        let submitter = &ctx.accounts.submitter;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Validate signal
        validation::validate_signal(&signal)?;
        
        // Validate signal expiry time
        if signal.expiry_time - current_time > agent_account.max_signal_validity_period {
            return Err(AgentError::SignalValidityTooLong.into());
        }
        
        // Verify signal using model's verification key
        let verified = verify_signal(&signal, &verification, &model_account.verification_key)?;
        
        // Set signal account state
        signal_account.model_id = model_account.model_id;
        signal_account.signal_id = model_account.next_signal_id;
        signal_account.bump = *ctx.bumps.get("signal_account").unwrap();
        signal_account.strategy_id = strategy_account.strategy_id;
        signal_account.creation_time = current_time;
        signal_account.expiry_time = signal.expiry_time;
        signal_account.verified = verified;
        signal_account.executed = false;
        signal_account.execution_result = 0;
        signal_account.execution_time = 0;
        signal_account.submitter = submitter.key();
        signal_account.verification_hash = verification.hash;
        signal_account.signal = signal;
        
        // Update model state
        model_account.next_signal_id += 1;
        model_account.total_signals += 1;
        
        // Emit signal submission event
        emit!(events::SignalSubmitted {
            model_id: model_account.model_id,
            signal_id: signal_account.signal_id,
            strategy_id: strategy_account.strategy_id,
            submitter: submitter.key(),
            timestamp: current_time,
            signal_type: signal.signal_type,
            verified,
        });
        
        Ok(())
    }

    /// Execute a validated trading signal
    pub fn execute_signal(
        ctx: Context<crate::ExecuteSignal>,
        signal_id: u64,
        execution_params: ExecutionParams,
    ) -> Result<()> {
        let agent_account = &ctx.accounts.agent_account;
        let signal_account = &mut ctx.accounts.signal_account;
        let strategy_account = &mut ctx.accounts.strategy_account;
        let strategy_vault = &ctx.accounts.strategy_vault;
        let executor = &ctx.accounts.executor;
        let token_program = &ctx.accounts.token_program;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Validate that signal matches the requested ID
        if signal_account.signal_id != signal_id {
            return Err(AgentError::SignalIdMismatch.into());
        }
        
        // Validate execution parameters
        validation::validate_execution_params(&execution_params, &signal_account.signal)?;
        
        // Get current price for the trading pair
        let current_price = get_current_price(&signal_account.signal.asset_pair)?;
        
        // Execute the trade based on signal type
        let execution_result = match signal_account.signal.signal_type {
            SignalType::Buy => {
                // Implement buy logic
                // This would involve interacting with DEXes through CPI
                // For simplicity, we're just updating state here
                0 // Placeholder for actual P&L calculation
            },
            SignalType::Sell => {
                // Implement sell logic
                0 // Placeholder for actual P&L calculation
            },
            SignalType::Hold => {
                // No action needed for hold signals
                0
            },
        };
        
        // Update signal state
        signal_account.executed = true;
        signal_account.execution_result = execution_result;
        signal_account.execution_time = current_time;
        
        // Update strategy state
        strategy_account.total_trades += 1;
        if execution_result > 0 {
            strategy_account.profitable_trades += 1;
            strategy_account.total_profit += execution_result;
            
            // Update model successful signals count
            // This requires loading the model account, which we don't have in this context
            // In a real implementation, we would need to include the model account in the context
        }
        
        // Emit signal execution event
        emit!(events::SignalExecuted {
            model_id: signal_account.model_id,
            signal_id: signal_account.signal_id,
            strategy_id: strategy_account.strategy_id,
            executor: executor.key(),
            timestamp: current_time,
            execution_result,
            price: current_price,
        });
        
        Ok(())
    }
}

/// Vault management instructions for depositing and withdrawing funds
pub mod vault {
    use super::*;

    /// Deposit funds into a strategy vault
    pub fn deposit_funds(
        ctx: Context<crate::DepositFunds>,
        amount: u64,
    ) -> Result<()> {
        let agent_account = &ctx.accounts.agent_account;
        let strategy_account = &mut ctx.accounts.strategy_account;
        let strategy_vault = &ctx.accounts.strategy_vault;
        let depositor = &ctx.accounts.depositor;
        let depositor_token_account = &ctx.accounts.depositor_token_account;
        let token_program = &ctx.accounts.token_program;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Validate deposit amount
        if amount == 0 {
            return Err(AgentError::InvalidAmount.into());
        }
        
        // For initial deposit, check minimum amount
        if strategy_vault.amount == 0 && amount < agent_account.min_strategy_deposit {
            return Err(AgentError::DepositTooSmall.into());
        }
        
        // Transfer tokens to vault
        let cpi_accounts = Transfer {
            from: depositor_token_account.to_account_info(),
            to: strategy_vault.to_account_info(),
            authority: depositor.to_account_info(),
        };
        
        let cpi_ctx = CpiContext::new(
            token_program.to_account_info(),
            cpi_accounts,
        );
        
        token::transfer(cpi_ctx, amount)?;
        
        // Update strategy state if this is the initial deposit
        if strategy_account.initial_deposit == 0 {
            strategy_account.initial_deposit = amount;
            strategy_account.high_water_mark = amount;
        }
        
        // Emit deposit event
        emit!(events::FundsDeposited {
            strategy_id: strategy_account.strategy_id,
            depositor: depositor.key(),
            timestamp: current_time,
            amount,
        });
        
        Ok(())
    }

    /// Withdraw funds from a strategy vault
    pub fn withdraw_funds(
        ctx: Context<crate::WithdrawFunds>,
        amount: u64,
    ) -> Result<()> {
        let strategy_account = &mut ctx.accounts.strategy_account;
        let strategy_vault = &ctx.accounts.strategy_vault;
        let owner = &ctx.accounts.owner;
        let recipient_token_account = &ctx.accounts.recipient_token_account;
        let token_program = &ctx.accounts.token_program;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Validate withdrawal amount
        if amount == 0 {
            return Err(AgentError::InvalidAmount.into());
        }
        
        if amount > strategy_vault.amount {
            return Err(AgentError::InsufficientFunds.into());
        }
        
        // Calculate any performance fees before withdrawal
        let (performance_fee, protocol_fee) = calculate_fees(
            strategy_account,
            strategy_vault.amount,
            current_time,
        )?;
        
        // If there are fees to collect, we should do that first in a separate transaction
        if performance_fee > 0 || protocol_fee > 0 {
            return Err(AgentError::FeeCollectionRequired.into());
        }
        
        // Transfer tokens from vault to recipient
        let seeds = &[
            STRATEGY_SEED.as_bytes(),
            &strategy_account.strategy_id.to_le_bytes(),
            &[strategy_account.bump],
        ];
        let signer = &[&seeds[..]];
        
        let cpi_accounts = Transfer {
            from: strategy_vault.to_account_info(),
            to: recipient_token_account.to_account_info(),
            authority: strategy_account.to_account_info(),
        };
        
        let cpi_ctx = CpiContext::new_with_signer(
            token_program.to_account_info(),
            cpi_accounts,
            signer,
        );
        
        token::transfer(cpi_ctx, amount)?;
        
        // Update strategy state
        // If full withdrawal, reset high water mark
        if amount == strategy_vault.amount {
            strategy_account.high_water_mark = 0;
        }
        
        // Emit withdrawal event
        emit!(events::FundsWithdrawn {
            strategy_id: strategy_account.strategy_id,
            owner: owner.key(),
            timestamp: current_time,
            amount,
        });
        
        Ok(())
    }

    /// Collect fees from strategy profits
    pub fn collect_fees(
        ctx: Context<crate::CollectFees>,
        strategy_id: u64,
    ) -> Result<()> {
        let agent_account = &ctx.accounts.agent_account;
        let strategy_account = &mut ctx.accounts.strategy_account;
        let strategy_vault = &ctx.accounts.strategy_vault;
        let fee_recipient_account = &ctx.accounts.fee_recipient_account;
        let token_program = &ctx.accounts.token_program;
        let authority = &ctx.accounts.authority;
        let clock = Clock::get()?;
        let current_time = clock.unix_timestamp;
        
        // Ensure strategy ID matches
        if strategy_account.strategy_id != strategy_id {
            return Err(AgentError::StrategyIdMismatch.into());
        }
        
        // Calculate fees
        let (performance_fee, protocol_fee) = calculate_fees(
            strategy_account,
            strategy_vault.amount,
            current_time,
        )?;
        
        let total_fee = performance_fee + protocol_fee;
        
        // Skip if no fees to collect
        if total_fee == 0 {
            return Err(AgentError::NoFeesToCollect.into());
        }
        
        // Transfer fees from vault to recipient
        let seeds = &[
            STRATEGY_SEED.as_bytes(),
            &strategy_account.strategy_id.to_le_bytes(),
            &[strategy_account.bump],
        ];
        let signer = &[&seeds[..]];
        
        let cpi_accounts = Transfer {
            from: strategy_vault.to_account_info(),
            to: fee_recipient_account.to_account_info(),
            authority: strategy_account.to_account_info(),
        };
        
        let cpi_ctx = CpiContext::new_with_signer(
            token_program.to_account_info(),
            cpi_accounts,
            signer,
        );
        
        token::transfer(cpi_ctx, total_fee)?;
        
        // Update strategy state
        strategy_account.total_fees_collected += total_fee;
        strategy_account.last_profit_calculation = current_time;
        
        // Update high water mark if needed
        let current_value = strategy_vault.amount + total_fee; // Add back the fees to get the value before fee collection
        if current_value > strategy_account.high_water_mark {
            strategy_account.high_water_mark = current_value;
        }
        
        // Update agent state
        agent_account.last_fee_collection = current_time;
        
        // Emit fee collection event
        emit!(events::FeesCollected {
            strategy_id: strategy_account.strategy_id,
            collector: authority.key(),
            timestamp: current_time,
            performance_fee,
            protocol_fee,
        });
        
        Ok(())
    }
}