//! Error types for the Minos Vault program
//!
//! This module defines custom error types that can be returned by the 
//! Minos Vault program's instructions.

use anchor_lang::prelude::*;
use solana_program::program_error::ProgramError;

/// Custom error types for the Minos Vault program
#[error_code]
pub enum VaultError {
    /// 0: Invalid vault authority
    #[msg("The provided authority does not match the vault authority")]
    InvalidAuthority,

    /// 1: Invalid strategy manager
    #[msg("The provided strategy manager does not match the vault strategy manager")]
    InvalidStrategyManager,

    /// 2: Invalid fee recipient
    #[msg("The provided fee recipient does not match the vault fee recipient")]
    InvalidFeeRecipient,

    /// 3: Vault is paused
    #[msg("The vault is currently paused")]
    VaultIsPaused,

    /// 4: Vault is not paused
    #[msg("The vault is not currently paused")]
    VaultIsNotPaused,

    /// 5: Deposit amount is too small
    #[msg("Deposit amount is below the minimum required")]
    DepositTooSmall,

    /// 6: Deposit amount is too large
    #[msg("Deposit amount exceeds the maximum allowed per user")]
    DepositTooLarge,

    /// 7: Vault capacity exceeded
    #[msg("Deposit would exceed vault capacity")]
    VaultCapacityExceeded,

    /// 8: Withdrawal amount is too small
    #[msg("Withdrawal amount is too small")]
    WithdrawalTooSmall,

    /// 9: Insufficient funds for withdrawal
    #[msg("Insufficient funds for withdrawal")]
    InsufficientFunds,

    /// 10: Withdrawal during lockup period
    #[msg("Cannot withdraw during lockup period")]
    WithdrawalDuringLockup,

    /// 11: Invalid share calculation
    #[msg("Share calculation resulted in invalid value")]
    InvalidShareCalculation,

    /// 12: Invalid asset calculation
    #[msg("Asset calculation resulted in invalid value")]
    InvalidAssetCalculation,

    /// 13: Invalid fee calculation
    #[msg("Fee calculation resulted in invalid value")]
    InvalidFeeCalculation,

    /// 14: Invalid fee parameters
    #[msg("Fee parameters are invalid")]
    InvalidFeeParameters,

    /// 15: Invalid strategy params
    #[msg("Strategy parameters are invalid")]
    InvalidStrategyParams,

    /// 16: Mismatched allocations
    #[msg("Number of target assets must match number of allocations")]
    MismatchedAllocations,

    /// 17: Invalid allocations
    #[msg("Allocations must sum to 10000 basis points (100%)")]
    InvalidAllocations,

    /// 18: Strategy not found
    #[msg("Strategy not found")]
    StrategyNotFound,

    /// 19: Strategy already exists
    #[msg("Strategy with this ID already exists")]
    StrategyAlreadyExists,

    /// 20: Maximum strategies reached
    #[msg("Vault has reached maximum number of strategies")]
    MaxStrategiesReached,

    /// 21: Invalid risk profile
    #[msg("Risk profile value is invalid (must be between 1 and 5)")]
    InvalidRiskProfile,

    /// 22: Invalid vault name
    #[msg("Vault name is invalid or too long")]
    InvalidVaultName,

    /// 23: Invalid description
    #[msg("Description is invalid or too long")]
    InvalidDescription,

    /// 24: Insufficient treasury balance
    #[msg("Vault treasury has insufficient balance")]
    InsufficientTreasuryBalance,

    /// 25: Invalid token account
    #[msg("Token account is invalid for this operation")]
    InvalidTokenAccount,

    /// 26: Invalid mint
    #[msg("Token mint is invalid for this operation")]
    InvalidMint,

    /// 27: Invalid account owner
    #[msg("Account owner is invalid for this operation")]
    InvalidAccountOwner,

    /// 28: Share supply is zero
    #[msg("Share supply is zero, cannot perform operation")]
    ZeroShareSupply,

    /// 29: Treasury is empty
    #[msg("Treasury is empty, cannot perform operation")]
    EmptyTreasury,

    /// 30: Zero amount
    #[msg("Amount cannot be zero")]
    ZeroAmount,

    /// 31: Strategy execution too frequent
    #[msg("Strategy execution is too frequent")]
    StrategyExecutionTooFrequent,

    /// 32: Operation would leave account with insufficient funds
    #[msg("Operation would leave account with insufficient funds")]
    InsufficientFundsAfterOperation,

    /// 33: Oracle data is stale
    #[msg("Oracle data is stale")]
    StaleOracleData,

    /// 34: Oracle account is invalid
    #[msg("Oracle account is invalid")]
    InvalidOracleAccount,

    /// 35: Oracle price is invalid
    #[msg("Oracle price is invalid")]
    InvalidOraclePrice,

    /// 36: Invalid user profile parameters
    #[msg("User profile parameters are invalid")]
    InvalidUserProfileParams,

    /// 37: Risk tolerance is out of range
    #[msg("Risk tolerance must be between 1 and 10")]
    RiskToleranceOutOfRange,

    /// 38: Invalid rewards mint
    #[msg("Rewards mint is invalid for this operation")]
    InvalidRewardsMint,

    /// 39: Rewards period has not started
    #[msg("Rewards period has not started yet")]
    RewardsPeriodNotStarted,

    /// 40: Rewards period has ended
    #[msg("Rewards period has ended")]
    RewardsPeriodEnded,

    /// 41: No rewards to claim
    #[msg("No rewards available to claim")]
    NoRewardsToClaim,

    /// 42: Invalid reward calculation
    #[msg("Reward calculation resulted in invalid value")]
    InvalidRewardCalculation,

    /// 43: Insufficient reward supply
    #[msg("Insufficient rewards in the distribution account")]
    InsufficientRewardSupply,

    /// 44: Too many assets in strategy
    #[msg("Too many assets in strategy, maximum allowed is 20")]
    TooManyAssets,

    /// 45: Too many strategies
    #[msg("Too many strategies for vault, maximum allowed is 10")]
    TooManyStrategies,

    /// 46: Invalid time period
    #[msg("Time period is invalid")]
    InvalidTimePeriod,

    /// 47: AI Model not found
    #[msg("AI Model ID not found")]
    AIModelNotFound,

    /// 48: AI optimization is disabled
    #[msg("AI optimization is disabled for this vault")]
    AIOptimizationDisabled,

    /// 49: Not enough lamports for rent-exemption
    #[msg("Not enough lamports to make account rent-exempt")]
    NotRentExempt,

    /// 50: Data length mismatch
    #[msg("Data length mismatch during deserialization")]
    DataLengthMismatch,

    /// 51: Operation requires exclusive lock on vault
    #[msg("Operation requires exclusive lock on vault")]
    RequiresExclusiveLock,

    /// 52: Invalid parameter length
    #[msg("Parameter has invalid length")]
    InvalidParameterLength,

    /// 53: Invalid parameter value
    #[msg("Parameter has invalid value")]
    InvalidParameterValue,

    /// 54: Insufficient system fund
    #[msg("Insufficient system fund to complete operation")]
    InsufficientSystemFund,

    /// 55: Claim rewards too frequent
    #[msg("Cannot claim rewards, cooldown period not elapsed")]
    ClaimRewardsTooFrequent,

    /// 56: Transaction size limit exceeded
    #[msg("Transaction size limit exceeded")]
    TransactionSizeLimitExceeded,

    /// 57: Update too frequent
    #[msg("Update is too frequent")]
    UpdateTooFrequent,

    /// 58: Invalid signature
    #[msg("Invalid signature for this operation")]
    InvalidSignature,

    /// 59: Numerical overflow
    #[msg("Numerical overflow occurred")]
    NumericalOverflow,

    /// 60: Numerical underflow
    #[msg("Numerical underflow occurred")]
    NumericalUnderflow,

    /// 61: Unauthorized operation
    #[msg("Account is not authorized for this operation")]
    UnauthorizedOperation,

    /// 62: Invalid account initialization
    #[msg("Account initialization failed")]
    InvalidAccountInitialization,

    /// 63: Emergency mode active
    #[msg("Emergency mode is active, only certain operations are allowed")]
    EmergencyModeActive,
    
    /// 64: Emergency mode inactive
    #[msg("Emergency mode is not active")]
    EmergencyModeInactive,
    
    /// 65: Operation timed out
    #[msg("Operation timed out")]
    OperationTimeout,
    
    /// 66: Invalid closing authority
    #[msg("Invalid closing authority for this account")]
    InvalidClosingAuthority,
    
    /// 67: Feature is not enabled
    #[msg("This feature is not enabled")]
    FeatureNotEnabled,
    
    /// 68: Account already initialized
    #[msg("Account is already initialized")]
    AccountAlreadyInitialized,
    
    /// 69: Unauthorized protocol upgrade
    #[msg("Unauthorized protocol upgrade attempt")]
    UnauthorizedProtocolUpgrade,
    
    /// 70: Protocol version mismatch
    #[msg("Protocol version mismatch")]
    ProtocolVersionMismatch,
    
    /// 71: Tag limit exceeded
    #[msg("User tag limit exceeded")]
    TagLimitExceeded,
    
    /// 72: Tag is too long
    #[msg("User tag is too long")]
    TagTooLong,
    
    /// 73: Insufficient resources
    #[msg("Insufficient computational resources to complete operation")]
    InsufficientResources,
    
    /// 74: Cross-program execution failed
    #[msg("Cross-program execution failed")]
    CrossProgramExecutionFailed,
    
    /// 75: Invalid timestamp
    #[msg("Invalid timestamp value")]
    InvalidTimestamp,
    
    /// 76: Vault migration in progress
    #[msg("Vault is currently being migrated, operation not permitted")]
    VaultMigrationInProgress,
    
    /// 77: Vault immutable
    #[msg("Vault configuration is immutable")]
    VaultImmutable,
    
    /// 78: Invalid vault version
    #[msg("Invalid vault version")]
    InvalidVaultVersion,
    
    /// 79: Operation not supported
    #[msg("Operation not supported")]
    OperationNotSupported,
}

/// Convert from VaultError to ProgramError
impl From<VaultError> for ProgramError {
    fn from(e: VaultError) -> Self {
        ProgramError::Custom(e as u32)
    }
}

/// Helper functions for error handling
pub mod errors_util {
    use super::*;
    
    /// Validate that allocations sum to 100% (10000 basis points)
    pub fn validate_allocations(allocations: &[u16]) -> Result<()> {
        let sum: u32 = allocations.iter().map(|x| *x as u32).sum();
        if sum != 10000 {
            return Err(VaultError::InvalidAllocations.into());
        }
        Ok(())
    }
    
    /// Validate risk profile is between 1 and 5
    pub fn validate_risk_profile(risk_profile: u8) -> Result<()> {
        if risk_profile < 1 || risk_profile > 5 {
            return Err(VaultError::InvalidRiskProfile.into());
        }
        Ok(())
    }
    
    /// Validate user risk tolerance is between 1 and 10
    pub fn validate_risk_tolerance(risk_tolerance: u8) -> Result<()> {
        if risk_tolerance < 1 || risk_tolerance > 10 {
            return Err(VaultError::RiskToleranceOutOfRange.into());
        }
        Ok(())
    }
    
    /// Validate fee parameters are in valid ranges
    pub fn validate_fee_params(fee_config: &super::state::FeeConfig) -> Result<()> {
        // Management fee should not exceed 1000 bps (10%)
        if fee_config.management_fee_bps > 1000 {
            return Err(VaultError::InvalidFeeParameters.into());
        }
        
        // Performance fee should not exceed 3000 bps (30%)
        if fee_config.performance_fee_bps > 3000 {
            return Err(VaultError::InvalidFeeParameters.into());
        }
        
        // Withdrawal fee should not exceed if 500 bps (5%)
        if fee_config.withdrawal_fee_bps > 500 {
            return Err(VaultError::InvalidFeeParameters.into());
        }
        
        // Deposit fee should not exceed 300 bps (3%)
        if fee_config.deposit_fee_bps > 300 {
            return Err(VaultError::InvalidFeeParameters.into());
        }
        
        Ok(())
    }
    
    /// Check if strategy execution should be allowed (time-based)
    pub fn validate_strategy_execution_time(
        last_execution: i64,
        frequency: u64,
        current_time: i64,
    ) -> Result<()> {
        let time_since_last = current_time.checked_sub(last_execution)
            .ok_or(VaultError::InvalidTimestamp)?;
            
        if time_since_last < frequency as i64 {
            return Err(VaultError::StrategyExecutionTooFrequent.into());
        }
        
        Ok(())
    }
    
    /// Check if an account owner matches the expected owner
    pub fn validate_account_owner(
        account_info: &AccountInfo,
        expected_owner: &Pubkey,
    ) -> Result<()> {
        if account_info.owner != expected_owner {
            return Err(VaultError::InvalidAccountOwner.into());
        }
        Ok(())
    }
}