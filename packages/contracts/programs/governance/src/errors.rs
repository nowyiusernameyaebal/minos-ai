//! Error types for the governance program
//!
//! This module defines custom error types for the governance program,
//! providing detailed error information for debugging and user feedback.

use anchor_lang::prelude::*;

/// Errors that can occur in the governance program
#[error_code]
pub enum GovernanceError {
    /// The operation requires governance authority
    #[msg("Operation requires governance authority")]
    UnauthorizedAccess,
    
    /// The provided token mint doesn't match the governance token
    #[msg("Token mint doesn't match governance token")]
    InvalidTokenMint,
    
    /// The token account owner doesn't match the expected owner
    #[msg("Token account owner mismatch")]
    InvalidOwner,
    
    /// The delegate doesn't match the expected delegate
    #[msg("Delegate mismatch")]
    InvalidDelegate,
    
    /// The voter has insufficient tokens for the operation
    #[msg("Insufficient tokens for this operation")]
    InsufficientTokens,
    
    /// The proposal title exceeds the maximum allowed length
    #[msg("Proposal title exceeds maximum length")]
    TitleTooLong,
    
    /// The proposal description exceeds the maximum allowed length
    #[msg("Proposal description exceeds maximum length")]
    DescriptionTooLong,
    
    /// The proposal has too many actions
    #[msg("Proposal has too many actions")]
    TooManyActions,
    
    /// The proposal is not in the required state for this operation
    #[msg("Invalid proposal state for this operation")]
    InvalidProposalState,
    
    /// The proposal doesn't have any actions defined
    #[msg("Proposal must have at least one action")]
    NoProposalActions,
    
    /// The voting period has already ended
    #[msg("Voting period has ended")]
    VotingPeriodEnded,
    
    /// The voting period has not ended yet
    #[msg("Voting period has not ended yet")]
    VotingPeriodNotEnded,
    
    /// The voter has already voted on this proposal
    #[msg("Voter has already cast a vote on this proposal")]
    AlreadyVoted,
    
    /// The timelock period has not expired yet
    #[msg("Timelock period has not expired yet")]
    TimelockNotExpired,
    
    /// The quorum percentage is invalid (must be between 1 and 100)
    #[msg("Invalid quorum percentage (must be between 1 and 100)")]
    InvalidQuorumPercentage,
    
    /// The approval threshold is invalid (must be between 1 and 100)
    #[msg("Invalid approval threshold (must be between 1 and 100)")]
    InvalidApprovalThreshold,
    
    /// The voting period is invalid
    #[msg("Invalid voting period (must be between minimum and maximum)")]
    InvalidVotingPeriod,
    
    /// The timelock period is invalid
    #[msg("Invalid timelock period (must be between minimum and maximum)")]
    InvalidTimelockPeriod,
    
    /// The lock duration is invalid
    #[msg("Invalid lock duration")]
    InvalidLockDuration,
    
    /// The tokens are still locked and cannot be withdrawn
    #[msg("Tokens are still locked and cannot be withdrawn")]
    TokensStillLocked,
    
    /// The treasury has insufficient funds for the withdrawal
    #[msg("Insufficient treasury funds for this operation")]
    InsufficientTreasuryFunds,
    
    /// The proposal execution failed
    #[msg("Proposal execution failed")]
    ExecutionFailed,
    
    /// Veto authority required for this operation
    #[msg("Veto authority required for this operation")]
    RequiresVetoAuthority,
    
    /// Arithmetic overflow occurred
    #[msg("Arithmetic overflow")]
    ArithmeticOverflow,
    
    /// The account is not a valid PDA for this operation
    #[msg("Invalid PDA for this operation")]
    InvalidPda,
    
    /// The provided executor is not authorized to execute this proposal
    #[msg("Executor is not authorized to execute this proposal")]
    UnauthorizedExecutor,
    
    /// The required accounts for executing this action are missing
    #[msg("Missing required accounts for executing this action")]
    MissingRequiredAccounts,
    
    /// The proposal has expired and can no longer be executed
    #[msg("Proposal has expired and can no longer be executed")]
    ProposalExpired,
    
    /// The proposal cannot be cancelled in its current state
    #[msg("Proposal cannot be cancelled in its current state")]
    CannotCancelProposal,
    
    /// The provided action type is invalid for this proposal
    #[msg("Invalid action type for this proposal")]
    InvalidActionType,
    
    /// The provided parameter is invalid
    #[msg("Invalid parameter")]
    InvalidParameter,
    
    /// The maximum number of delegates has been reached
    #[msg("Maximum number of delegates reached")]
    MaxDelegatesReached,
    
    /// The governance configuration update is invalid
    #[msg("Invalid governance configuration update")]
    InvalidConfigUpdate,
    
    /// The proposal queue is full
    #[msg("Proposal queue is full")]
    ProposalQueueFull,
    
    /// The proposal activation threshold not met
    #[msg("Proposal activation threshold not met")]
    ActivationThresholdNotMet,
    
    /// This operation is not allowed during an active vote
    #[msg("Operation not allowed during active vote")]
    NotAllowedDuringActiveVote,
    
    /// The governance token is currently locked in vesting
    #[msg("Governance tokens are currently locked in vesting")]
    TokensInVesting,
    
    /// The instruction is currently paused
    #[msg("This instruction is currently paused")]
    InstructionPaused,
    
    /// Vote switching is not allowed
    #[msg("Vote switching is not allowed")]
    VoteSwitchingNotAllowed,
    
    /// Risk threshold exceeded for this operation
    #[msg("Risk threshold exceeded for this operation")]
    RiskThresholdExceeded,
    
    /// Insufficient voting power for this operation
    #[msg("Insufficient voting power for this operation")]
    InsufficientVotingPower,
    
    /// Invalid vote distribution
    #[msg("Invalid vote distribution")]
    InvalidVoteDistribution,
    
    /// The proposal has failed the simulation
    #[msg("Proposal failed simulation")]
    ProposalSimulationFailed,
    
    /// The operation exceeds rate limits
    #[msg("Operation exceeds rate limits")]
    RateLimitExceeded,
    
    /// The operation would create a governance deadlock
    #[msg("Operation would create a governance deadlock")]
    GovernanceDeadlock,
    
    /// The provided signature is invalid
    #[msg("Invalid signature")]
    InvalidSignature,
    
    /// The provided timestamp is invalid
    #[msg("Invalid timestamp")]
    InvalidTimestamp,
    
    /// The vote has already been counted
    #[msg("Vote already counted")]
    VoteAlreadyCounted,
    
    /// The proposal requires a higher approval threshold
    #[msg("Proposal requires a higher approval threshold")]
    RequiresHigherApprovalThreshold,
}